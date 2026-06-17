from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import ContextCompressor
from qkp_reorder.data_loader_multidoc import (
    format_documents_as_context,
    get_answer_position,
    load_multidoc_qa,
)
from qkp_reorder.evaluate import contains_answer, simple_token_f1
from qkp_reorder.llm_inference import generate_answer
from qkp_reorder.prompt_builder import build_qa_prompt
from qkp_reorder.tokenizer_utils import count_tokens


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "stage6_position"
    / "processed"
    / "nq_position_5x20.json"
)
RESULT_DIR = PROJECT_ROOT / "results" / "innovation"
RESULT_PATH = RESULT_DIR / "ours_position_results.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "innovation"
LOG_PATH = LOG_DIR / "stage5_ours_position_log.txt"

_DEFAULT_METHODS = "ours_full"

TOKEN_BUDGET = 1500
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5


def _get_env_methods() -> List[str]:
    raw = os.getenv("STAGE5_METHODS", _DEFAULT_METHODS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_max_samples_to_run() -> int | None:
    value = os.getenv("STAGE5_MAX_SAMPLES")
    if not value or not value.strip():
        return None
    return int(value)


def sample_by_position(samples: List[Dict], max_total: int) -> List[Dict]:
    """Sample evenly across answer_position so debug covers all positions."""
    from collections import defaultdict

    by_pos: Dict[int, List[Dict]] = defaultdict(list)
    for s in samples:
        pos = get_answer_position(s)
        if pos is not None:
            by_pos[pos].append(s)

    positions = sorted(by_pos.keys())
    per_pos = max(1, max_total // len(positions))

    sampled = []
    for pos in positions:
        sampled.extend(by_pos[pos][:per_pos])

    return sampled


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_existing_results() -> pd.DataFrame:
    if RESULT_PATH.exists():
        return pd.read_csv(RESULT_PATH)
    return pd.DataFrame()


def is_valid_completed_row(row: pd.Series) -> bool:
    raw_error = row.get("error_message", "")
    raw_prediction = row.get("prediction", "")
    error_message = "" if pd.isna(raw_error) else str(raw_error)
    prediction = "" if pd.isna(raw_prediction) else str(raw_prediction)
    empty_prediction = row.get("empty_prediction", 0)

    try:
        empty_prediction = int(empty_prediction)
    except Exception:
        empty_prediction = 0

    return (not error_message.strip()) and bool(prediction.strip()) and empty_prediction == 0


def get_completed_pairs(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    if df.empty or "sample_id" not in df.columns or "method" not in df.columns:
        return set()

    completed = set()
    for _, row in df.iterrows():
        if is_valid_completed_row(row):
            completed.add((str(row["sample_id"]), str(row["method"])))
    return completed


def append_row(row: Dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])

    if RESULT_PATH.exists():
        row_df.to_csv(
            RESULT_PATH,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
    else:
        row_df.to_csv(
            RESULT_PATH,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8",
        )


def call_generate_answer_with_retry(prompt: str) -> Dict[str, Any]:
    result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    answer = result.get("answer") or ""
    if answer.strip():
        result["retry_used"] = False
        return result

    retry_result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=ANSWER_RETRY_MAX_TOKENS,
    )
    retry_result["retry_used"] = True
    return retry_result


def run_one_method(
    compressor: ContextCompressor,
    sample: Dict[str, Any],
    method: str,
) -> Dict[str, Any]:
    sample_id = sample["sample_id"]
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    answer_position = get_answer_position(sample)
    inferred_answer_position = sample.get("inferred_answer_position")
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "experiment": "exp2_position",
        "sample_id": sample_id,
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "answer_position": answer_position,
        "inferred_answer_position": inferred_answer_position,
        "num_documents": len(sample["documents"]),
        "source": sample.get("source", ""),
        "method": method,
        "token_budget": TOKEN_BUDGET,
    }

    try:
        compression_result = compressor.compress_context(
            context=context,
            question=question,
            method=method,
            token_budget=TOKEN_BUDGET,
        )
        compressed_context = compression_result["compressed_context"]
        prompt = build_qa_prompt(compressed_context, question)
        prompt_tokens = count_tokens(prompt)

        generation_result = call_generate_answer_with_retry(prompt)
        prediction = generation_result.get("answer") or ""
        empty_prediction = int(not prediction.strip())

        return {
            **base_row,
            "original_tokens": compression_result["original_tokens"],
            "compressed_tokens": compression_result["compressed_tokens"],
            "prompt_tokens": prompt_tokens,
            "compression_ratio": compression_result["compression_ratio"],
            "token_saving_ratio": compression_result["token_saving_ratio"],
            "compression_time": compression_result["compression_time"],
            "fallback_used": compression_result.get("fallback_used", False),
            "answer_model": generation_result["model"],
            "answer_time": generation_result["answer_time"],
            "retry_used": generation_result.get("retry_used", False),
            "prediction": prediction,
            "empty_prediction": empty_prediction,
            "contains_answer": contains_answer(prediction, gold_answers),
            "f1": simple_token_f1(prediction, gold_answers),
            "usage": json.dumps(generation_result["usage"], ensure_ascii=False),
            "error_message": "",
        }
    except Exception as exc:
        return {
            **base_row,
            "original_tokens": None,
            "compressed_tokens": None,
            "prompt_tokens": None,
            "compression_ratio": None,
            "token_saving_ratio": None,
            "compression_time": None,
            "fallback_used": None,
            "answer_model": None,
            "answer_time": None,
            "retry_used": None,
            "prediction": "",
            "empty_prediction": 1,
            "contains_answer": 0,
            "f1": 0.0,
            "usage": None,
            "error_message": repr(exc),
        }


def main() -> None:
    setup_logging()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    methods = _get_env_methods()
    all_samples = load_multidoc_qa(DATA_PATH)

    max_samples_to_run = get_max_samples_to_run()
    if max_samples_to_run is not None:
        samples = sample_by_position(all_samples, max_samples_to_run)
        logging.info(
            f"Debug mode: max_samples={max_samples_to_run}, "
            f"sampled {len(samples)} across positions"
        )
    else:
        samples = all_samples

    existing_df = load_existing_results()
    completed_pairs = get_completed_pairs(existing_df)

    logging.info(f"Methods to run: {methods}")
    logging.info(f"Loaded {len(samples)} samples (from {len(all_samples)} total).")
    logging.info(f"Existing rows: {len(existing_df)}")
    logging.info(f"Valid completed pairs: {len(completed_pairs)}")
    logging.info(f"Result path: {RESULT_PATH}")
    logging.info(f"Log path: {LOG_PATH}")

    compressor = ContextCompressor(device_map="cpu")
    total_planned = len(samples) * len(methods)
    current_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]

        logging.info("#" * 60)
        logging.info(f"Sample: {sample_id}")
        logging.info(f"Question: {sample['question']}")
        logging.info(f"Answer position: {get_answer_position(sample)}")
        logging.info(f"Inferred answer position: {sample.get('inferred_answer_position')}")
        logging.info(f"Num documents: {len(sample['documents'])}")

        for method in methods:
            current_count += 1
            pair = (sample_id, method)

            if pair in completed_pairs:
                logging.info(
                    f"[{current_count}/{total_planned}] "
                    f"Skip completed: {sample_id} - {method}"
                )
                continue

            logging.info("=" * 40)
            logging.info(f"[{current_count}/{total_planned}] Running method: {method}")

            row = run_one_method(
                compressor=compressor,
                sample=sample,
                method=method,
            )
            append_row(row)

            logging.info(f"compressed_tokens: {row['compressed_tokens']}")
            logging.info(f"prompt_tokens: {row['prompt_tokens']}")
            logging.info(f"fallback_used: {row['fallback_used']}")
            logging.info(f"retry_used: {row['retry_used']}")
            logging.info(f"empty_prediction: {row['empty_prediction']}")
            logging.info(f"contains_answer: {row['contains_answer']}")
            logging.info(f"f1: {row['f1']}")
            logging.info(f"error_message: {row['error_message']}")

            time.sleep(SLEEP_BETWEEN_CALLS)

    logging.info("#" * 60)
    logging.info("Stage 5 position innovation experiment finished.")
    logging.info(f"Methods: {methods}")
    logging.info(f"Samples attempted: {len(samples)}")
    logging.info(f"Results saved to: {RESULT_PATH}")
    logging.info(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()

