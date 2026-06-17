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
RESULT_PATH = RESULT_DIR / "ours_budget_sensitivity_results.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "innovation"
LOG_PATH = LOG_DIR / "stage7_ours_budget_sensitivity_log.txt"

_DEFAULT_BUDGETS = "500,750,1000,1500,2000"

METHOD = "ours_full"
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5


def _get_env_budgets() -> List[int]:
    raw = os.getenv("STAGE7_BUDGETS", _DEFAULT_BUDGETS)
    return [int(b.strip()) for b in raw.split(",") if b.strip()]


def get_max_samples_to_run() -> int | None:
    value = os.getenv("STAGE7_MAX_SAMPLES")
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
    """Use (sample_id, str(token_budget)) as key."""
    if df.empty or "sample_id" not in df.columns or "token_budget" not in df.columns:
        return set()

    completed = set()
    for _, row in df.iterrows():
        if is_valid_completed_row(row):
            completed.add((str(row["sample_id"]), str(row["token_budget"])))
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
    token_budget: int,
) -> Dict[str, Any]:
    sample_id = sample["sample_id"]
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    answer_position = get_answer_position(sample)
    inferred_answer_position = sample.get("inferred_answer_position")
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "experiment": "exp7_budget_sensitivity",
        "sample_id": sample_id,
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "answer_position": answer_position,
        "inferred_answer_position": inferred_answer_position,
        "num_documents": len(sample["documents"]),
        "source": sample.get("source", ""),
        "method": METHOD,
        "token_budget": token_budget,
    }

    try:
        compression_result = compressor.compress_context(
            context=context,
            question=question,
            method=METHOD,
            token_budget=token_budget,
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

    budgets = _get_env_budgets()
    all_samples = load_multidoc_qa(DATA_PATH)

    max_samples_to_run = get_max_samples_to_run()
    if max_samples_to_run is not None:
        samples = sample_by_position(all_samples, max_samples_to_run)
        logging.info(
            "Debug mode: max_samples=%s, sampled %s across positions",
            max_samples_to_run,
            len(samples),
        )
    else:
        samples = all_samples

    existing_df = load_existing_results()
    completed_pairs = get_completed_pairs(existing_df)

    logging.info("Method: %s", METHOD)
    logging.info("Budgets to run: %s", budgets)
    logging.info("Loaded %s samples (from %s total).", len(samples), len(all_samples))
    logging.info("Existing rows: %s", len(existing_df))
    logging.info("Valid completed pairs: %s", len(completed_pairs))
    logging.info("Result path: %s", RESULT_PATH)
    logging.info("Log path: %s", LOG_PATH)

    compressor = ContextCompressor(device_map="cpu")
    total_planned = len(samples) * len(budgets)
    current_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]

        logging.info("#" * 60)
        logging.info("Sample: %s", sample_id)
        logging.info("Question: %s", sample["question"])
        logging.info("Answer position: %s", get_answer_position(sample))
        logging.info("Inferred answer position: %s", sample.get("inferred_answer_position"))
        logging.info("Num documents: %s", len(sample["documents"]))

        for budget in budgets:
            current_count += 1
            pair = (sample_id, str(budget))

            if pair in completed_pairs:
                logging.info(
                    "[%s/%s] Skip completed: %s - budget %s",
                    current_count, total_planned, sample_id, budget,
                )
                continue

            logging.info("=" * 40)
            logging.info(
                "[%s/%s] Running budget: %s", current_count, total_planned, budget,
            )

            row = run_one_method(
                compressor=compressor,
                sample=sample,
                token_budget=budget,
            )
            append_row(row)

            logging.info("compressed_tokens: %s", row["compressed_tokens"])
            logging.info("prompt_tokens: %s", row["prompt_tokens"])
            logging.info("fallback_used: %s", row["fallback_used"])
            logging.info("retry_used: %s", row["retry_used"])
            logging.info("empty_prediction: %s", row["empty_prediction"])
            logging.info("contains_answer: %s", row["contains_answer"])
            logging.info("f1: %s", row["f1"])
            logging.info("error_message: %s", row["error_message"])

            time.sleep(SLEEP_BETWEEN_CALLS)

    logging.info("#" * 60)
    logging.info("Stage 7 budget sensitivity experiment finished.")
    logging.info("Method: %s", METHOD)
    logging.info("Budgets: %s", budgets)
    logging.info("Samples attempted: %s", len(samples))
    logging.info("Results saved to: %s", RESULT_PATH)
    logging.info("Log saved to: %s", LOG_PATH)


if __name__ == "__main__":
    main()

