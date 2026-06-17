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
from qkp_reorder.data_loader_multidoc import format_documents_as_context, load_multidoc_qa
from qkp_reorder.evaluate import contains_answer, simple_token_f1
from qkp_reorder.llm_inference import generate_answer
from qkp_reorder.prompt_builder import build_qa_prompt
from qkp_reorder.tokenizer_utils import count_tokens


DATA_PATH = (
    PROJECT_ROOT / "data" / "stage7" / "processed" / "longbench_small_3x50.json"
)
RESULT_DIR = PROJECT_ROOT / "results" / "innovation"
RESULT_PATH = RESULT_DIR / "ours_longbench_small_results.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "innovation"
LOG_PATH = LOG_DIR / "stage6_ours_longbench_log.txt"

_DEFAULT_METHODS = "ours_full"

TOKEN_BUDGET = 2000
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5


def _get_env_methods() -> List[str]:
    raw = os.getenv("STAGE6_METHODS", _DEFAULT_METHODS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_max_samples_to_run() -> int | None:
    value = os.getenv("STAGE6_MAX_SAMPLES")
    if not value or not value.strip():
        return None
    return int(value)


def is_rerun_empty_mode() -> bool:
    return os.getenv("STAGE6_RERUN_EMPTY", "").strip() == "1"


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


def call_generate_answer_with_extra_retries(prompt: str) -> Dict[str, Any]:
    """Retry up to 3 times with escalating max_tokens for stubborn empty predictions."""
    token_limits = [ANSWER_MAX_TOKENS, ANSWER_RETRY_MAX_TOKENS, 8192]
    retry_count = 0

    for i, max_tok in enumerate(token_limits):
        result = generate_answer(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tok,
        )
        answer = result.get("answer") or ""
        if answer.strip():
            result["retry_used"] = (i > 0)
            result["retry_attempts"] = i + 1
            return result
        retry_count = i

    result["retry_used"] = True
    result["retry_attempts"] = retry_count + 1
    return result


def run_one_method(
    compressor: ContextCompressor,
    sample: Dict[str, Any],
    method: str,
) -> Dict[str, Any]:
    sample_id = sample["sample_id"]
    task_name = sample.get("task_name", "")
    task_display_name = sample.get("task_display_name", task_name)
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "experiment": "exp_innovation_longbench_small",
        "task_name": task_name,
        "task_display_name": task_display_name,
        "sample_id": sample_id,
        "raw_id": sample.get("raw_id", ""),
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "length": sample.get("length"),
        "language": sample.get("language", ""),
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


def run_one_method_rerun(
    compressor: ContextCompressor,
    sample: Dict[str, Any],
    method: str,
) -> Dict[str, Any]:
    """Like run_one_method but uses extra retries for stubborn empty predictions."""
    sample_id = sample["sample_id"]
    task_name = sample.get("task_name", "")
    task_display_name = sample.get("task_display_name", task_name)
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "experiment": "exp_innovation_longbench_small",
        "task_name": task_name,
        "task_display_name": task_display_name,
        "sample_id": sample_id,
        "raw_id": sample.get("raw_id", ""),
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "length": sample.get("length"),
        "language": sample.get("language", ""),
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

        generation_result = call_generate_answer_with_extra_retries(prompt)
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


def _debug_sample_across_tasks(samples: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    """Select up to max_n samples evenly distributed across tasks."""
    if max_n <= 0 or max_n >= len(samples):
        return samples

    task_samples: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        task = s.get("task_name", "unknown")
        task_samples.setdefault(task, []).append(s)

    tasks = sorted(task_samples.keys())
    per_task = max(1, max_n // len(tasks))

    selected: List[Dict[str, Any]] = []
    for task in tasks:
        selected.extend(task_samples[task][:per_task])

    # If we selected more than max_n due to rounding, trim
    return selected[:max_n]


def rerun_empty_samples() -> None:
    """Rerun only samples with empty_prediction=1, replace rows in-place in the CSV."""
    setup_logging()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULT_PATH.exists():
        logging.error("Result CSV not found; nothing to rerun.")
        return

    df = pd.read_csv(RESULT_PATH)
    logging.info(f"Loaded {len(df)} rows from {RESULT_PATH}")

    empty_mask = pd.to_numeric(df["empty_prediction"], errors="coerce").fillna(1).astype(int) == 1
    empty_indices = df.index[empty_mask].tolist()

    if not empty_indices:
        logging.info("No empty predictions found. Nothing to rerun.")
        return

    logging.info(f"Found {len(empty_indices)} rows with empty_prediction=1")

    all_samples = load_multidoc_qa(DATA_PATH)
    sample_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in all_samples:
        sample_map[(s["sample_id"], s.get("task_name", ""))] = s

    compressor = ContextCompressor(device_map="cpu")
    replaced = 0

    for idx in empty_indices:
        row = df.iloc[idx]
        sample_id = str(row["sample_id"])
        task_name = str(row.get("task_name", ""))
        method = str(row.get("method", "ours_full"))
        key = (sample_id, task_name)

        if key not in sample_map:
            logging.warning(f"Sample not found in data: {key}")
            continue

        logging.info(f"Rerunning empty sample: {sample_id} ({task_name})")
        new_row = run_one_method_rerun(
            compressor=compressor,
            sample=sample_map[key],
            method=method,
        )

        logging.info(
            "result: empty_prediction=%s, contains_answer=%s, f1=%s, "
            "retry_used=%s, retry_attempts=%s",
            new_row["empty_prediction"],
            new_row["contains_answer"],
            new_row["f1"],
            new_row.get("retry_used"),
            new_row.get("retry_attempts", "N/A"),
        )

        if new_row["empty_prediction"]:
            logging.warning(
                f"Rerun still empty for {sample_id}; keeping old row but "
                "recording attempt. prediction='%s'",
                str(new_row.get("prediction", ""))[:120],
            )

        # Replace row values in-place
        for col, val in new_row.items():
            df.at[idx, col] = val
        replaced += 1

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Write back the full 150-row CSV atomically
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
    logging.info(
        f"Rerun complete: replaced {replaced} rows. "
        f"Empty predictions remaining: "
        f"{pd.to_numeric(df['empty_prediction'], errors='coerce').fillna(1).astype(int).sum()}"
    )
    logging.info(f"CSV rewritten: {RESULT_PATH}")


def main() -> None:
    # ── Rerun-empty mode: fix empty predictions in-place ──
    if is_rerun_empty_mode():
        rerun_empty_samples()
        return

    setup_logging()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    methods = _get_env_methods()
    all_samples = load_multidoc_qa(DATA_PATH)
    max_samples_to_run = get_max_samples_to_run()

    if max_samples_to_run is not None:
        samples = _debug_sample_across_tasks(all_samples, max_samples_to_run)
    else:
        samples = all_samples

    existing_df = load_existing_results()
    completed_pairs = get_completed_pairs(existing_df)

    logging.info(f"Methods to run: {methods}")
    logging.info(f"Loaded {len(samples)} samples (total available: {len(all_samples)}).")
    logging.info(f"Existing rows: {len(existing_df)}")
    logging.info(f"Valid completed pairs: {len(completed_pairs)}")
    logging.info(f"Result path: {RESULT_PATH}")
    logging.info(f"Log path: {LOG_PATH}")

    compressor = ContextCompressor(device_map="cpu")
    total_planned = len(samples) * len(methods)
    current_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]

        logging.info("#" * 100)
        logging.info(f"Sample: {sample_id}")
        logging.info(f"Task: {sample.get('task_name')}")
        logging.info(f"Question: {sample['question']}")
        logging.info(f"Context length field: {sample.get('length')}")

        for method in methods:
            current_count += 1
            pair = (sample_id, method)

            if pair in completed_pairs:
                logging.info(
                    f"[{current_count}/{total_planned}] "
                    f"Skip completed: {sample_id} - {method}"
                )
                continue

            logging.info("=" * 80)
            logging.info(f"[{current_count}/{total_planned}] Running method: {method}")

            row = run_one_method(
                compressor=compressor,
                sample=sample,
                method=method,
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

    logging.info("#" * 100)
    logging.info("Stage 6 LongBench small innovation experiment finished.")
    logging.info(f"Methods: {methods}")
    logging.info(f"Samples attempted: {len(samples)}")
    logging.info(f"Results saved to: {RESULT_PATH}")
    logging.info(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()

