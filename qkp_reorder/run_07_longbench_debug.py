from __future__ import annotations

import json
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
    PROJECT_ROOT / "data" / "stage7" / "processed" / "longbench_debug_3x10.json"
)
RESULT_DIR = PROJECT_ROOT / "results" / "stage7"
RESULT_PATH = RESULT_DIR / "longbench_debug_results.csv"

METHODS = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]

TOKEN_BUDGET = 2000
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5


def get_max_samples_to_run() -> int | None:
    """Read optional smoke-test sample limit from STAGE7_MAX_SAMPLES."""
    value = os.getenv("STAGE7_MAX_SAMPLES")
    if not value or not value.strip():
        return None
    return int(value)


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
    task_name = sample.get("task_name", "")
    task_display_name = sample.get("task_display_name", task_name)
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "experiment": "exp3_longbench_debug",
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


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_multidoc_qa(DATA_PATH)
    max_samples_to_run = get_max_samples_to_run()
    if max_samples_to_run is not None:
        samples = samples[:max_samples_to_run]

    existing_df = load_existing_results()
    completed_pairs = get_completed_pairs(existing_df)

    print(f"Loaded {len(samples)} samples.")
    print(f"Existing rows: {len(existing_df)}")
    print(f"Valid completed pairs: {len(completed_pairs)}")
    print(f"Result path: {RESULT_PATH}")

    compressor = ContextCompressor(device_map="cpu")
    total_planned = len(samples) * len(METHODS)
    current_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]

        print("\n" + "#" * 100)
        print(f"Sample: {sample_id}")
        print(f"Task: {sample.get('task_name')}")
        print(f"Question: {sample['question']}")
        print(f"Context length field: {sample.get('length')}")

        for method in METHODS:
            current_count += 1
            pair = (sample_id, method)

            if pair in completed_pairs:
                print(
                    f"[{current_count}/{total_planned}] "
                    f"Skip completed: {sample_id} - {method}"
                )
                continue

            print("\n" + "=" * 80)
            print(f"[{current_count}/{total_planned}] Running method: {method}")

            row = run_one_method(
                compressor=compressor,
                sample=sample,
                method=method,
            )
            append_row(row)

            print("compressed_tokens:", row["compressed_tokens"])
            print("prompt_tokens:", row["prompt_tokens"])
            print("fallback_used:", row["fallback_used"])
            print("retry_used:", row["retry_used"])
            print("empty_prediction:", row["empty_prediction"])
            print("contains_answer:", row["contains_answer"])
            print("f1:", row["f1"])
            print("error_message:", row["error_message"])

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n" + "#" * 100)
    print("Stage 7A LongBench debug experiment finished.")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()

