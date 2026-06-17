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
from qkp_reorder.data_loader_multidoc import (
    format_documents_as_context,
    get_answer_position,
    load_multidoc_qa,
)
from qkp_reorder.evaluate import contains_answer, simple_token_f1
from qkp_reorder.llm_inference import generate_answer
from qkp_reorder.prompt_builder import build_qa_prompt
from qkp_reorder.tokenizer_utils import count_tokens


DATA_PATH = PROJECT_ROOT / "data" / "stage5" / "processed" / "nq_multidoc_small_20.json"
RESULT_DIR = PROJECT_ROOT / "results" / "stage5"
RESULT_PATH = RESULT_DIR / "exp1_nq_small_results.csv"

METHODS = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]

TOKEN_BUDGET = 750
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5


def get_max_samples_to_run() -> int | None:
    """Read optional smoke-test sample limit from STAGE5_MAX_SAMPLES."""
    value = os.getenv("STAGE5_MAX_SAMPLES")
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    return int(value)


def load_existing_results() -> pd.DataFrame:
    """Load existing result CSV when present."""
    if RESULT_PATH.exists():
        return pd.read_csv(RESULT_PATH)
    return pd.DataFrame()


def get_completed_pairs(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Return completed (sample_id, method) pairs without non-empty errors."""
    if df.empty or "sample_id" not in df.columns or "method" not in df.columns:
        return set()

    if "error_message" in df.columns:
        error_mask = df["error_message"].fillna("").astype(str).str.len().gt(0)
        df = df[~error_mask]

    if "empty_prediction" in df.columns:
        empty_mask = (
            pd.to_numeric(df["empty_prediction"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        df = df[~empty_mask]

    return set(zip(df["sample_id"].astype(str), df["method"].astype(str)))


def append_row(row: Dict[str, Any]) -> None:
    """Append one row to the result CSV immediately."""
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


def generate_answer_with_empty_retry(prompt: str) -> Dict[str, Any]:
    """Retry once with a larger budget if DeepSeek returns empty final content."""
    result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    answer = result.get("answer") or ""
    if answer.strip():
        result["empty_prediction_retry"] = False
        return result

    retry_result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=ANSWER_RETRY_MAX_TOKENS,
    )
    retry_result["empty_prediction_retry"] = True
    return retry_result


def run_one_method(
    compressor: ContextCompressor,
    sample: Dict[str, Any],
    method: str,
) -> Dict[str, Any]:
    """Run one sample-method pair and return one CSV row."""
    sample_id = sample["sample_id"]
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]
    answer_position = get_answer_position(sample)
    context = format_documents_as_context(sample["documents"])

    base_row = {
        "sample_id": sample_id,
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "answer_position": answer_position,
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

        generation_result = generate_answer_with_empty_retry(prompt)
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
            "prediction": prediction,
            "empty_prediction": empty_prediction,
            "empty_prediction_retry": generation_result.get(
                "empty_prediction_retry", False
            ),
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
            "prediction": "",
            "empty_prediction": 1,
            "empty_prediction_retry": False,
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
    print(f"Existing completed pairs: {len(completed_pairs)}")
    print(f"Result path: {RESULT_PATH}")

    compressor = ContextCompressor(device_map="cpu")
    total_planned = len(samples) * len(METHODS)
    current_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]

        print("\n" + "#" * 100)
        print(f"Sample: {sample_id}")
        print(f"Question: {sample['question']}")
        print(f"Answer position: {sample.get('answer_position')}")
        print(f"Num documents: {len(sample['documents'])}")

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
            print("empty_prediction:", row["empty_prediction"])
            print("empty_prediction_retry:", row["empty_prediction_retry"])
            print("contains_answer:", row["contains_answer"])
            print("f1:", row["f1"])
            print("error_message:", row["error_message"])

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n" + "#" * 100)
    print("Stage 5 small NQ experiment finished.")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()

