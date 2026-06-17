from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

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


DATA_PATH = PROJECT_ROOT / "data" / "stage4" / "processed" / "nq_multidoc_debug.json"
RESULT_DIR = PROJECT_ROOT / "results" / "stage4"
RESULT_PATH = RESULT_DIR / "stage4_nq_debug_results.csv"

METHODS = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]

TOKEN_BUDGET = 750
ANSWER_MAX_TOKENS = 1024
ANSWER_RETRY_MAX_TOKENS = 2048


def make_base_row(
    sample: Dict[str, Any],
    method: str,
    token_budget: int,
    answer_position: int | None,
) -> Dict[str, Any]:
    """Create a stable stage-4 CSV row skeleton."""
    return {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "gold_answers": json.dumps(sample["gold_answers"], ensure_ascii=False),
        "answer_position": answer_position,
        "num_documents": len(sample["documents"]),
        "method": method,
        "token_budget": token_budget,
        "original_tokens": "",
        "compressed_tokens": "",
        "prompt_tokens": "",
        "compression_ratio": "",
        "token_saving_ratio": "",
        "compression_time": "",
        "fallback_used": "",
        "answer_model": "",
        "answer_time": "",
        "prediction": "",
        "contains_answer": "",
        "f1": "",
        "usage": "",
        "empty_prediction_retry": False,
        "error_message": "",
        "source": json.dumps(sample.get("source", {}), ensure_ascii=False),
    }


def generate_answer_with_empty_retry(prompt: str) -> Dict[str, Any]:
    """Retry once with a larger output budget if the model returns empty content."""
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
    context: str,
    method: str,
    answer_position: int | None,
) -> Dict[str, Any]:
    """Run one sample-method pair end to end."""
    row = make_base_row(sample, method, TOKEN_BUDGET, answer_position)
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]

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

    row.update(
        {
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
            "contains_answer": contains_answer(prediction, gold_answers),
            "f1": simple_token_f1(prediction, gold_answers),
            "usage": json.dumps(generation_result["usage"], ensure_ascii=False),
            "empty_prediction_retry": generation_result.get(
                "empty_prediction_retry", False
            ),
        }
    )
    return row


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_multidoc_qa(DATA_PATH)
    compressor = ContextCompressor(device_map="cpu")
    rows = []

    for sample in samples:
        sample_id = sample["sample_id"]
        question = sample["question"]
        answer_position = get_answer_position(sample)
        context = format_documents_as_context(sample["documents"])

        print("\n" + "#" * 100)
        print(f"Sample: {sample_id}")
        print(f"Question: {question}")
        print(f"Answer position: {answer_position}")
        print(f"Num documents: {len(sample['documents'])}")

        for method in METHODS:
            print("\n" + "=" * 80)
            print(f"Running method: {method}")

            try:
                row = run_one_method(
                    compressor=compressor,
                    sample=sample,
                    context=context,
                    method=method,
                    answer_position=answer_position,
                )
            except Exception as exc:
                row = make_base_row(sample, method, TOKEN_BUDGET, answer_position)
                row["error_message"] = str(exc)
                print("error_message:", row["error_message"])
            else:
                print("compressed_tokens:", row["compressed_tokens"])
                print("prompt_tokens:", row["prompt_tokens"])
                print("fallback_used:", row["fallback_used"])
                print("empty_prediction_retry:", row["empty_prediction_retry"])
                print("prediction:", row["prediction"])
                print("contains_answer:", row["contains_answer"])
                print("f1:", round(row["f1"], 3))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")

    print("\n" + "#" * 100)
    print(f"Saved stage 4 results to: {RESULT_PATH}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()

