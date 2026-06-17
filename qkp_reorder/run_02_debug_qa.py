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
from qkp_reorder.evaluate import contains_answer, simple_token_f1
from qkp_reorder.llm_inference import generate_answer
from qkp_reorder.prompt_builder import build_qa_prompt
from qkp_reorder.tokenizer_utils import count_tokens


RESULT_DIR = PROJECT_ROOT / "results" / "debug"
RESULT_PATH = RESULT_DIR / "debug_results.csv"
ANSWER_MAX_TOKENS = 1024


def build_debug_sample() -> Dict[str, Any]:
    """Build one temporary QA sample for the stage-2 full pipeline test."""
    context = """
    Document 1:
    Prompt compression is a technique that reduces the length of input prompts
    before sending them to a large language model. It can reduce inference cost
    and latency.

    Document 2:
    LongLLMLingua is a prompt compression method designed for long-context
    question answering. It uses question-aware compression to preserve information
    that is relevant to the user question.

    Document 3:
    The lost in the middle problem refers to the observation that large language
    models may fail to use relevant information when it is placed in the middle
    of a long context. LongLLMLingua considers document reordering to mitigate
    this problem.

    Document 4:
    BM25 is a traditional lexical retrieval method. It scores documents or
    sentences based on term matching with a query.

    Document 5:
    Truncation is a simple baseline that keeps the beginning of a context and
    discards the remaining content after a fixed token budget.
    """

    return {
        "sample_id": "debug_001",
        "question": "What problem does LongLLMLingua try to mitigate?",
        "gold_answers": [
            "lost in the middle",
            "lost in the middle problem",
        ],
        "context": context,
    }


def build_error_row(
    sample: Dict[str, Any],
    method: str,
    token_budget: int,
    error: Exception,
) -> Dict[str, Any]:
    """Create a CSV row for a failed method while preserving the run."""
    return {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "gold_answers": json.dumps(sample["gold_answers"], ensure_ascii=False),
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
        "error_message": str(error),
    }


def run_method(
    compressor: ContextCompressor,
    sample: Dict[str, Any],
    method: str,
    token_budget: int,
) -> Dict[str, Any]:
    """Run compression, answer generation, and evaluation for one method."""
    context = sample["context"]
    question = sample["question"]
    gold_answers: List[str] = sample["gold_answers"]

    compression_result = compressor.compress_context(
        context=context,
        question=question,
        method=method,
        token_budget=token_budget,
    )

    compressed_context = compression_result["compressed_context"]
    prompt = build_qa_prompt(compressed_context, question)
    prompt_tokens = count_tokens(prompt)

    generation_result = generate_answer(
        prompt=prompt,
        temperature=0.0,
        max_tokens=ANSWER_MAX_TOKENS,
    )

    prediction = generation_result["answer"]
    ca = contains_answer(prediction, gold_answers)
    f1 = simple_token_f1(prediction, gold_answers)

    return {
        "sample_id": sample["sample_id"],
        "question": question,
        "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
        "method": method,
        "token_budget": token_budget,
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
        "contains_answer": ca,
        "f1": f1,
        "usage": json.dumps(generation_result["usage"], ensure_ascii=False),
        "error_message": "",
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sample = build_debug_sample()
    methods = [
        "original",
        "truncate",
        "bm25",
        "llmlingua",
        "longllmlingua",
    ]
    token_budget = 120
    compressor = ContextCompressor(device_map="cpu")
    rows = []

    for method in methods:
        print("\n" + "=" * 80)
        print(f"Running method: {method}")

        try:
            row = run_method(
                compressor=compressor,
                sample=sample,
                method=method,
                token_budget=token_budget,
            )
        except Exception as exc:
            row = build_error_row(sample, method, token_budget, exc)
            print("error_message:", row["error_message"])
        else:
            print("compressed_tokens:", row["compressed_tokens"])
            print("prompt_tokens:", row["prompt_tokens"])
            print("fallback_used:", row["fallback_used"])
            print("prediction:", row["prediction"])
            print("contains_answer:", row["contains_answer"])
            print("f1:", round(row["f1"], 3))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Saved debug results to: {RESULT_PATH}")


if __name__ == "__main__":
    main()

