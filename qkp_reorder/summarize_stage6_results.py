from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_PATH = PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_summary.csv"


def parse_usage(value: Any) -> Dict[str, Any]:
    """Parse usage JSON saved in the result CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}

    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    usage_dicts = (
        df["usage"].apply(parse_usage) if "usage" in df.columns else [{}] * len(df)
    )
    df["api_prompt_tokens"] = [usage.get("prompt_tokens") for usage in usage_dicts]
    df["api_completion_tokens"] = [
        usage.get("completion_tokens") for usage in usage_dicts
    ]
    df["api_total_tokens"] = [usage.get("total_tokens") for usage in usage_dicts]

    numeric_cols = [
        "original_tokens",
        "compressed_tokens",
        "prompt_tokens",
        "compression_ratio",
        "token_saving_ratio",
        "compression_time",
        "answer_time",
        "contains_answer",
        "f1",
        "empty_prediction",
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fallback_used" in df.columns:
        df["fallback_used_numeric"] = (
            df["fallback_used"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
        )
    else:
        df["fallback_used_numeric"] = 0

    if "retry_used" in df.columns:
        df["retry_used_numeric"] = (
            df["retry_used"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
        )
    else:
        df["retry_used_numeric"] = 0

    if "error_message" in df.columns:
        df["has_error"] = (
            df["error_message"].fillna("").astype(str).str.len().gt(0).astype(int)
        )
    else:
        df["has_error"] = 0

    summary = (
        df.groupby("method")
        .agg(
            rows=("sample_id", "count"),
            unique_samples=("sample_id", "nunique"),
            avg_original_tokens=("original_tokens", "mean"),
            avg_compressed_tokens=("compressed_tokens", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
            avg_token_saving_ratio=("token_saving_ratio", "mean"),
            avg_compression_time=("compression_time", "mean"),
            avg_answer_time=("answer_time", "mean"),
            contains_answer_rate=("contains_answer", "mean"),
            avg_f1=("f1", "mean"),
            empty_prediction_count=("empty_prediction", "sum"),
            fallback_used_count=("fallback_used_numeric", "sum"),
            retry_used_count=("retry_used_numeric", "sum"),
            error_count=("has_error", "sum"),
            avg_api_prompt_tokens=("api_prompt_tokens", "mean"),
            avg_api_completion_tokens=("api_completion_tokens", "mean"),
            avg_api_total_tokens=("api_total_tokens", "mean"),
        )
        .reset_index()
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")

    print("Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

