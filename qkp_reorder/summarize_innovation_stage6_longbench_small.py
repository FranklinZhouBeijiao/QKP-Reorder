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

OURS_RESULT_PATH = (
    PROJECT_ROOT / "results" / "innovation" / "ours_longbench_small_results.csv"
)
REPRO_RESULT_PATH = PROJECT_ROOT / "results" / "stage7" / "longbench_small_results.csv"
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "innovation" / "ours_longbench_small_summary.csv"
)
COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "innovation" / "ours_longbench_small_comparison.csv"
)


def parse_usage(value: Any) -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}

    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue

    return {}


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    usage_dicts = (
        df["usage"].apply(parse_usage) if "usage" in df.columns else [{}] * len(df)
    )
    df["api_prompt_tokens"] = [u.get("prompt_tokens") for u in usage_dicts]
    df["api_completion_tokens"] = [u.get("completion_tokens") for u in usage_dicts]
    df["api_total_tokens"] = [u.get("total_tokens") for u in usage_dicts]

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

    agg_dict: Dict[str, Any] = {
        "sample_count": ("sample_id", "count"),
        "valid_predictions": ("empty_prediction", lambda x: (x == 0).sum()),
        "empty_predictions": ("empty_prediction", "sum"),
        "contains_answer_rate": ("contains_answer", "mean"),
        "avg_f1": ("f1", "mean"),
        "avg_compressed_tokens": ("compressed_tokens", "mean"),
        "avg_token_saving_ratio": ("token_saving_ratio", "mean"),
        "avg_prompt_tokens": ("prompt_tokens", "mean"),
        "avg_compression_time": ("compression_time", "mean"),
        "avg_answer_time": ("answer_time", "mean"),
        "retry_used_count": ("retry_used_numeric", "sum"),
        "error_count": ("has_error", "sum"),
    }

    if "api_total_tokens" in df.columns:
        agg_dict["api_total_tokens_avg"] = ("api_total_tokens", "mean")

    summary = (
        df.groupby(["task_name", "method"])
        .agg(**agg_dict)
        .reset_index()
    )

    return summary


def main() -> None:
    if not OURS_RESULT_PATH.exists():
        raise FileNotFoundError(f"Our result file not found: {OURS_RESULT_PATH}")

    ours_df = pd.read_csv(OURS_RESULT_PATH)
    ours_summary = compute_summary(ours_df)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ours_summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print("=== Ours LongBench Small Summary ===")
    print(ours_summary.to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")

    # Build comparison with reproduction results
    if REPRO_RESULT_PATH.exists():
        repro_df = pd.read_csv(REPRO_RESULT_PATH)
        repro_summary = compute_summary(repro_df)
        comparison = pd.concat([repro_summary, ours_summary], ignore_index=True)
    else:
        print(f"[WARN] Repro results not found: {REPRO_RESULT_PATH}")
        comparison = ours_summary

    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")
    print(f"\nSaved comparison to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()

