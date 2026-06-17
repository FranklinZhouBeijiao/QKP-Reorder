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

from qkp_reorder.evaluate_nq_style_metrics import (
    exact_match_score,
    metric_max_over_ground_truths,
    parse_answers,
    token_f1_score,
)

RESULT_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_ablation_results.csv"
INNOVATION_DIR = PROJECT_ROOT / "results" / "innovation"
SUMMARY_PATH = INNOVATION_DIR / "ours_ablation_summary.csv"
BY_METHOD_PATH = INNOVATION_DIR / "ours_ablation_summary_by_method.csv"
BY_POSITION_PATH = INNOVATION_DIR / "ours_ablation_summary_by_position.csv"

METHOD_ORDER = ["ours_bm25_only", "ours_keyword", "ours_full"]
POSITION_ORDER = [1, 5, 10, 15, 20]


def parse_usage(value: Any) -> Dict[str, Any]:
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


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    usage_dicts = (
        df["usage"].apply(parse_usage) if "usage" in df.columns else [{}] * len(df)
    )
    df["api_prompt_tokens"] = [u.get("prompt_tokens") for u in usage_dicts]
    df["api_completion_tokens"] = [u.get("completion_tokens") for u in usage_dicts]
    df["api_total_tokens"] = [u.get("total_tokens") for u in usage_dicts]

    numeric_cols = [
        "answer_position", "original_tokens", "compressed_tokens", "prompt_tokens",
        "compression_ratio", "token_saving_ratio", "compression_time",
        "answer_time", "contains_answer", "f1", "empty_prediction",
        "api_prompt_tokens", "api_completion_tokens", "api_total_tokens",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "fallback_used" in df.columns:
        df["fallback_used_numeric"] = (
            df["fallback_used"].astype(str).str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0)
        )
    else:
        df["fallback_used_numeric"] = 0

    if "retry_used" in df.columns:
        df["retry_used_numeric"] = (
            df["retry_used"].astype(str).str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0)
        )
    else:
        df["retry_used_numeric"] = 0

    if "error_message" in df.columns:
        df["has_error"] = (
            df["error_message"].fillna("").astype(str).str.len().gt(0).astype(int)
        )
    else:
        df["has_error"] = 0

    return df


def add_nq_style_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gold_answers_list"] = out["gold_answers"].apply(parse_answers)

    out["nq_style_em"] = out.apply(
        lambda r: metric_max_over_ground_truths(
            exact_match_score,
            str(r.get("prediction", "") or ""),
            r["gold_answers_list"],
        ),
        axis=1,
    )
    out["nq_style_f1"] = out.apply(
        lambda r: metric_max_over_ground_truths(
            token_f1_score,
            str(r.get("prediction", "") or ""),
            r["gold_answers_list"],
        ),
        axis=1,
    )
    return out


def compute_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "sample_count": ("sample_id", "count"),
        "unique_samples": ("sample_id", "nunique"),
        "valid_predictions": ("empty_prediction", lambda x: (x == 0).sum()),
        "empty_predictions": ("empty_prediction", "sum"),
        "contains_answer_avg": ("contains_answer", "mean"),
        "f1_avg": ("f1", "mean"),
        "nq_style_em_avg": ("nq_style_em", "mean"),
        "nq_style_f1_avg": ("nq_style_f1", "mean"),
        "original_tokens_avg": ("original_tokens", "mean"),
        "compressed_tokens_avg": ("compressed_tokens", "mean"),
        "prompt_tokens_avg": ("prompt_tokens", "mean"),
        "token_saving_ratio_avg": ("token_saving_ratio", "mean"),
        "compression_time_avg": ("compression_time", "mean"),
        "answer_time_avg": ("answer_time", "mean"),
        "retry_used_count": ("retry_used_numeric", "sum"),
        "error_count": ("has_error", "sum"),
        "api_prompt_tokens_avg": ("api_prompt_tokens", "mean"),
        "api_completion_tokens_avg": ("api_completion_tokens", "mean"),
        "api_total_tokens_avg": ("api_total_tokens", "mean"),
    }

    summary = (
        df.groupby(["method"])
        .agg(**agg_spec)
        .reset_index()
    )

    present_methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    summary["method"] = pd.Categorical(
        summary["method"], categories=present_methods, ordered=True
    )
    summary = summary.sort_values("method").reset_index(drop=True)
    return summary


def compute_position_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "sample_count": ("sample_id", "count"),
        "valid_predictions": ("empty_prediction", lambda x: (x == 0).sum()),
        "empty_predictions": ("empty_prediction", "sum"),
        "contains_answer_avg": ("contains_answer", "mean"),
        "f1_avg": ("f1", "mean"),
        "nq_style_f1_avg": ("nq_style_f1", "mean"),
        "compressed_tokens_avg": ("compressed_tokens", "mean"),
        "token_saving_ratio_avg": ("token_saving_ratio", "mean"),
        "answer_time_avg": ("answer_time", "mean"),
        "error_count": ("has_error", "sum"),
    }

    summary = (
        df.groupby(["answer_position", "method"])
        .agg(**agg_spec)
        .reset_index()
    )

    summary["answer_position"] = pd.Categorical(
        summary["answer_position"], categories=POSITION_ORDER, ordered=True
    )
    present_methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    summary["method"] = pd.Categorical(
        summary["method"], categories=present_methods, ordered=True
    )
    summary = summary.sort_values(["answer_position", "method"]).reset_index(drop=True)
    return summary


def main() -> None:
    INNOVATION_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)
    df = add_derived_columns(df)
    df = add_nq_style_metrics(df)

    total_samples = df["sample_id"].nunique()
    methods_found = sorted(df["method"].unique())
    print(f"Loaded {len(df)} rows, {total_samples} unique samples")
    print(f"Methods found: {methods_found}")

    # Overall summary by method
    method_summary = compute_method_summary(df)
    method_summary.to_csv(BY_METHOD_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Method summary saved to: {BY_METHOD_PATH}")

    # Summary by answer_position x method
    pos_summary = compute_position_summary(df)
    pos_summary.to_csv(BY_POSITION_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Position-method summary saved to: {BY_POSITION_PATH}")

    # Combined summary
    combined_cols = [
        "method", "sample_count", "unique_samples", "valid_predictions",
        "empty_predictions", "contains_answer_avg", "f1_avg",
        "nq_style_em_avg", "nq_style_f1_avg",
        "compressed_tokens_avg", "token_saving_ratio_avg",
        "prompt_tokens_avg", "api_total_tokens_avg",
        "compression_time_avg", "answer_time_avg",
        "retry_used_count", "error_count",
    ]
    combined = method_summary[
        [c for c in combined_cols if c in method_summary.columns]
    ]
    combined.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Combined summary saved to: {SUMMARY_PATH}")

    # Print overview
    print("\n=== Ablation Results Overview ===")
    print(method_summary.to_string(index=False))

    print("\n=== Position × Method ===")
    print(pos_summary.to_string(index=False))

    # Check for empty predictions and errors
    empty_total = int(method_summary["empty_predictions"].sum())
    error_total = int(method_summary["error_count"].sum())
    print(f"\nTotal empty_predictions: {empty_total}")
    print(f"Total errors: {error_total}")

    if empty_total > 0:
        print("WARNING: Non-zero empty predictions present. Review required.")
    if error_total > 0:
        print("WARNING: Non-zero errors present. Review required.")


if __name__ == "__main__":
    main()

