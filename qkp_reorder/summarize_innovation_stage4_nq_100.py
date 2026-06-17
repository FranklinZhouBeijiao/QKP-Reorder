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
    add_nq_style_metrics,
    parse_answers,
    metric_max_over_ground_truths,
    exact_match_score,
    token_f1_score,
)

REPRO_RESULT_PATH = PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_results.csv"
OURS_RESULT_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_nq_100_results.csv"
INNOVATION_DIR = PROJECT_ROOT / "results" / "innovation"
SUMMARY_PATH = INNOVATION_DIR / "ours_nq_100_summary.csv"
COMPARISON_PATH = INNOVATION_DIR / "ours_nq_100_comparison.csv"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_bm25_only",
    "ours_keyword",
    "ours_full",
]


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


def load_and_label(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    df = pd.read_csv(path)
    df["source_file"] = label
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    usage_dicts = (
        df["usage"].apply(parse_usage) if "usage" in df.columns else [{}] * len(df)
    )
    df["api_prompt_tokens"] = [u.get("prompt_tokens") for u in usage_dicts]
    df["api_completion_tokens"] = [u.get("completion_tokens") for u in usage_dicts]
    df["api_total_tokens"] = [u.get("total_tokens") for u in usage_dicts]

    numeric_cols = [
        "original_tokens", "compressed_tokens", "prompt_tokens",
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

    summary = (
        df.groupby("method")
        .agg(
            sample_count=("sample_id", "count"),
            unique_samples=("sample_id", "nunique"),
            valid_predictions=("empty_prediction", lambda x: (x == 0).sum()),
            empty_predictions=("empty_prediction", "sum"),
            contains_answer_avg=("contains_answer", "mean"),
            f1_avg=("f1", "mean"),
            original_tokens_avg=("original_tokens", "mean"),
            compressed_tokens_avg=("compressed_tokens", "mean"),
            prompt_tokens_avg=("prompt_tokens", "mean"),
            compression_ratio_avg=("compression_ratio", "mean"),
            token_saving_ratio_avg=("token_saving_ratio", "mean"),
            compression_time_avg=("compression_time", "mean"),
            answer_time_avg=("answer_time", "mean"),
            fallback_used_count=("fallback_used_numeric", "sum"),
            retry_used_count=("retry_used_numeric", "sum"),
            error_count=("has_error", "sum"),
            api_prompt_tokens_avg=("api_prompt_tokens", "mean"),
            api_completion_tokens_avg=("api_completion_tokens", "mean"),
            api_total_tokens_avg=("api_total_tokens", "mean"),
        )
        .reset_index()
    )

    return summary


def add_nq_style_metrics_to_df(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_nq_style_summary(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["nq_style_em", "nq_style_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby("method")
        .agg(
            nq_style_em=("nq_style_em", "mean"),
            nq_style_f1=("nq_style_f1", "mean"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    INNOVATION_DIR.mkdir(parents=True, exist_ok=True)

    repro_df = load_and_label(REPRO_RESULT_PATH, "reproduction/stage6")
    ours_df = load_and_label(OURS_RESULT_PATH, "innovation/stage4")

    if repro_df is None and ours_df is None:
        raise FileNotFoundError("Neither reproduction nor innovation results found.")

    combined = pd.concat(
        [d for d in [repro_df, ours_df] if d is not None],
        ignore_index=True,
    )

    summary = compute_summary(combined)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values("method").reset_index(drop=True)

    eval_df = add_nq_style_metrics_to_df(combined)
    nq_style_summary = compute_nq_style_summary(eval_df)

    summary = summary.merge(nq_style_summary, on="method", how="left")

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Summary saved to: {SUMMARY_PATH}")

    comparison_cols = [
        "method", "sample_count", "valid_predictions", "empty_predictions",
        "contains_answer_avg", "f1_avg", "nq_style_em", "nq_style_f1",
        "compressed_tokens_avg", "token_saving_ratio_avg",
        "prompt_tokens_avg", "api_total_tokens_avg", "answer_time_avg",
        "compression_time_avg", "retry_used_count", "error_count",
    ]
    comparison = summary[[c for c in comparison_cols if c in summary.columns]]
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Comparison saved to: {COMPARISON_PATH}")

    print("\n=== Summary Table ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

