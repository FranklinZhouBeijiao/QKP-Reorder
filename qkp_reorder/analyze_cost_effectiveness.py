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

FINAL_DIR = PROJECT_ROOT / "results" / "final"

INPUT_FILES = {
    "exp1_nq_100": PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_results.csv",
    "exp2_position": PROJECT_ROOT
    / "results"
    / "stage6_position"
    / "exp2_position_results.csv",
    "exp3_longbench": PROJECT_ROOT
    / "results"
    / "stage7"
    / "longbench_small_results.csv",
}

BY_EXPERIMENT_PATH = FINAL_DIR / "cost_effectiveness_by_experiment.csv"
OVERALL_PATH = FINAL_DIR / "cost_effectiveness_overall.csv"
NOTES_PATH = FINAL_DIR / "cost_effectiveness_notes.md"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]


def parse_usage(value: Any) -> Dict[str, Any]:
    """Parse API usage field saved in CSV."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and pd.isna(value):
        return {}

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return {}


def extract_usage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add API token usage columns with robust fallbacks."""
    if "usage" not in df.columns:
        df["api_prompt_tokens"] = pd.NA
        df["api_completion_tokens"] = pd.NA
        df["api_total_tokens"] = pd.NA
        return df

    usage_dicts = df["usage"].apply(parse_usage)

    df["api_prompt_tokens"] = [
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
        for usage in usage_dicts
    ]
    df["api_completion_tokens"] = [
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_token_count")
        for usage in usage_dicts
    ]
    df["api_total_tokens"] = [
        usage.get("total_tokens") or usage.get("total_token_count")
        for usage in usage_dicts
    ]

    for col in ["api_prompt_tokens", "api_completion_tokens", "api_total_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_total = df["api_total_tokens"].isna()
    df.loc[missing_total, "api_total_tokens"] = (
        df.loc[missing_total, "api_prompt_tokens"].fillna(0)
        + df.loc[missing_total, "api_completion_tokens"].fillna(0)
    )

    if "prompt_tokens" in df.columns:
        df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce")
        missing_api_prompt = df["api_prompt_tokens"].isna()
        df.loc[missing_api_prompt, "api_prompt_tokens"] = df.loc[
            missing_api_prompt, "prompt_tokens"
        ]

    zero_or_missing_total = df["api_total_tokens"].isna() | (
        df["api_total_tokens"] <= 0
    )
    df.loc[zero_or_missing_total, "api_total_tokens"] = df.loc[
        zero_or_missing_total, "api_prompt_tokens"
    ]

    return df


def load_result_file(experiment_name: str, path: Path) -> pd.DataFrame:
    """Load one experiment result CSV and keep valid completed rows."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file for {experiment_name}: {path}")

    df = pd.read_csv(path)
    df["experiment"] = experiment_name

    if "error_message" in df.columns:
        df = df[df["error_message"].fillna("").astype(str).str.len() == 0]

    if "empty_prediction" in df.columns:
        df = df[
            pd.to_numeric(df["empty_prediction"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(0)
        ]

    df = extract_usage_columns(df)

    numeric_cols = [
        "compressed_tokens",
        "prompt_tokens",
        "token_saving_ratio",
        "answer_time",
        "contains_answer",
        "f1",
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_cost_effectiveness_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Add derived cost-effectiveness metrics to an aggregate table."""
    summary["f1_per_1k_tokens"] = summary.apply(
        lambda row: row["avg_f1"] / (row["avg_api_total_tokens"] / 1000)
        if pd.notna(row["avg_f1"])
        and pd.notna(row["avg_api_total_tokens"])
        and row["avg_api_total_tokens"] > 0
        else pd.NA,
        axis=1,
    )

    summary["contains_answer_per_1k_tokens"] = summary.apply(
        lambda row: row["contains_answer_rate"]
        / (row["avg_api_total_tokens"] / 1000)
        if pd.notna(row["contains_answer_rate"])
        and pd.notna(row["avg_api_total_tokens"])
        and row["avg_api_total_tokens"] > 0
        else pd.NA,
        axis=1,
    )

    summary["tokens_per_correct_answer"] = summary.apply(
        lambda row: row["avg_api_total_tokens"] / row["contains_answer_rate"]
        if pd.notna(row["contains_answer_rate"]) and row["contains_answer_rate"] > 0
        else pd.NA,
        axis=1,
    )

    return summary


def summarize_by_experiment(all_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_df.groupby(["experiment", "method"])
        .agg(
            rows=("method", "count"),
            avg_compressed_tokens=("compressed_tokens", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_api_prompt_tokens=("api_prompt_tokens", "mean"),
            avg_api_completion_tokens=("api_completion_tokens", "mean"),
            avg_api_total_tokens=("api_total_tokens", "mean"),
            avg_token_saving_ratio=("token_saving_ratio", "mean"),
            avg_answer_time=("answer_time", "mean"),
            contains_answer_rate=("contains_answer", "mean"),
            avg_f1=("f1", "mean"),
        )
        .reset_index()
    )
    summary = add_cost_effectiveness_metrics(summary)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    return summary.sort_values(["experiment", "method"]).reset_index(drop=True)


def summarize_overall(all_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_df.groupby(["method"])
        .agg(
            rows=("method", "count"),
            avg_compressed_tokens=("compressed_tokens", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_api_prompt_tokens=("api_prompt_tokens", "mean"),
            avg_api_completion_tokens=("api_completion_tokens", "mean"),
            avg_api_total_tokens=("api_total_tokens", "mean"),
            avg_token_saving_ratio=("token_saving_ratio", "mean"),
            avg_answer_time=("answer_time", "mean"),
            contains_answer_rate=("contains_answer", "mean"),
            avg_f1=("f1", "mean"),
        )
        .reset_index()
    )
    summary = add_cost_effectiveness_metrics(summary)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    return summary.sort_values("method").reset_index(drop=True)


def build_notes(overall: pd.DataFrame) -> str:
    lines = [
        "# Cost-Effectiveness Analysis Notes",
        "",
        "This analysis reuses existing experiment CSV files and does not call DeepSeek API again.",
        "",
        "## Metrics",
        "",
        "- `avg_api_total_tokens`: average API token usage parsed from the `usage` field.",
        "- `f1_per_1k_tokens`: average F1 divided by average total API tokens per 1,000 tokens.",
        "- `contains_answer_per_1k_tokens`: contains-answer rate divided by average total API tokens per 1,000 tokens.",
        "- `tokens_per_correct_answer`: average total API tokens required per strict answer hit.",
        "",
        "## Overall Summary",
        "",
        "| Method | Avg Total Tokens | Contains Answer | Avg F1 | F1 per 1k Tokens | Tokens per Correct Answer |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | "
            f"{row['avg_api_total_tokens']:.2f} | "
            f"{row['contains_answer_rate']:.3f} | "
            f"{row['avg_f1']:.3f} | "
            f"{row['f1_per_1k_tokens']:.3f} | "
            f"{row['tokens_per_correct_answer']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guidance",
            "",
            "- Original often has strong answer quality but highest token cost.",
            "- Truncate reduces token cost but may lose important evidence in long-context tasks.",
            "- BM25 can be cost-effective when lexical evidence is strong.",
            "- LongLLMLingua should be interpreted by balancing quality and token reduction, not by accuracy alone.",
            "- These are token proxy costs, not exact monetary costs.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for experiment_name, path in INPUT_FILES.items():
        df = load_result_file(experiment_name, path)
        dfs.append(df)
        print(f"[OK] Loaded {experiment_name}: {len(df)} valid rows")

    all_df = pd.concat(dfs, ignore_index=True)
    by_experiment = summarize_by_experiment(all_df)
    overall = summarize_overall(all_df)

    by_experiment.to_csv(BY_EXPERIMENT_PATH, index=False, encoding="utf-8")
    overall.to_csv(OVERALL_PATH, index=False, encoding="utf-8")
    NOTES_PATH.write_text(build_notes(overall), encoding="utf-8")

    print(f"[OK] Wrote by-experiment summary: {BY_EXPERIMENT_PATH}")
    print(f"[OK] Wrote overall summary: {OVERALL_PATH}")
    print(f"[OK] Wrote notes: {NOTES_PATH}")
    print("\nOverall:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

