from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INNOVATION_RESULT_DIR = PROJECT_ROOT / "results" / "innovation"
OFFICIAL_EVAL_JSON = (
    PROJECT_ROOT
    / "external"
    / "LongBench"
    / "LongBench"
    / "pred"
    / "innovation_ours_full"
    / "result.json"
)
REPRO_OFFICIAL_SUMMARY = (
    PROJECT_ROOT / "results" / "final" / "longbench_official_eval_summary.csv"
)
OURS_SIMPLIFIED_SUMMARY = (
    INNOVATION_RESULT_DIR / "ours_longbench_small_summary.csv"
)

OFFICIAL_SUMMARY_PATH = (
    INNOVATION_RESULT_DIR / "ours_longbench_official_eval_summary.csv"
)
COMPARISON_PATH = (
    INNOVATION_RESULT_DIR / "ours_longbench_official_eval_comparison.csv"
)

TASK_ORDER = ["hotpotqa", "2wikimqa", "multifieldqa_en"]
METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_full",
]


def build_ours_official_summary() -> pd.DataFrame:
    if not OFFICIAL_EVAL_JSON.exists():
        print(f"[WARN] Official eval JSON not found: {OFFICIAL_EVAL_JSON}")
        return pd.DataFrame()

    with OFFICIAL_EVAL_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []
    for task, score in data.items():
        rows.append(
            {
                "method": "ours_full",
                "task_name": task,
                "official_score": score,
                "source_json": str(OFFICIAL_EVAL_JSON.relative_to(PROJECT_ROOT)),
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_comparison(ours_official: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if REPRO_OFFICIAL_SUMMARY.exists():
        repro_df = pd.read_csv(REPRO_OFFICIAL_SUMMARY)
        frames.append(repro_df)

    if not ours_official.empty:
        frames.append(ours_official)

    if not frames:
        return pd.DataFrame()

    official_df = pd.concat(frames, ignore_index=True)

    # Enrich with simplified metrics if available
    simplified_cols = [
        "task_name",
        "method",
        "contains_answer_rate",
        "avg_f1",
        "avg_compressed_tokens",
        "avg_token_saving_ratio",
    ]

    if OURS_SIMPLIFIED_SUMMARY.exists():
        ours_simple = pd.read_csv(OURS_SIMPLIFIED_SUMMARY)
        if REPRO_OFFICIAL_SUMMARY.exists() and REPRO_OFFICIAL_SUMMARY.parent.parent / "stage7" / "longbench_small_summary.csv":
            repro_simple_path = (
                PROJECT_ROOT / "results" / "stage7" / "longbench_small_summary.csv"
            )
            if repro_simple_path.exists():
                repro_simple = pd.read_csv(repro_simple_path)
                all_simple = pd.concat([repro_simple, ours_simple], ignore_index=True)
                cols = [c for c in simplified_cols if c in all_simple.columns]
                official_df = official_df.merge(
                    all_simple[cols], on=["task_name", "method"], how="left"
                )
            else:
                cols = [c for c in simplified_cols if c in ours_simple.columns]
                official_df = official_df.merge(
                    ours_simple[cols], on=["task_name", "method"], how="left"
                )

    # Sort
    official_df["task_name"] = pd.Categorical(
        official_df["task_name"], categories=TASK_ORDER, ordered=True
    )
    official_df["method"] = pd.Categorical(
        official_df["method"], categories=METHOD_ORDER, ordered=True
    )
    official_df = official_df.sort_values(["task_name", "method"])
    return official_df


def main() -> None:
    INNOVATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build ours official summary
    ours_official = build_ours_official_summary()
    if not ours_official.empty:
        ours_official.to_csv(OFFICIAL_SUMMARY_PATH, index=False, encoding="utf-8")
        print(f"[OK] Wrote ours official summary: {OFFICIAL_SUMMARY_PATH}")
        print(ours_official.to_string(index=False))
    else:
        print("[WARN] No ours official eval data available yet.")
        ours_official = pd.DataFrame()

    # Step 2: Build full comparison
    comparison = build_comparison(ours_official)
    if not comparison.empty:
        comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")
        print(f"\n[OK] Wrote comparison: {COMPARISON_PATH}")
        print("\n=== LongBench Official Eval Comparison ===")
        print(comparison.to_string(index=False))
    else:
        print("[WARN] No comparison data available.")


if __name__ == "__main__":
    main()

