from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_DIR = PROJECT_ROOT / "results" / "final"

OFFICIAL_JSON_FILES = {
    "original": FINAL_DIR / "longbench_official_eval_original.json",
    "truncate": FINAL_DIR / "longbench_official_eval_truncate.json",
    "bm25": FINAL_DIR / "longbench_official_eval_bm25.json",
    "llmlingua": FINAL_DIR / "longbench_official_eval_llmlingua.json",
    "longllmlingua": FINAL_DIR / "longbench_official_eval_longllmlingua.json",
}

SIMPLIFIED_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "stage7" / "longbench_small_summary.csv"
)

OFFICIAL_SUMMARY_PATH = FINAL_DIR / "longbench_official_eval_summary.csv"
COMPARISON_PATH = FINAL_DIR / "longbench_official_eval_comparison.csv"
NOTES_PATH = FINAL_DIR / "longbench_official_eval_notes.md"

TASK_ORDER = ["hotpotqa", "2wikimqa", "multifieldqa_en"]
METHOD_ORDER = ["original", "truncate", "bm25", "llmlingua", "longllmlingua"]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing official eval json: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_official_summary() -> pd.DataFrame:
    rows = []

    for method, path in OFFICIAL_JSON_FILES.items():
        data = load_json(path)
        for task, score in data.items():
            rows.append(
                {
                    "method": method,
                    "task_name": task,
                    "official_score": score,
                    "source_json": str(path.relative_to(PROJECT_ROOT)),
                }
            )

    df = pd.DataFrame(rows)
    df["task_name"] = pd.Categorical(
        df["task_name"], categories=TASK_ORDER, ordered=True
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["task_name", "method"])


def build_comparison(official_df: pd.DataFrame) -> pd.DataFrame:
    if not SIMPLIFIED_SUMMARY_PATH.exists():
        print(f"[WARN] Missing simplified summary: {SIMPLIFIED_SUMMARY_PATH}")
        return official_df

    simple_df = pd.read_csv(SIMPLIFIED_SUMMARY_PATH)
    keep_cols = [
        "task_name",
        "method",
        "contains_answer_rate",
        "avg_f1",
        "avg_compressed_tokens",
        "avg_token_saving_ratio",
    ]
    keep_cols = [col for col in keep_cols if col in simple_df.columns]
    simple_df = simple_df[keep_cols].copy()

    return official_df.merge(simple_df, on=["task_name", "method"], how="left")


def build_notes(official_df: pd.DataFrame) -> str:
    lines = [
        "# LongBench Official Evaluation Notes",
        "",
        "This file summarizes the supplemental official LongBench evaluation.",
        "",
        "## Evaluation Source",
        "",
        "- Official script: `external/LongBench/LongBench/eval.py`",
        "- Predictions converted from: `results/stage7/longbench_small_results.csv`",
        "- Evaluation mode: standard LongBench eval, not LongBench-E",
        "- No additional DeepSeek API calls were made.",
        "",
        "## Official Scores",
        "",
        "| Task | Method | Official Score |",
        "|---|---|---:|",
    ]

    for _, row in official_df.iterrows():
        lines.append(
            f"| {row['task_name']} | {row['method']} | "
            f"{float(row['official_score']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The official LongBench scores are task-specific and should be preferred when reporting LongBench results.",
            "- The previous `contains_answer` and `simple_token_f1` metrics remain useful for cross-experiment consistency, but they are simplified metrics.",
            "- MultiFieldQA-en may show differences between simplified and official metrics because open-ended answers are not always captured by strict string matching.",
            "- This evaluation reuses existing predictions; it does not change the answer model or compressed contexts.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    official_df = build_official_summary()
    official_df.to_csv(OFFICIAL_SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"[OK] Wrote official summary: {OFFICIAL_SUMMARY_PATH}")

    comparison_df = build_comparison(official_df)
    comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")
    print(f"[OK] Wrote comparison: {COMPARISON_PATH}")

    NOTES_PATH.write_text(build_notes(official_df), encoding="utf-8")
    print(f"[OK] Wrote notes: {NOTES_PATH}")

    print("\nOfficial summary:")
    print(official_df.to_string(index=False))


if __name__ == "__main__":
    main()

