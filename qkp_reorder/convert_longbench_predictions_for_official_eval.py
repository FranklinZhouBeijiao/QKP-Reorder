from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_CSV = PROJECT_ROOT / "results" / "stage7" / "longbench_small_results.csv"

LONGBENCH_ROOT = PROJECT_ROOT / "external" / "LongBench" / "LongBench"
PRED_ROOT = LONGBENCH_ROOT / "pred"

TASKS = [
    "hotpotqa",
    "2wikimqa",
    "multifieldqa_en",
]

METHODS = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]


def parse_answers(value: Any) -> List[str]:
    """Convert gold_answers stored in CSV to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        if isinstance(parsed, str):
            return [parsed] if parsed.strip() else []

    return [text]


def normalize_prediction(value: Any) -> str:
    """Return a JSON-serializable prediction string."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def main() -> None:
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"Missing result CSV: {RESULT_CSV}")

    if not LONGBENCH_ROOT.exists():
        raise FileNotFoundError(
            f"Missing LongBench official repo path: {LONGBENCH_ROOT}. "
            "Please clone THUDM/LongBench into external/LongBench first."
        )

    df = pd.read_csv(RESULT_CSV)

    required_cols = ["task_name", "method", "prediction", "gold_answers"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    if "error_message" in df.columns:
        df = df[df["error_message"].fillna("").astype(str).str.len() == 0]

    if "empty_prediction" in df.columns:
        df = df[
            pd.to_numeric(df["empty_prediction"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(0)
        ]

    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for method in METHODS:
        method_dir = PRED_ROOT / method
        method_dir.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            sub = df[(df["method"] == method) & (df["task_name"] == task)].copy()
            if sub.empty:
                print(f"[WARN] No rows for method={method}, task={task}")
                continue

            if "sample_id" in sub.columns:
                sub = sub.sort_values("sample_id")

            out_path = method_dir / f"{task}.jsonl"
            written = 0

            with out_path.open("w", encoding="utf-8") as file:
                for _, row in sub.iterrows():
                    item = {
                        "pred": normalize_prediction(row.get("prediction")),
                        "answers": parse_answers(row.get("gold_answers")),
                        "all_classes": None,
                    }

                    length = row.get("length", None)
                    if pd.notna(length):
                        try:
                            item["length"] = int(float(length))
                        except Exception:
                            item["length"] = length

                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1

            total_written += written
            print(f"[OK] Wrote {written} rows: {out_path}")

    print("\n" + "=" * 100)
    print(f"Total written rows: {total_written}")
    print(f"Pred root: {PRED_ROOT}")


if __name__ == "__main__":
    main()

