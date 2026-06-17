from __future__ import annotations

import ast
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "results" / "final"

EXP1_INPUT = PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_results.csv"
EXP2_INPUT = PROJECT_ROOT / "results" / "stage6_position" / "exp2_position_results.csv"

EXP1_RESULT_OUT = FINAL_DIR / "nq_style_eval_exp1_results.csv"
EXP1_SUMMARY_OUT = FINAL_DIR / "nq_style_eval_exp1_summary.csv"
EXP2_RESULT_OUT = FINAL_DIR / "nq_style_eval_exp2_results.csv"
EXP2_SUMMARY_OUT = FINAL_DIR / "nq_style_eval_exp2_summary.csv"
NOTES_OUT = FINAL_DIR / "nq_style_eval_notes.md"

METHOD_ORDER = ["original", "truncate", "bm25", "llmlingua", "longllmlingua"]
POSITION_ORDER = [1, 5, 10, 15, 20]


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)

    def lower(s: str) -> str:
        return s.lower()

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def parse_answers(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]

    if isinstance(value, float) and pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass

    return [text]


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: list[str],
) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def load_valid_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    if "error_message" in df.columns:
        df = df[df["error_message"].fillna("").astype(str).str.len() == 0]

    if "empty_prediction" in df.columns:
        empty_text = df["empty_prediction"].fillna("").astype(str).str.strip().str.lower()
        empty_numeric = pd.to_numeric(df["empty_prediction"], errors="coerce").fillna(0)
        is_empty = empty_text.isin(["true", "1", "yes"]) | (empty_numeric.astype(int) == 1)
        df = df[~is_empty]

    return df.copy()


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

    out["normalized_prediction"] = out["prediction"].apply(normalize_answer)
    return out


def summarize_exp1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["contains_answer", "f1", "nq_style_em", "nq_style_f1"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    summary = (
        out.groupby("method")
        .agg(
            rows=("sample_id", "count"),
            contains_answer_rate=("contains_answer", "mean"),
            simple_avg_f1=("f1", "mean"),
            nq_style_em=("nq_style_em", "mean"),
            nq_style_f1=("nq_style_f1", "mean"),
        )
        .reset_index()
    )

    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values("method").reset_index(drop=True)


def summarize_exp2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["answer_position", "contains_answer", "f1", "nq_style_em", "nq_style_f1"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    summary = (
        out.groupby(["answer_position", "method"])
        .agg(
            rows=("sample_id", "count"),
            contains_answer_rate=("contains_answer", "mean"),
            simple_avg_f1=("f1", "mean"),
            nq_style_em=("nq_style_em", "mean"),
            nq_style_f1=("nq_style_f1", "mean"),
        )
        .reset_index()
    )

    summary["answer_position"] = pd.Categorical(
        summary["answer_position"],
        categories=POSITION_ORDER,
        ordered=True,
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values(["answer_position", "method"]).reset_index(drop=True)


def build_notes(exp1_summary: pd.DataFrame) -> str:
    lines = [
        "# NQ-style Normalized EM/F1 Evaluation Notes\n",
        "This supplemental evaluation reuses existing predictions and does not call DeepSeek API again.\n",
        "## Evaluation Definition\n",
        "- Lowercase normalization",
        "- Punctuation removal",
        "- English article removal: a / an / the",
        "- Whitespace normalization",
        "- Exact Match over normalized strings",
        "- Token-level F1 over normalized tokens",
        "- For multiple gold answers, the maximum score is used.\n",
        "## Scope\n",
        (
            "This is not the full original NaturalQuestions official evaluation. It is a "
            "normalized short-answer matching protocol suitable for Lost-in-the-Middle / "
            "NQ-style QA data.\n"
        ),
        "## Experiment 1 Summary\n",
        "| Method | Contains Answer | Simple F1 | NQ-style EM | NQ-style F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, r in exp1_summary.iterrows():
        lines.append(
            f"| {r['method']} | {float(r['contains_answer_rate']):.3f} | "
            f"{float(r['simple_avg_f1']):.3f} | {float(r['nq_style_em']):.3f} | "
            f"{float(r['nq_style_f1']):.3f} |"
        )

    lines.extend(
        [
            "\n## Reporting Guidance\n",
            "- For NQ and Lost-in-the-Middle, report NQ-style EM/F1 as normalized QA metrics.",
            "- Keep contains_answer and simple F1 as auxiliary project-wide metrics.",
            "- Do not call this a full official NaturalQuestions evaluation.\n",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    exp1 = load_valid_rows(EXP1_INPUT)
    exp2 = load_valid_rows(EXP2_INPUT)

    exp1_eval = add_nq_style_metrics(exp1)
    exp2_eval = add_nq_style_metrics(exp2)

    exp1_summary = summarize_exp1(exp1_eval)
    exp2_summary = summarize_exp2(exp2_eval)

    exp1_eval.to_csv(EXP1_RESULT_OUT, index=False, encoding="utf-8-sig")
    exp1_summary.to_csv(EXP1_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    exp2_eval.to_csv(EXP2_RESULT_OUT, index=False, encoding="utf-8-sig")
    exp2_summary.to_csv(EXP2_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    NOTES_OUT.write_text(build_notes(exp1_summary), encoding="utf-8")

    print(f"[OK] Wrote: {EXP1_RESULT_OUT}")
    print(f"[OK] Wrote: {EXP1_SUMMARY_OUT}")
    print(f"[OK] Wrote: {EXP2_RESULT_OUT}")
    print(f"[OK] Wrote: {EXP2_SUMMARY_OUT}")
    print(f"[OK] Wrote: {NOTES_OUT}")
    print("\nExperiment 1 NQ-style summary:")
    print(exp1_summary)
    print("\nExperiment 2 NQ-style summary:")
    print(exp2_summary)


if __name__ == "__main__":
    main()

