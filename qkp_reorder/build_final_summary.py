from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "results" / "final"

EXP1_PATH = FINAL_DIR / "final_exp1_nq_summary.csv"
EXP2_PATH = FINAL_DIR / "final_exp2_position_summary.csv"
EXP3_PATH = FINAL_DIR / "final_exp3_longbench_summary.csv"

FINAL_SUMMARY_PATH = FINAL_DIR / "final_experiment_summary.md"
KEY_FINDINGS_PATH = FINAL_DIR / "key_findings.md"


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_method_row(df: pd.DataFrame, method: str) -> Optional[pd.Series]:
    if "method" not in df.columns:
        return None
    sub = df[df["method"] == method]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_exp1_section(df: pd.DataFrame) -> str:
    rows = [
        "## Experiment 1: NaturalQuestions Multi-document QA",
        "",
        "### Setting",
        "",
        "- Dataset: Lost-in-the-Middle NQ 10-document setting",
        "- Samples: 100",
        "- Documents per sample: 10",
        "- Answer document position: 5",
        "- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua",
        "- Answer model: DeepSeek API",
        "",
        "### Summary Table",
        "",
        "| Method | Avg. Compressed Tokens | Token Saving | Contains Answer | Avg. F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"| {r['method']} | "
            f"{fmt(r.get('avg_compressed_tokens'))} | "
            f"{fmt(r.get('avg_token_saving_ratio'))} | "
            f"{fmt(r.get('contains_answer_rate'))} | "
            f"{fmt(r.get('avg_f1'))} |"
        )

    llm_row = get_method_row(df, "longllmlingua")
    orig_row = get_method_row(df, "original")

    rows.extend(["", "### Key Observation", ""])
    if llm_row is not None:
        rows.append(
            "- LongLLMLingua reduces the input context to "
            f"{fmt(llm_row.get('avg_compressed_tokens'))} tokens on average, "
            f"with a token saving ratio of {fmt(llm_row.get('avg_token_saving_ratio'))}."
        )

    if llm_row is not None and orig_row is not None:
        rows.append(
            "- Compared with Original, LongLLMLingua keeps a comparable "
            f"contains-answer rate ({fmt(llm_row.get('contains_answer_rate'))} vs. "
            f"{fmt(orig_row.get('contains_answer_rate'))}) while substantially "
            "reducing input tokens."
        )

    rows.append("")
    return "\n".join(rows)


def build_exp2_section(df: pd.DataFrame) -> str:
    rows = [
        "## Experiment 2: Lost in the Middle Position Sensitivity",
        "",
        "### Setting",
        "",
        "- Dataset: Lost-in-the-Middle NQ 20-document setting",
        "- Positions: 1, 5, 10, 15, 20",
        "- Samples per position: 20",
        "- Total samples: 100",
        "- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua",
        "",
        "### Summary Table",
        "",
        "| Position | Method | Contains Answer | Avg. F1 | Avg. Tokens |",
        "|---:|---|---:|---:|---:|",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"| {int(r['answer_position'])} | {r['method']} | "
            f"{fmt(r.get('contains_answer_rate'))} | "
            f"{fmt(r.get('avg_f1'))} | "
            f"{fmt(r.get('avg_compressed_tokens'))} |"
        )

    rows.extend(["", "### Key Observation", ""])

    for pos in [15, 20]:
        tr = df[(df["answer_position"] == pos) & (df["method"] == "truncate")]
        ll = df[(df["answer_position"] == pos) & (df["method"] == "longllmlingua")]
        if not tr.empty and not ll.empty:
            rows.append(
                f"- At Position {pos}, Truncate obtains a contains-answer rate of "
                f"{fmt(tr.iloc[0].get('contains_answer_rate'))}, while "
                f"LongLLMLingua obtains {fmt(ll.iloc[0].get('contains_answer_rate'))}."
            )

    rows.append(
        "- This result suggests that simple truncation is highly sensitive to answer "
        "position, whereas LongLLMLingua is more stable when the answer document is "
        "placed later in the context."
    )
    rows.append("")
    return "\n".join(rows)


def build_exp3_section(df: pd.DataFrame) -> str:
    rows = [
        "## Experiment 3: LongBench Subset",
        "",
        "### Setting",
        "",
        "- Dataset: LongBench",
        "- Tasks: HotpotQA, 2WikiMultihopQA, MultiFieldQA-en",
        "- Samples per task: 50",
        "- Total samples: 150",
        "- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua",
        "",
        "### Summary Table",
        "",
        "| Task | Method | Contains Answer | Avg. F1 | Avg. Tokens |",
        "|---|---|---:|---:|---:|",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"| {r['task_name']} | {r['method']} | "
            f"{fmt(r.get('contains_answer_rate'))} | "
            f"{fmt(r.get('avg_f1'))} | "
            f"{fmt(r.get('avg_compressed_tokens'))} |"
        )

    rows.extend(["", "### Key Observation", ""])
    rows.append(
        "- On multi-hop QA tasks such as HotpotQA and 2WikiMultihopQA, Truncate is "
        "generally weaker, indicating that simple prefix preservation may discard "
        "useful evidence."
    )
    rows.append("- BM25 remains a competitive baseline for lexical multi-hop QA retrieval.")
    rows.append(
        "- MultiFieldQA-en shows lower contains-answer rates across methods, suggesting "
        "that strict string matching may be too conservative for open-ended answers."
    )
    rows.append("")
    return "\n".join(rows)


def build_limitations_section() -> str:
    return """## Implementation Differences and Limitations

This reproduction should be interpreted as a course-project reproduction under a unified experimental pipeline.

Important implementation notes:

1. The answer model is DeepSeek API rather than the exact answer model used in the original paper.
2. The compressor uses a CPU-compatible LLMLingua-2 model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`.
3. LLMLingua and LongLLMLingua may produce the same compressed token length under this implementation.
4. Evaluation uses simplified `contains_answer` and `simple_token_f1` metrics.
5. LongBench official task-specific metrics are not used in the current stage.
6. Results should be interpreted as relative comparisons under the same answer model and evaluation pipeline.
"""


def build_key_findings() -> str:
    return """# Key Findings

## Main Findings

1. In the NQ 100-sample experiment, LongLLMLingua substantially reduces input tokens while maintaining comparable answer quality.
2. In the position sensitivity experiment, Truncate drops sharply when the answer document is placed later, while LongLLMLingua is more stable.
3. In the LongBench subset, task type matters: BM25 is competitive on multi-hop QA, while MultiFieldQA-en is difficult to evaluate with strict contains-answer matching.
4. Across experiments, compression can reduce prompt length significantly, but answer quality depends on task structure and evidence distribution.

## Notes for Report Writing

- Avoid claiming exact reproduction of the original paper's numerical results.
- Use the phrasing: "under our DeepSeek-based reproduction setting" or "in our unified evaluation pipeline".
- Clearly state that LongBench is evaluated with simplified contains-answer and token-F1 metrics.
"""


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    exp1 = load_csv(EXP1_PATH)
    exp2 = load_csv(EXP2_PATH)
    exp3 = load_csv(EXP3_PATH)

    parts = [
        "# Final Experiment Summary",
        "",
        "This document summarizes the final reproduction results for the LongLLMLingua course project.",
        "",
        build_exp1_section(exp1),
        build_exp2_section(exp2),
        build_exp3_section(exp3),
        build_limitations_section(),
    ]

    FINAL_SUMMARY_PATH.write_text("\n".join(parts), encoding="utf-8")
    KEY_FINDINGS_PATH.write_text(build_key_findings(), encoding="utf-8")

    print(f"[OK] Wrote final summary: {FINAL_SUMMARY_PATH}")
    print(f"[OK] Wrote key findings: {KEY_FINDINGS_PATH}")


if __name__ == "__main__":
    main()

