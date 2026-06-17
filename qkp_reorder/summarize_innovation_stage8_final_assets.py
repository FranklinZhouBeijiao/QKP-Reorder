from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INNOVATION_DIR = PROJECT_ROOT / "results" / "innovation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "innovation"
LOG_DIR = PROJECT_ROOT / "logs" / "innovation"
DOC_DIR = PROJECT_ROOT / "docs" / "innovation"

LOG_PATH = LOG_DIR / "stage8_final_assets_log.txt"

MASTER_SUMMARY_PATH = INNOVATION_DIR / "innovation_master_summary.csv"
REPORT_TABLES_PATH = INNOVATION_DIR / "innovation_report_tables.md"
KEY_FINDINGS_PATH = INNOVATION_DIR / "innovation_key_findings.md"
EXPERIMENT_SUMMARY_PATH = INNOVATION_DIR / "innovation_experiment_summary.md"
REPORT_NOTES_PATH = INNOVATION_DIR / "innovation_report_notes.md"
PPT_OUTLINE_PATH = INNOVATION_DIR / "innovation_ppt_outline.md"
FIGURE_INDEX_CSV_PATH = INNOVATION_DIR / "innovation_figure_index.csv"
FIGURE_INDEX_MD_PATH = INNOVATION_DIR / "innovation_figure_index.md"
CHINESE_CASE_STUDY_PATH = INNOVATION_DIR / "innovation_chinese_case_study.md"
STAGE8_DOC_PATH = DOC_DIR / "阶段8_结果汇总图表与报告PPT支撑执行文档.md"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def read_csv(name: str) -> pd.DataFrame:
    path = INNOVATION_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    logging.info("Reading %s", path)
    return pd.read_csv(path)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def select_rows(df: pd.DataFrame, method_col: str = "method", methods: Iterable[str] = ()) -> pd.DataFrame:
    methods = list(methods)
    if not methods:
        return df.copy()
    out = df[df[method_col].isin(methods)].copy()
    out[method_col] = pd.Categorical(out[method_col], categories=methods, ordered=True)
    return out.sort_values(method_col).reset_index(drop=True)


def build_master_summary(
    nq: pd.DataFrame,
    position: pd.DataFrame,
    longbench_official: pd.DataFrame,
    ablation: pd.DataFrame,
    budget: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    ours_nq = nq[nq["method"] == "ours_full"].iloc[0]
    best_nq = nq.sort_values("f1_avg", ascending=False).iloc[0]
    rows.append(
        {
            "section": "NQ 100",
            "scope": "100 samples, token budget 750",
            "primary_metric": "NQ-style F1",
            "ours_result": round(float(ours_nq["nq_style_f1"]), 3),
            "comparison_reference": best_nq["method"],
            "comparison_reference_result": round(float(best_nq["nq_style_f1"]), 3),
            "report_message": "Ours is a lightweight sixth method but does not beat BM25 on NQ 100.",
        }
    )

    pos20 = position[(position["method"] == "ours_full") & (position["answer_position"] == 20)].iloc[0]
    trunc20 = position[(position["method"] == "truncate") & (position["answer_position"] == 20)].iloc[0]
    rows.append(
        {
            "section": "Lost-in-the-Middle",
            "scope": "Position 20, 20 samples",
            "primary_metric": "contains_answer",
            "ours_result": round(float(pos20["contains_answer_avg"]), 3),
            "comparison_reference": "truncate",
            "comparison_reference_result": round(float(trunc20["contains_answer_avg"]), 3),
            "report_message": "Ours is much more stable than truncate when answer evidence is late.",
        }
    )

    lb_ours = longbench_official[longbench_official["method"] == "ours_full"].copy()
    rows.append(
        {
            "section": "LongBench official",
            "scope": "hotpotqa, 2wikimqa, multifieldqa_en; 50 samples each",
            "primary_metric": "official_score average",
            "ours_result": round(float(lb_ours["official_score"].mean()), 3),
            "comparison_reference": "BM25 and LongLLMLingua",
            "comparison_reference_result": "",
            "report_message": "Ours trails BM25 overall, beats LongLLMLingua on hotpotqa and multifieldqa_en, and trails it on 2wikimqa.",
        }
    )

    best_ablation = ablation.sort_values("nq_style_f1_avg", ascending=False).iloc[0]
    rows.append(
        {
            "section": "Ablation",
            "scope": "100 position-sensitivity samples, token budget 1500",
            "primary_metric": "NQ-style F1",
            "ours_result": round(float(best_ablation["nq_style_f1_avg"]), 3),
            "comparison_reference": best_ablation["method"],
            "comparison_reference_result": round(float(best_ablation["nq_style_f1_avg"]), 3),
            "report_message": "Keyword protection contributes the clearest gain; naive middle-aware reordering is not consistently positive.",
        }
    )

    budget1500 = budget[budget["token_budget"] == 1500].iloc[0]
    budget2000 = budget[budget["token_budget"] == 2000].iloc[0]
    rows.append(
        {
            "section": "Budget sensitivity",
            "scope": "ours_full, budgets 500/750/1000/1500/2000",
            "primary_metric": "NQ-style F1 and API tokens",
            "ours_result": round(float(budget1500["nq_style_f1_avg"]), 3),
            "comparison_reference": "2000 token budget",
            "comparison_reference_result": round(float(budget2000["nq_style_f1_avg"]), 3),
            "report_message": "1500 tokens is the recommended cost-quality trade-off; 2000 adds small quality gain with higher cost.",
        }
    )

    return pd.DataFrame(rows)


def build_report_tables(
    nq: pd.DataFrame,
    position: pd.DataFrame,
    longbench_official: pd.DataFrame,
    ablation: pd.DataFrame,
    budget: pd.DataFrame,
    master: pd.DataFrame,
) -> str:
    nq_table = nq[[
        "method",
        "contains_answer_avg",
        "nq_style_f1",
        "compressed_tokens_avg",
        "token_saving_ratio_avg",
        "api_total_tokens_avg",
    ]].copy()
    nq_table.columns = [
        "method",
        "contains_answer",
        "nq_style_f1",
        "compressed_tokens",
        "token_saving",
        "api_tokens",
    ]
    for col in nq_table.columns[1:]:
        nq_table[col] = nq_table[col].map(lambda x: fmt(x, 3))

    pos_focus = position[
        (position["answer_position"].isin([15, 20]))
        & (position["method"].isin(["truncate", "bm25", "longllmlingua", "ours_full"]))
    ][[
        "answer_position",
        "method",
        "contains_answer_avg",
        "nq_style_f1_avg",
    ]].copy()
    pos_focus.columns = ["position", "method", "contains_answer", "nq_style_f1"]
    for col in ["contains_answer", "nq_style_f1"]:
        pos_focus[col] = pos_focus[col].map(lambda x: fmt(x, 3))

    lb = longbench_official[[
        "task_name",
        "method",
        "official_score",
        "avg_compressed_tokens",
        "avg_token_saving_ratio",
    ]].copy()
    lb.columns = ["task", "method", "official_score", "compressed_tokens", "token_saving"]
    for col in ["official_score", "compressed_tokens", "token_saving"]:
        lb[col] = lb[col].map(lambda x: fmt(x, 3))

    ab = ablation[[
        "method",
        "contains_answer_avg",
        "nq_style_f1_avg",
        "api_total_tokens_avg",
        "answer_time_avg",
    ]].copy()
    ab.columns = ["method", "contains_answer", "nq_style_f1", "api_tokens", "answer_time"]
    for col in ab.columns[1:]:
        ab[col] = ab[col].map(lambda x: fmt(x, 3))

    bud = budget[[
        "token_budget",
        "contains_answer_avg",
        "nq_style_f1_avg",
        "api_total_tokens_avg",
        "token_saving_ratio_avg",
    ]].copy()
    bud.columns = ["token_budget", "contains_answer", "nq_style_f1", "api_tokens", "token_saving"]
    for col in bud.columns[1:]:
        bud[col] = bud[col].map(lambda x: fmt(x, 3))

    sections = [
        "# Innovation Report Tables",
        "",
        "## One-slide Executive Summary",
        md_table(master, list(master.columns)),
        "",
        "## NQ 100 Main Comparison",
        md_table(nq_table, list(nq_table.columns)),
        "",
        "## Lost-in-the-Middle Late-position Focus",
        md_table(pos_focus, list(pos_focus.columns)),
        "",
        "## LongBench Official Evaluation",
        md_table(lb, list(lb.columns)),
        "",
        "## Ablation",
        md_table(ab, list(ab.columns)),
        "",
        "## Budget Sensitivity",
        md_table(bud, list(bud.columns)),
        "",
    ]
    return "\n".join(sections)


def build_figure_index() -> tuple[pd.DataFrame, str]:
    figure_roles = {
        "fig_ours_nq_100_f1.png": "NQ 100 main comparison: F1",
        "fig_ours_nq_100_tokens.png": "NQ 100 token cost comparison",
        "fig_ours_position_f1.png": "Lost-in-the-Middle position sensitivity: F1",
        "fig_ours_position_contains_answer.png": "Lost-in-the-Middle evidence retention",
        "fig_ours_longbench_official_score.png": "LongBench official score comparison",
        "fig_ours_ablation_f1.png": "Ablation: component contribution",
        "fig_ours_ablation_deep_position_contains_answer.png": "Ablation: late-position evidence retention",
        "fig_ours_budget_quality_cost_curve.png": "Budget sensitivity: quality-cost curve",
        "fig_ours_budget_token_saving.png": "Budget sensitivity: token saving",
    }

    rows = []
    for path in sorted(FIGURE_DIR.glob("*.png")):
        rows.append(
            {
                "figure": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "ppt_use": figure_roles.get(path.name, "Supplementary figure"),
            }
        )
    df = pd.DataFrame(rows)

    md = ["# Innovation Figure Index", ""]
    md.append(md_table(df, ["figure", "relative_path", "ppt_use"]))
    md.append("")
    return df, "\n".join(md)


def build_chinese_case_study() -> str:
    from qkp_reorder.ours_utils import (
        compute_protection_bonus,
        extract_entity_like_terms,
        extract_numbers,
        extract_question_keywords,
        is_jieba_available,
        split_into_segments,
        tokenize_for_bm25,
    )

    question = "2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？"
    context = (
        "上海团队介绍了多模态检索系统。"
        "2024年北京人工智能大会上，DeepSeek发布了DeepSeek-R1相关工具，重点展示中文推理能力。"
        "随后NASA报告了AI_report-2024中的英文实验，样本数为1,234，增长12.5%。"
    )
    segments = split_into_segments(context)
    keywords = extract_question_keywords(question)

    rows = []
    for segment in segments:
        rows.append(
            {
                "segment": segment,
                "tokens": ", ".join(tokenize_for_bm25(segment)),
                "numbers": ", ".join(extract_numbers(segment)),
                "entities": ", ".join(extract_entity_like_terms(segment)),
                "protection_bonus": compute_protection_bonus(segment, keywords),
            }
        )
    case_df = pd.DataFrame(rows)

    return "\n".join(
        [
            "# Chinese Segmentation Case Study",
            "",
            f"- jieba_available: `{is_jieba_available()}`",
            f"- question: {question}",
            f"- extracted_keywords: `{', '.join(keywords)}`",
            "",
            "## Segment-level Analysis",
            "",
            md_table(case_df, ["segment", "tokens", "numbers", "entities", "protection_bonus"]),
            "",
            "## Report Message",
            "",
            "This case shows that the Ours method includes reusable Chinese sentence segmentation, mixed Chinese/English tokenization, number extraction, entity-like term extraction, and keyword/entity/number protection. It should be presented as an extension capability rather than as the main reason for English benchmark gains.",
            "",
        ]
    )


def build_key_findings() -> str:
    return """# Innovation Key Findings

## Main Results

1. Ours is a fully implemented sixth method: Chinese/English segmentation, question-aware BM25, keyword/entity/number protection, middle-aware reordering, and token-budget truncation.
2. On NQ 100, Ours is valid and lightweight but does not outperform BM25 or LongLLMLingua. The report should describe this honestly as a low-cost interpretable method rather than a universally stronger compressor.
3. On Lost-in-the-Middle, Ours is much more robust than truncate when the answer document is late. At position 20, Ours keeps a contains-answer rate of 0.55 in the main comparison, while truncate drops to 0.20.
4. On LongBench official evaluation, Ours trails BM25 overall, beats LongLLMLingua on hotpotqa and multifieldqa_en, and trails LongLLMLingua on 2wikimqa. It should be reported as competitive but not dominant.
5. Ablation shows that keyword protection is the clearest positive component. The current middle-aware reordering is interpretable but not consistently beneficial, and should be described as a limitation/future-work point.
6. Budget sensitivity shows monotonic quality improvement from 500 to 2000 tokens, with diminishing returns. A 1500-token budget is the best reporting trade-off for Ours in the position-sensitivity setting.
7. The Chinese segmentation module is implemented and demonstrated with a case study, but the main benchmark evidence remains mostly English; avoid over-claiming Chinese benchmark improvements.

## Recommended One-sentence Conclusion

Under the unified DeepSeek-based evaluation pipeline, Ours provides an interpretable, low-cost compression variant that improves over naive truncation in late-evidence settings and exposes useful component-level findings, while BM25 and LongLLMLingua remain strong baselines depending on task type.
"""


def build_experiment_summary(master: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Innovation Experiment Summary",
            "",
            "## Scope",
            "",
            "- Stage 2: Chinese/English segmentation and keyword/entity/number protection utilities.",
            "- Stage 3: Ours compressor variants registered as reusable methods.",
            "- Stage 4: NQ 100 main comparison with `ours_full`.",
            "- Stage 5: Lost-in-the-Middle answer-position sensitivity with `ours_full`.",
            "- Stage 6: LongBench three-task experiment and official LongBench evaluation.",
            "- Stage 7: Ablation and budget sensitivity analysis.",
            "- Stage 8: Report/PPT-ready summary assets.",
            "",
            "## Master Summary",
            "",
            md_table(master, list(master.columns)),
            "",
            "## Evaluation Wording",
            "",
            "- NQ and Lost-in-the-Middle use NQ-style normalized short-answer EM/F1, not full official NaturalQuestions evaluation.",
            "- LongBench reporting should prefer official `eval.py` scores.",
            "- All innovation answer generation uses the same DeepSeek-based pipeline as the reproduction experiments.",
            "- Cost is reported as an API-token proxy, not as a monetary estimate.",
            "",
        ]
    )


def build_report_notes() -> str:
    return """# Innovation Report Notes

## Suggested Report Structure

1. Reproduction baseline and unified DeepSeek pipeline.
2. Motivation: truncation is position-sensitive; BM25 is strong but lexical; LLMLingua/LongLLMLingua add compressor cost and are not always dominant in this setup.
3. Ours method: segmentation, question-aware BM25, keyword protection, middle-aware reordering, fixed budget truncation.
4. Main results: NQ 100, Lost-in-the-Middle, LongBench.
5. Ablation: keyword protection gives the clearest gain; middle-aware reordering needs improvement.
6. Budget sensitivity: 1500 tokens is the recommended reporting budget.
7. Chinese case study: implemented capability and extensibility.
8. Limitations and future work.

## PPT-ready Claims

- Safe claim: Ours is interpretable and low-cost, and it clearly improves over truncate in late-evidence settings.
- Safe claim: BM25 remains a strong baseline, especially for lexical multi-hop QA.
- Safe claim: Keyword/entity/number protection is the most useful added component in our ablation.
- Avoid: "Ours beats all baselines" or "Chinese segmentation explains the English benchmark gains."

## Limitations

- Ours relies on lexical overlap and heuristic protection, so it can miss semantically relevant evidence when wording differs strongly.
- The middle-aware reordering design is simple and can disturb useful context order.
- DeepSeek answer generation introduces stochasticity and retry effects.
- Token cost is a proxy based on logged API usage, not a fixed monetary calculation.
- The Chinese module is demonstrated by case study; the main benchmark set is still mostly English.
"""


def build_ppt_outline() -> str:
    return """# Innovation PPT Outline

## Slide 1: Task Split and Innovation Goal

- Reproduction is complete; this part adds a sixth method, Ours.
- Goal: improve evidence retention under compression while keeping the method lightweight and interpretable.

## Slide 2: Method Overview

- Chinese/English compatible segmentation.
- Question-aware BM25 segment scoring.
- Keyword/entity/number protection.
- Middle-aware reordering.
- Same token-budget truncation as baselines.

Recommended figure: `figures/innovation/fig_ours_nq_100_f1.png`

## Slide 3: NQ 100 Main Result

- Ours is valid and low-cost but does not beat BM25 on this fixed-position NQ setting.
- Use this slide to show honest comparison and avoid over-claiming.

Recommended figures: `fig_ours_nq_100_f1.png`, `fig_ours_nq_100_tokens.png`

## Slide 4: Lost-in-the-Middle Result

- Main story: Ours is more robust than truncate when evidence is late.
- At position 20, truncate is weak; Ours keeps substantially more answer evidence.

Recommended figures: `fig_ours_position_contains_answer.png`, `fig_ours_position_f1.png`

## Slide 5: LongBench Official Evaluation

- Report official scores, not only simplified string metrics.
- Ours is competitive but not dominant: it trails BM25 overall, beats LongLLMLingua on hotpotqa and multifieldqa_en, and trails it on 2wikimqa.

Recommended figure: `fig_ours_longbench_official_score.png`

## Slide 6: Ablation

- Keyword protection is the strongest component.
- Current middle-aware reordering is not consistently positive; this is a useful analysis result.

Recommended figures: `fig_ours_ablation_f1.png`, `fig_ours_ablation_deep_position_contains_answer.png`

## Slide 7: Budget Sensitivity

- Higher budget improves quality, but returns diminish.
- 1500 tokens is a good quality-cost trade-off.

Recommended figure: `fig_ours_budget_quality_cost_curve.png`

## Slide 8: Chinese Case Study

- Show Chinese segmentation, keyword extraction, number/entity protection.
- Explain it as method extensibility rather than primary benchmark gain.

Recommended source: `results/innovation/innovation_chinese_case_study.md`

## Slide 9: Limitations and Future Work

- BM25 remains strong.
- Middle-aware reordering needs a better design.
- Add semantic retrieval or learned reranking in future work.
"""


def build_stage8_doc(master: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# 阶段8：结果汇总、图表与报告 PPT 支撑执行文档",
            "",
            "## 阶段目标",
            "",
            "将阶段2-7的创新实验结果整理为报告和PPT可直接使用的表格、图表索引、结论、局限性说明和讲稿提纲。本阶段不调用 DeepSeek API，只复用已有结果文件。",
            "",
            "## 新增脚本",
            "",
            "- `llmlingua/src/summarize_innovation_stage8_final_assets.py`：读取创新结果 CSV 与图表目录，生成报告/PPT支撑材料。",
            "",
            "## 实际运行命令",
            "",
            "```powershell",
            "cd path/to/qkp-reorder",
            "python src\\summarize_innovation_stage8_final_assets.py",
            "```",
            "",
            "## 输出文件",
            "",
            "- `results/innovation/innovation_master_summary.csv`",
            "- `results/innovation/innovation_report_tables.md`",
            "- `results/innovation/innovation_key_findings.md`",
            "- `results/innovation/innovation_experiment_summary.md`",
            "- `results/innovation/innovation_report_notes.md`",
            "- `results/innovation/innovation_ppt_outline.md`",
            "- `results/innovation/innovation_figure_index.csv`",
            "- `results/innovation/innovation_figure_index.md`",
            "- `results/innovation/innovation_chinese_case_study.md`",
            "- `logs/innovation/stage8_final_assets_log.txt`",
            "",
            "## 汇总表",
            "",
            md_table(master, list(master.columns)),
            "",
            "## 报告/PPT建议",
            "",
            "- 主线不要写成“全面超越”，而应写成“轻量、可解释、在后位证据场景优于简单截断”。",
            "- NQ与Lost-in-the-Middle指标称为“NQ-style normalized short-answer EM/F1”。",
            "- LongBench优先报告官方 `eval.py` 分数。",
            "- 中文分句作为可扩展能力和case study展示，不解释为英文主实验收益来源。",
            "- 消融实验强调 keyword protection 是最大正向组件，middle-aware reordering 当前实现效果有限。",
            "",
            "## 验收结果",
            "",
            "- 创新结果均有索引：见 `innovation_master_summary.csv` 与 `innovation_report_tables.md`。",
            "- 图表索引已生成：见 `innovation_figure_index.csv` / `.md`。",
            "- 结论不夸大：见 `innovation_key_findings.md` 与 `innovation_report_notes.md`。",
            "- 已明确区分 reproduction 与 innovation，并保留 API 成本、评估口径、模型差异限制说明。",
            "",
        ]
    )


def main() -> None:
    setup_logging()
    INNOVATION_DIR.mkdir(parents=True, exist_ok=True)

    nq = read_csv("ours_nq_100_comparison.csv")
    position = read_csv("ours_position_comparison.csv")
    longbench_official = read_csv("ours_longbench_official_eval_comparison.csv")
    ablation = read_csv("ours_ablation_summary.csv")
    budget = read_csv("ours_budget_sensitivity_summary.csv")

    master = build_master_summary(nq, position, longbench_official, ablation, budget)
    master.to_csv(MASTER_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    logging.info("Wrote %s", MASTER_SUMMARY_PATH)

    REPORT_TABLES_PATH.write_text(
        build_report_tables(nq, position, longbench_official, ablation, budget, master),
        encoding="utf-8",
    )
    KEY_FINDINGS_PATH.write_text(build_key_findings(), encoding="utf-8")
    EXPERIMENT_SUMMARY_PATH.write_text(build_experiment_summary(master), encoding="utf-8")
    REPORT_NOTES_PATH.write_text(build_report_notes(), encoding="utf-8")
    PPT_OUTLINE_PATH.write_text(build_ppt_outline(), encoding="utf-8")
    CHINESE_CASE_STUDY_PATH.write_text(build_chinese_case_study(), encoding="utf-8")

    figure_df, figure_md = build_figure_index()
    figure_df.to_csv(FIGURE_INDEX_CSV_PATH, index=False, encoding="utf-8-sig")
    FIGURE_INDEX_MD_PATH.write_text(figure_md, encoding="utf-8")

    STAGE8_DOC_PATH.write_text(build_stage8_doc(master), encoding="utf-8")

    for path in [
        REPORT_TABLES_PATH,
        KEY_FINDINGS_PATH,
        EXPERIMENT_SUMMARY_PATH,
        REPORT_NOTES_PATH,
        PPT_OUTLINE_PATH,
        FIGURE_INDEX_CSV_PATH,
        FIGURE_INDEX_MD_PATH,
        CHINESE_CASE_STUDY_PATH,
        STAGE8_DOC_PATH,
    ]:
        logging.info("Wrote %s", path)

    logging.info("Stage 8 final assets complete.")


if __name__ == "__main__":
    main()

