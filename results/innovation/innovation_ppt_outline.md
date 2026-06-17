# Innovation PPT Outline

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
