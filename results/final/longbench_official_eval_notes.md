# LongBench Official Evaluation Notes

This file summarizes the supplemental official LongBench evaluation.

## Evaluation Source

- Official script: `external/LongBench/LongBench/eval.py`
- Predictions converted from: `results/stage7/longbench_small_results.csv`
- Evaluation mode: standard LongBench eval, not LongBench-E
- No additional DeepSeek API calls were made.

## Official Scores

| Task | Method | Official Score |
|---|---|---:|
| hotpotqa | original | 67.37 |
| hotpotqa | truncate | 19.58 |
| hotpotqa | bm25 | 60.94 |
| hotpotqa | llmlingua | 36.84 |
| hotpotqa | longllmlingua | 45.57 |
| 2wikimqa | original | 71.08 |
| 2wikimqa | truncate | 26.98 |
| 2wikimqa | bm25 | 65.71 |
| 2wikimqa | llmlingua | 59.93 |
| 2wikimqa | longllmlingua | 62.68 |
| multifieldqa_en | original | 57.80 |
| multifieldqa_en | truncate | 50.12 |
| multifieldqa_en | bm25 | 52.60 |
| multifieldqa_en | llmlingua | 45.45 |
| multifieldqa_en | longllmlingua | 42.59 |

## Notes

- The official LongBench scores are task-specific and should be preferred when reporting LongBench results.
- The previous `contains_answer` and `simple_token_f1` metrics remain useful for cross-experiment consistency, but they are simplified metrics.
- MultiFieldQA-en may show differences between simplified and official metrics because open-ended answers are not always captured by strict string matching.
- This evaluation reuses existing predictions; it does not change the answer model or compressed contexts.
