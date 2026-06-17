# Final Experiment Summary

This document summarizes the final reproduction results for the LongLLMLingua course project.

## Experiment 1: NaturalQuestions Multi-document QA

### Setting

- Dataset: Lost-in-the-Middle NQ 10-document setting
- Samples: 100
- Documents per sample: 10
- Answer document position: 5
- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua
- Answer model: DeepSeek API

### Summary Table

| Method | Avg. Compressed Tokens | Token Saving | Contains Answer | Avg. F1 |
|---|---:|---:|---:|---:|
| bm25 | 756.430 | 0.497 | 0.620 | 0.531 |
| llmlingua | 720.300 | 0.522 | 0.600 | 0.494 |
| longllmlingua | 720.300 | 0.522 | 0.630 | 0.508 |
| original | 1510.090 | 0.000 | 0.630 | 0.481 |
| truncate | 750.000 | 0.502 | 0.620 | 0.502 |

### Key Observation

- LongLLMLingua reduces the input context to 720.300 tokens on average, with a token saving ratio of 0.522.
- Compared with Original, LongLLMLingua keeps a comparable contains-answer rate (0.630 vs. 0.630) while substantially reducing input tokens.

## Experiment 2: Lost in the Middle Position Sensitivity

### Setting

- Dataset: Lost-in-the-Middle NQ 20-document setting
- Positions: 1, 5, 10, 15, 20
- Samples per position: 20
- Total samples: 100
- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua

### Summary Table

| Position | Method | Contains Answer | Avg. F1 | Avg. Tokens |
|---:|---|---:|---:|---:|
| 1 | bm25 | 0.700 | 0.396 | 1513.500 |
| 1 | llmlingua | 0.700 | 0.511 | 1428.150 |
| 1 | longllmlingua | 0.750 | 0.504 | 1428.150 |
| 1 | original | 0.800 | 0.424 | 3019.700 |
| 1 | truncate | 0.900 | 0.500 | 1500.000 |
| 5 | bm25 | 0.700 | 0.449 | 1513.550 |
| 5 | llmlingua | 0.750 | 0.534 | 1426.200 |
| 5 | longllmlingua | 0.750 | 0.500 | 1426.200 |
| 5 | original | 0.800 | 0.521 | 3019.700 |
| 5 | truncate | 0.900 | 0.557 | 1500.000 |
| 10 | bm25 | 0.700 | 0.474 | 1513.550 |
| 10 | llmlingua | 0.700 | 0.444 | 1427.800 |
| 10 | longllmlingua | 0.750 | 0.461 | 1427.800 |
| 10 | original | 0.750 | 0.495 | 3019.700 |
| 10 | truncate | 0.700 | 0.487 | 1500.000 |
| 15 | bm25 | 0.500 | 0.472 | 1513.550 |
| 15 | llmlingua | 0.700 | 0.489 | 1427.250 |
| 15 | longllmlingua | 0.750 | 0.502 | 1427.250 |
| 15 | original | 0.650 | 0.465 | 3019.700 |
| 15 | truncate | 0.250 | 0.236 | 1500.000 |
| 20 | bm25 | 0.650 | 0.411 | 1513.900 |
| 20 | llmlingua | 0.850 | 0.514 | 1425.650 |
| 20 | longllmlingua | 0.800 | 0.546 | 1425.650 |
| 20 | original | 0.750 | 0.445 | 3020.400 |
| 20 | truncate | 0.200 | 0.189 | 1500.000 |

### Key Observation

- At Position 15, Truncate obtains a contains-answer rate of 0.250, while LongLLMLingua obtains 0.750.
- At Position 20, Truncate obtains a contains-answer rate of 0.200, while LongLLMLingua obtains 0.800.
- This result suggests that simple truncation is highly sensitive to answer position, whereas LongLLMLingua is more stable when the answer document is placed later in the context.

## Experiment 3: LongBench Subset

### Setting

- Dataset: LongBench
- Tasks: HotpotQA, 2WikiMultihopQA, MultiFieldQA-en
- Samples per task: 50
- Total samples: 150
- Methods: Original, Truncate, BM25, LLMLingua, LongLLMLingua

### Summary Table

| Task | Method | Contains Answer | Avg. F1 | Avg. Tokens |
|---|---|---:|---:|---:|
| 2wikimqa | bm25 | 0.680 | 0.657 | 1979.180 |
| 2wikimqa | llmlingua | 0.700 | 0.599 | 1882.400 |
| 2wikimqa | longllmlingua | 0.700 | 0.627 | 1882.400 |
| 2wikimqa | original | 0.760 | 0.711 | 7024.160 |
| 2wikimqa | truncate | 0.360 | 0.270 | 1969.340 |
| hotpotqa | bm25 | 0.500 | 0.609 | 2007.520 |
| hotpotqa | llmlingua | 0.300 | 0.368 | 1955.200 |
| hotpotqa | longllmlingua | 0.360 | 0.456 | 1955.200 |
| hotpotqa | original | 0.560 | 0.674 | 13294.760 |
| hotpotqa | truncate | 0.180 | 0.196 | 2000.000 |
| multifieldqa_en | bm25 | 0.220 | 0.526 | 1994.640 |
| multifieldqa_en | llmlingua | 0.220 | 0.455 | 1996.860 |
| multifieldqa_en | longllmlingua | 0.200 | 0.426 | 1996.860 |
| multifieldqa_en | original | 0.320 | 0.578 | 7068.860 |
| multifieldqa_en | truncate | 0.280 | 0.501 | 1986.340 |

### Key Observation

- On multi-hop QA tasks such as HotpotQA and 2WikiMultihopQA, Truncate is generally weaker, indicating that simple prefix preservation may discard useful evidence.
- BM25 remains a competitive baseline for lexical multi-hop QA retrieval.
- MultiFieldQA-en shows lower contains-answer rates across methods, suggesting that strict string matching may be too conservative for open-ended answers.

### Supplemental Official LongBench Evaluation

In addition to the simplified `contains_answer` and `simple_token_f1` metrics, we converted the existing LongBench predictions to the official LongBench `eval.py` format and re-evaluated them without making additional DeepSeek API calls.

Official evaluation outputs:

```text
results/final/longbench_official_eval_summary.csv
results/final/longbench_official_eval_comparison.csv
results/final/longbench_official_eval_notes.md
```

The official scores should be preferred when reporting LongBench task-specific results, while the simplified metrics remain useful for cross-experiment consistency.

## Supplementary Analysis: NQ-style Normalized EM/F1

For the NQ and Lost-in-the-Middle experiments, we further added NQ-style normalized short-answer evaluation. This evaluation lowercases answers, removes punctuation and English articles, normalizes whitespace, and computes exact match and token-level F1 over multiple gold answers.

This supplemental evaluation reuses existing predictions and does not call DeepSeek API again. It should be described as NQ-style normalized short-answer EM/F1, not as the full original NaturalQuestions official evaluation.

The results are saved in:

```text
results/final/nq_style_eval_exp1_results.csv
results/final/nq_style_eval_exp1_summary.csv
results/final/nq_style_eval_exp2_results.csv
results/final/nq_style_eval_exp2_summary.csv
results/final/nq_style_eval_notes.md
```

Representative figures are saved in:

```text
figures/final/fig_nq_style_exp1_em.png
figures/final/fig_nq_style_exp1_f1.png
figures/final/fig_nq_style_exp2_position_em.png
figures/final/fig_nq_style_exp2_position_f1.png
```

## Supplementary Analysis: Cost-Effectiveness

In addition to answer quality, we further analyzed token-level cost-effectiveness using the API usage fields stored in the result CSV files. This analysis does not involve additional API calls.

We use average total API tokens as a proxy for cost, and report derived metrics such as F1 per 1k tokens and tokens per correct answer. These metrics help evaluate whether a compression method achieves a better balance between answer quality and token consumption.

The results are saved in:

```text
results/final/cost_effectiveness_by_experiment.csv
results/final/cost_effectiveness_overall.csv
results/final/cost_effectiveness_notes.md
```

Representative figures are saved in:

```text
figures/final/fig_cost_avg_total_tokens.png
figures/final/fig_cost_avg_f1.png
figures/final/fig_cost_f1_per_1k_tokens.png
figures/final/fig_cost_tokens_per_correct_answer.png
```

## Implementation Differences and Limitations

This reproduction should be interpreted as a course-project reproduction under a unified experimental pipeline.

Important implementation notes:

1. The answer model is DeepSeek API rather than the exact answer model used in the original paper.
2. The compressor uses a CPU-compatible LLMLingua-2 model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`.
3. LLMLingua and LongLLMLingua may produce the same compressed token length under this implementation.
4. Evaluation uses simplified `contains_answer` and `simple_token_f1` metrics for cross-experiment consistency.
5. LongBench official `eval.py` scores are additionally provided for the LongBench subset.
6. NQ and Lost-in-the-Middle are additionally reported with NQ-style normalized short-answer EM/F1.
7. Results should be interpreted as relative comparisons under the same answer model and evaluation pipeline.
