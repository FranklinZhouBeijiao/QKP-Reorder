# Cost-Effectiveness Analysis Notes

This analysis reuses existing experiment CSV files and does not call DeepSeek API again.

## Metrics

- `avg_api_total_tokens`: average API token usage parsed from the `usage` field.
- `f1_per_1k_tokens`: average F1 divided by average total API tokens per 1,000 tokens.
- `contains_answer_per_1k_tokens`: contains-answer rate divided by average total API tokens per 1,000 tokens.
- `tokens_per_correct_answer`: average total API tokens required per strict answer hit.

## Overall Summary

| Method | Avg Total Tokens | Contains Answer | Avg F1 | F1 per 1k Tokens | Tokens per Correct Answer |
|---|---:|---:|---:|---:|---:|
| original | 5426.34 | 0.629 | 0.552 | 0.102 | 8632.81 |
| truncate | 1728.58 | 0.463 | 0.394 | 0.228 | 3734.59 |
| bm25 | 1782.39 | 0.563 | 0.534 | 0.299 | 3166.68 |
| llmlingua | 1866.54 | 0.557 | 0.487 | 0.261 | 3350.20 |
| longllmlingua | 1907.47 | 0.577 | 0.504 | 0.264 | 3305.01 |

## Interpretation Guidance

- Original often has strong answer quality but highest token cost.
- Truncate reduces token cost but may lose important evidence in long-context tasks.
- BM25 can be cost-effective when lexical evidence is strong.
- LongLLMLingua should be interpreted by balancing quality and token reduction, not by accuracy alone.
- These are token proxy costs, not exact monetary costs.
