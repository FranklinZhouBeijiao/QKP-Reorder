# Key Findings

## Main Findings

1. In the NQ 100-sample experiment, LongLLMLingua substantially reduces input tokens while maintaining comparable answer quality.
2. In the position sensitivity experiment, Truncate drops sharply when the answer document is placed later, while LongLLMLingua is more stable.
3. In the LongBench subset, task type matters: BM25 is competitive on multi-hop QA, while MultiFieldQA-en is difficult to evaluate with strict contains-answer matching.
4. Across experiments, compression can reduce prompt length significantly, but answer quality depends on task structure and evidence distribution.
5. As a supplemental check, the existing LongBench predictions were re-evaluated with the official LongBench `eval.py` without additional DeepSeek API calls.
6. Cost-effectiveness analysis shows that compression methods should be compared not only by answer quality, but also by token usage. In the current token-proxy analysis, BM25 obtains the highest overall F1 per 1k API tokens, while LongLLMLingua offers a stronger balance than simple truncation.
7. NQ-style normalized EM/F1 was added for the NQ and Lost-in-the-Middle experiments, providing a more appropriate normalized short-answer matching protocol than raw string containment.

## Notes for Report Writing

- Avoid claiming exact reproduction of the original paper's numerical results.
- Use the phrasing: "under our DeepSeek-based reproduction setting" or "in our unified evaluation pipeline".
- Clearly state that LongBench is reported with both simplified contains-answer/token-F1 metrics and supplemental official LongBench scores.
- For NQ and Lost-in-the-Middle, call the supplemental metrics "NQ-style normalized short-answer EM/F1", not full official NaturalQuestions evaluation.
- Treat cost as a token proxy rather than a real monetary estimate unless API prices are explicitly fixed by date.
