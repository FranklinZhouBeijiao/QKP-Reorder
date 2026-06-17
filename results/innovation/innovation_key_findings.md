# Innovation Key Findings

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
