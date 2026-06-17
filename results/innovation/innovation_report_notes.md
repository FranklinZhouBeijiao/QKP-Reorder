# Innovation Report Notes

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
