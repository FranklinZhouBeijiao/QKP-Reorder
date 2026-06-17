# NQ-style Normalized EM/F1 Evaluation Notes

This supplemental evaluation reuses existing predictions and does not call DeepSeek API again.

## Evaluation Definition

- Lowercase normalization
- Punctuation removal
- English article removal: a / an / the
- Whitespace normalization
- Exact Match over normalized strings
- Token-level F1 over normalized tokens
- For multiple gold answers, the maximum score is used.

## Scope

This is not the full original NaturalQuestions official evaluation. It is a normalized short-answer matching protocol suitable for Lost-in-the-Middle / NQ-style QA data.

## Experiment 1 Summary

| Method | Contains Answer | Simple F1 | NQ-style EM | NQ-style F1 |
|---|---:|---:|---:|---:|
| original | 0.630 | 0.481 | 0.300 | 0.481 |
| truncate | 0.620 | 0.502 | 0.340 | 0.502 |
| bm25 | 0.620 | 0.531 | 0.350 | 0.531 |
| llmlingua | 0.600 | 0.494 | 0.310 | 0.494 |
| longllmlingua | 0.630 | 0.508 | 0.320 | 0.508 |

## Reporting Guidance

- For NQ and Lost-in-the-Middle, report NQ-style EM/F1 as normalized QA metrics.
- Keep contains_answer and simple F1 as auxiliary project-wide metrics.
- Do not call this a full official NaturalQuestions evaluation.
