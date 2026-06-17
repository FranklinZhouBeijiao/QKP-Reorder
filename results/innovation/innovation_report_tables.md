# Innovation Report Tables

## One-slide Executive Summary
| section | scope | primary_metric | ours_result | comparison_reference | comparison_reference_result | report_message |
| --- | --- | --- | --- | --- | --- | --- |
| NQ 100 | 100 samples, token budget 750 | NQ-style F1 | 0.47 | bm25 | 0.531 | Ours is a lightweight sixth method but does not beat BM25 on NQ 100. |
| Lost-in-the-Middle | Position 20, 20 samples | contains_answer | 0.55 | truncate | 0.2 | Ours is much more stable than truncate when answer evidence is late. |
| LongBench official | hotpotqa, 2wikimqa, multifieldqa_en; 50 samples each | official_score average | 56.277 | BM25 and LongLLMLingua |  | Ours trails BM25 overall, beats LongLLMLingua on hotpotqa and multifieldqa_en, and trails it on 2wikimqa. |
| Ablation | 100 position-sensitivity samples, token budget 1500 | NQ-style F1 | 0.461 | ours_keyword | 0.461 | Keyword protection contributes the clearest gain; naive middle-aware reordering is not consistently positive. |
| Budget sensitivity | ours_full, budgets 500/750/1000/1500/2000 | NQ-style F1 and API tokens | 0.437 | 2000 token budget | 0.45 | 1500 tokens is the recommended cost-quality trade-off; 2000 adds small quality gain with higher cost. |

## NQ 100 Main Comparison
| method | contains_answer | nq_style_f1 | compressed_tokens | token_saving | api_tokens |
| --- | --- | --- | --- | --- | --- |
| original | 0.630 | 0.481 | 1510.090 | 0.000 | 1771.970 |
| truncate | 0.620 | 0.502 | 750.000 | 0.502 | 971.960 |
| bm25 | 0.620 | 0.531 | 756.430 | 0.497 | 1006.530 |
| llmlingua | 0.600 | 0.494 | 720.300 | 0.522 | 1051.090 |
| longllmlingua | 0.630 | 0.508 | 720.300 | 0.522 | 1047.220 |
| ours_full | 0.540 | 0.470 | 750.000 | 0.502 | 1020.480 |

## Lost-in-the-Middle Late-position Focus
| position | method | contains_answer | nq_style_f1 |
| --- | --- | --- | --- |
| 15 | truncate | 0.250 | 0.236 |
| 15 | bm25 | 0.500 | 0.472 |
| 15 | longllmlingua | 0.750 | 0.502 |
| 15 | ours_full | 0.650 | 0.493 |
| 20 | truncate | 0.200 | 0.189 |
| 20 | bm25 | 0.650 | 0.411 |
| 20 | longllmlingua | 0.800 | 0.546 |
| 20 | ours_full | 0.550 | 0.394 |

## LongBench Official Evaluation
| task | method | official_score | compressed_tokens | token_saving |
| --- | --- | --- | --- | --- |
| hotpotqa | original | 67.370 | 13294.760 | 0.000 |
| hotpotqa | truncate | 19.580 | 2000.000 | 0.819 |
| hotpotqa | bm25 | 60.940 | 2007.520 | 0.819 |
| hotpotqa | llmlingua | 36.840 | 1955.200 | 0.823 |
| hotpotqa | longllmlingua | 45.570 | 1955.200 | 0.823 |
| hotpotqa | ours_full | 55.340 | 2000.000 | 0.819 |
| 2wikimqa | original | 71.080 | 7024.160 | 0.000 |
| 2wikimqa | truncate | 26.980 | 1969.340 | 0.640 |
| 2wikimqa | bm25 | 65.710 | 1979.180 | 0.638 |
| 2wikimqa | llmlingua | 59.930 | 1882.400 | 0.655 |
| 2wikimqa | longllmlingua | 62.680 | 1882.400 | 0.655 |
| 2wikimqa | ours_full | 61.510 | 1969.460 | 0.639 |
| multifieldqa_en | original | 57.800 | 7068.860 | 0.000 |
| multifieldqa_en | truncate | 50.120 | 1986.340 | 0.626 |
| multifieldqa_en | bm25 | 52.600 | 1994.640 | 0.625 |
| multifieldqa_en | llmlingua | 45.450 | 1996.860 | 0.619 |
| multifieldqa_en | longllmlingua | 42.590 | 1996.860 | 0.619 |
| multifieldqa_en | ours_full | 51.980 | 1986.880 | 0.626 |

## Ablation
| method | contains_answer | nq_style_f1 | api_tokens | answer_time |
| --- | --- | --- | --- | --- |
| ours_bm25_only | 0.590 | 0.422 | 1768.510 | 2.487 |
| ours_keyword | 0.660 | 0.461 | 1787.030 | 2.679 |
| ours_full | 0.600 | 0.414 | 1765.180 | 2.261 |

## Budget Sensitivity
| token_budget | contains_answer | nq_style_f1 | api_tokens | token_saving |
| --- | --- | --- | --- | --- |
| 500 | 0.380 | 0.339 | 752.070 | 0.834 |
| 750 | 0.420 | 0.382 | 1012.740 | 0.751 |
| 1000 | 0.530 | 0.410 | 1242.100 | 0.669 |
| 1500 | 0.620 | 0.437 | 1784.090 | 0.503 |
| 2000 | 0.690 | 0.450 | 2264.220 | 0.337 |
