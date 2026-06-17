# QKP-Reorder Chinese/Mixed Case Study

- jieba_available: `True`
- sample_id: `cm_001`
- question: 2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？
- gold_answers: `深度求索, DeepSeek`
- extracted_keywords: `2024, 年, 北京, 人工智能, 大会, 上, 家, 公司, 发布, deepseek-r1, 相关, 工具`

## Compression Summary

| method         | method_label        | language_type   |   rows |   contains_answer_rate |   keyword_coverage |   avg_compressed_tokens |   avg_token_saving |   avg_compression_time |
|:---------------|:--------------------|:----------------|-------:|-----------------------:|-------------------:|------------------------:|-------------------:|-----------------------:|
| original       | Original            | overall         |     30 |               1        |           0.872222 |                119.7    |           0        |            1.07288e-06 |
| ours_keyword   | QKP-Reorder-Keyword | overall         |     30 |               1        |           0.872222 |                 56.7    |           0.46518  |            0.00189277  |
| ours_bm25_only | QKP-Reorder-BM25    | overall         |     30 |               1        |           0.872222 |                 55.8    |           0.473393 |            0.00138391  |
| longllmlingua  | LongLLMLingua       | overall         |     30 |               1        |           0.872222 |                 60      |           0.435712 |            1.00826     |
| ours_full      | QKP-Reorder         | overall         |     30 |               1        |           0.872222 |                 56.7    |           0.46518  |            0.00195461  |
| llmlingua      | LLMLingua           | overall         |     30 |               0.7      |           0.622222 |                 87.2667 |           0.214051 |            2.24879     |
| bm25           | BM25                | overall         |     30 |               0.566667 |           0.563889 |                 57.7667 |           0.461032 |            0.000283583 |
| truncate       | Truncate            | overall         |     30 |               0.466667 |           0.463889 |                 60      |           0.435712 |            9.38892e-05 |

## Segment-level Analysis

| segment | tokens | protection_bonus |
| --- | --- | ---: |
| 上海团队介绍了多模态检索系统， | 上海, 团队, 介绍, 多, 模态, 检索系统 | 0.60 |
| 主要面向医学影像。 | 主要, 面向, 医学影像 | 0.00 |
| 广州论坛讨论了低空经济数据平台， | 广州, 论坛, 讨论, 低空, 经济, 数据, 平台 | 0.00 |
| 未涉及大模型工具。 | 未, 涉及, 大, 模型, 工具 | 0.60 |
| 南京高校展示了校园智能问答助手， | 南京, 高校, 展示, 校园, 智能, 问答, 助手 | 0.00 |
| 发布时间是2023年。 | 发布, 时间, 2023, 年 | 1.45 |
| 成都实验室报告了中文语音识别基准， | 成都, 实验室, 报告, 中文, 语音, 识别, 基准 | 0.00 |
| 样本数为1,234。 | 样本数, 为, 1,234 | 0.25 |
| 2024年北京人工智能大会上， | 2024, 年, 北京, 人工智能, 大会, 上 | 3.25 |
| 深度求索发布了DeepSeek-R1相关工具， | 深度, 求索, 发布, deepseek-r1, 相关, 工具 | 2.40 |
| 重点展示中文推理能力。 | 重点, 展示, 中文, 推理, 能力 | 0.00 |
| 随后NASA报告了AI_report-2024中的英文实验， | 随后, nasa, 报告, ai_report-2024, 中, 英文, 实验 | 0.85 |
| 增长12.5%。 | 增长, 12.5% | 0.25 |
| 闭幕环节讨论了开源生态和模型安全。 | 闭幕, 环节, 讨论, 开源, 生态, 模型, 安全 | 0.00 |

## Compressed Contexts

### Original

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
上海团队介绍了多模态检索系统，主要面向医学影像。
广州论坛讨论了低空经济数据平台，未涉及大模型工具。
南京高校展示了校园智能问答助手，发布时间是2023年。
成都实验室报告了中文语音识别基准，样本数为1,234。
2024年北京人工智能大会上，深度求索发布了DeepSeek-R1相关工具，重点展示中文推理能力。
随后NASA报告了AI_report-2024中的英文实验，增长12.5%。
闭幕环节讨论了开源生态和模型安全。
```

### Truncate

- contains_answer: `0`
- keyword_coverage: `0.000`

```text
上海团队介绍了多模态检索系统，主要面向医学影像。
广州论坛讨论了低空经济数据平台，未涉及大模型工具。
南京高校
```

### BM25

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
2024年北京人工智能大会上，深度求索发布了DeepSeek-R1相关工具，重点展示中文推理能力。
闭幕环节讨论了开源生态和模型安全。
```

### LLMLingua

- contains_answer: `1`
- keyword_coverage: `0.500`

```text
检 索 医 学 影 像 广 州 低 空 经 济 南 京 答 2023 成 中 文 语 音 识 别 1 234 2024 DeepSeek - R1 中 文 推 理 NASA AI report 2024 英 文 实 12. 5 % 开 源 生 态 模 型 安 全
```

### LongLLMLingua

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
未涉及大模型工具。
发布时间是2023年。
样本数为1,234。
2024年北京人工智能大会上，
深度求索发布了DeepSeek-R1相关工具，
增长12.5%。

```

### QKP-Reorder-BM25

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
主要面向医学影像。
未涉及大模型工具。
发布时间是2023年。
2024年北京人工智能大会上，
深度求索发布了DeepSeek-R1相关工具，
```

### QKP-Reorder-Keyword

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
未涉及大模型工具。
发布时间是2023年。
样本数为1,234。
2024年北京人工智能大会上，
深度求索发布了DeepSeek-R1相关工具，
增长12.5%。
```

### QKP-Reorder

- contains_answer: `1`
- keyword_coverage: `1.000`

```text
2024年北京人工智能大会上，
发布时间是2023年。
未涉及大模型工具。
样本数为1,234。
增长12.5%。
深度求索发布了DeepSeek-R1相关工具，
```
