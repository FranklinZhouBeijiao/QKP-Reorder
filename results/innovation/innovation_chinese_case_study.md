# Chinese Segmentation Case Study

- jieba_available: `True`
- question: 2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？
- extracted_keywords: `2024, 年, 北京, 人工智能, 大会, 上, 家, 公司, 发布, deepseek-r1, 相关, 工具`

## Segment-level Analysis

| segment | tokens | numbers | entities | protection_bonus |
| --- | --- | --- | --- | --- |
| 上海团队介绍了多模态检索系统。 | 上海, 团队, 介绍, 多, 模态, 检索系统 |  |  | 0.6 |
| 2024年北京人工智能大会上， | 2024, 年, 北京, 人工智能, 大会, 上 | 2024 |  | 3.25 |
| DeepSeek发布了DeepSeek-R1相关工具， | deepseek, 发布, deepseek-r1, 相关, 工具 |  |  | 2.4 |
| 重点展示中文推理能力。 | 重点, 展示, 中文, 推理, 能力 |  |  | 0.0 |
| 随后NASA报告了AI_report-2024中的英文实验， | 随后, nasa, 报告, ai_report-2024, 中, 英文, 实验 | 2024 |  | 0.85 |
| 样本数为1,234， | 样本数, 为, 1,234 | 1,234 |  | 0.25 |
| 增长12.5%。 | 增长, 12.5% | 12.5% |  | 0.25 |

## Report Message

This case shows that the Ours method includes reusable Chinese sentence segmentation, mixed Chinese/English tokenization, number extraction, entity-like term extraction, and keyword/entity/number protection. It should be presented as an extension capability rather than as the main reason for English benchmark gains.
