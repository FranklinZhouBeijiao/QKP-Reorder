# Ours Method Design Notes

## 1. 方法目标

Ours 是在现有 LongLLMLingua 复现框架上新增的轻量级、可解释、无需训练的上下文压缩方法。目标是在相同 `token_budget` 下，比简单 `truncate` 更少受答案位置影响，比现有 `bm25` 更重视关键词、实体和数字证据，并通过 middle-aware reordering 缓解长上下文中间证据不易被回答模型利用的问题。

本方法只使用 `context` 和 `question`，不使用 gold answer、答案位置标签或评价结果。

## 2. 固定方法名

后续实现和实验统一使用以下三个方法名：

```text
ours_bm25_only
ours_keyword
ours_full
```

其中 `ours_full` 是主方法，`ours_bm25_only` 和 `ours_keyword` 是消融版本。

## 3. 三个版本差异

| 方法 | 分句 | BM25 | 关键词/实体/数字保护 | Middle-aware reordering | 用途 |
|---|---|---|---|---|---|
| `ours_bm25_only` | 中英文兼容 | 是 | 否 | 否 | 隔离分句与 BM25 选择效果 |
| `ours_keyword` | 中英文兼容 | 是 | 是 | 否 | 观察保护策略贡献 |
| `ours_full` | 中英文兼容 | 是 | 是 | 是 | 最终主方法 |

`ours_bm25_only` 和 `ours_keyword` 建议保持选中片段的原始相对顺序，避免把重排序收益混入消融结果。`ours_full` 在选择片段后再做位置重排。

## 4. 中英文分句策略

分句组件应将 `context` 切分为适合 BM25 打分和预算选择的短片段：

1. 先按空行、换行切分较大段落。
2. 再按英文 `. ? !` 和中文 `。？！，；` 等标点切分。
3. 包含 CJK 字符时启用中文标点规则和 `jieba` 分词。
4. 英文文本沿用正则 tokenization，保留英文单词、数字和下划线 token。
5. 过长片段应按 token 长度继续软切分，避免单个片段超过预算后无法入选。

## 5. Question-aware BM25 打分策略

BM25 查询来自 `question` 的 tokens，候选文档来自分句后的 context 片段。

基本流程：

1. 对 `question` 做中英文混合 tokenization。
2. 对每个片段做同样的 tokenization。
3. 用 `BM25Okapi` 得到每个片段的问题相关性分数。
4. 按分数从高到低选择片段，同时用 `count_tokens()` 累计预算。
5. 若问题 token 为空、片段为空或没有片段入选，回退到 `truncate_by_tokens(context, token_budget)`。

## 6. Keyword/entity/number protection

保护策略只使用问题和候选片段本身，不使用答案信息。

建议保护项：

- Question keywords：英文关键词过滤常见停用词；中文关键词优先用 `jieba.lcut()`。
- Numbers：年份、整数、小数、百分比、带单位数字。
- Entity-like strings：英文大写开头词组、全大写缩写、含连字符或下划线的专名样式 token。

推荐采用加权保护，而不是无条件硬保留：

```text
final_score = bm25_score + keyword_bonus + entity_bonus + number_bonus
```

这样可以增强关键证据的入选概率，同时降低无关数字或实体片段挤占预算的风险。

## 7. Middle-aware reordering

`ours_full` 在片段选择后执行 middle-aware reordering。目标是把最高价值证据放在上下文开头或首尾位置，减少关键证据落在压缩上下文中部的风险。

建议 deterministic 策略：

1. 按最终得分对选中片段排序。
2. 最高分片段放在开头。
3. 次高分片段放在末尾。
4. 其余高分片段在开头和末尾之间交替放置。
5. 普通片段尽量保持原始相对顺序，放在中间。
6. 拼接后再次调用 `truncate_by_tokens()` 保证不超过预算。

## 8. Token budget 公平比较原则

所有 Ours 版本必须和现有 baseline 使用相同 `token_budget`。

公平性要求：

- 不给 Ours 额外 token allowance。
- 片段选择阶段用 `count_tokens()` 统计预算。
- 最终输出统一用 `truncate_by_tokens()` 截断到预算内。
- token budget 敏感性实验中，对所有方法使用同一组预算。

## 9. 与现有方法的区别

与 `truncate` 的区别：

- `truncate` 只保留前缀文本，容易丢失中后部证据。
- Ours 根据问题选择证据片段，并在 `ours_full` 中调整高价值证据位置。

与 `bm25` 的区别：

- 现有 `bm25` 主要是英文简单分句 + BM25 选择 + 原始顺序拼接。
- Ours 增加中英文兼容分句、关键词/实体/数字保护，并在 `ours_full` 中加入 middle-aware reordering。

与 `longllmlingua` 的区别：

- `longllmlingua` 调用 LLMLingua 压缩模型。
- Ours 不调用压缩模型，不训练新模型，不调用 API，是规则化、可解释、低成本方法。
- Ours 只做当前框架下的相对比较，不声称替代或全面优于 LongLLMLingua。

## 10. 阶段2/3实现接口清单

阶段2建议实现文本处理组件：

- 中英文兼容分句函数。
- 中英文混合 BM25 tokenization 函数。
- question keyword 抽取函数。
- number 抽取函数。
- entity-like string 抽取函数。
- 片段保护加权函数。

阶段3建议接入统一压缩接口：

```python
def ours_bm25_only_context(context: str, question: str, token_budget: int) -> str:
    ...

def ours_keyword_context(context: str, question: str, token_budget: int) -> str:
    ...

def ours_full_context(context: str, question: str, token_budget: int) -> str:
    ...
```

并在 `src/compressors.py` 中：

- 导入三个函数。
- 将 `ours_bm25_only`、`ours_keyword`、`ours_full` 加入 `SUPPORTED_METHODS`。
- 在 `ContextCompressor.compress_context()` 中加入对应分支。
- 保持返回字段与现有方法一致。

## 11. 风险和边界

- 不使用 gold answer、答案位置标签或评价结果。
- 不覆盖 `results/stage6`、`results/stage6_position`、`results/stage7`、`results/final`。
- 不调用 DeepSeek API。
- 不运行完整实验。
- 不修改评价逻辑来追求单项指标。
- 不声称 Ours 一定全面最优。
- LongBench 结果后续应优先使用官方 `eval.py` 分数。
- NQ 和 Lost-in-the-Middle 后续补充指标应称为 NQ-style normalized short-answer EM/F1。
