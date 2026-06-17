# QKP-Reorder

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-research%20preview-orange.svg)](#reproduction-notes)

QKP-Reorder is a lightweight long-context prompt compression project built around a simple idea: keep question-relevant evidence, protect fragile keywords and entities, and move high-value context away from the middle of a long prompt.

It includes a cleaned reproduction pipeline inspired by LLMLingua / LongLLMLingua, a local Streamlit demo, compact result summaries, and an interpretable compression variant for English, Chinese, and mixed-language QA settings.

```text
QKP-Reorder = mixed Chinese/English segmentation
            + question-aware BM25 scoring
            + keyword/entity/number protection
            + middle-aware reordering
            + token-budget truncation
```

## Highlights

- Compares `Original`, `Truncate`, `BM25`, `LLMLingua`, `LongLLMLingua`, and three QKP variants.
- Runs the QKP methods locally without an API call.
- Provides an optional DeepSeek-compatible answer generation path through the OpenAI SDK.
- Keeps the release small by excluding raw benchmark data, local logs, virtual environments, browser caches, and API secrets.
- Ships with summary CSVs and report-ready figures from the cleaned experiment run.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_qkp_reorder.py
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/run_qkp_reorder.py
```

## Run The Demo

```bash
pip install -r requirements.txt
python -m streamlit run qkp_reorder/demo_streamlit_app.py
```

The demo supports local compression. To enable answer generation with a DeepSeek-compatible OpenAI SDK endpoint, copy `.env.example` to `.env` and add your own credentials:

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Never commit `.env`.

## Use As A Library

```python
from qkp_reorder.compressors import ContextCompressor

compressor = ContextCompressor()
result = compressor.compress_context(
    context="Long context goes here...",
    question="What evidence matters?",
    method="ours_full",
    token_budget=512,
)

print(result["compressed_context"])
```

Available QKP methods:

| Method | Description |
| --- | --- |
| `ours_bm25_only` | Mixed-language segmentation plus question-aware BM25. |
| `ours_keyword` | Adds keyword, entity, and number protection. |
| `ours_full` | Adds middle-aware reordering and is shown as QKP-Reorder. |

## Repository Layout

```text
qkp-reorder/
|-- qkp_reorder/              # Core compressor, baselines, evaluation, demo, scripts
|-- examples/                 # Minimal no-API usage example
|-- results/                  # Compact summaries and report notes
|-- figures/                  # Selected final and innovation figures
|-- data/README.md            # Dataset acquisition and redistribution notes
|-- third_party/              # Third-party license notices
|-- requirements.txt          # Lightweight demo/runtime dependencies
|-- requirements-reproduction.txt
|-- pyproject.toml
`-- README.md
```

## Reproduction Notes

For the fuller experiment environment:

```bash
pip install -r requirements-reproduction.txt
```

The original experiments used public long-context QA datasets and a unified compression-answer-evaluation pipeline. This cleaned release keeps the code, summary artifacts, and figures, but does not redistribute large raw datasets or full per-sample prediction dumps.

See `data/README.md` before downloading or publishing derived data. Check the licenses and redistribution terms for NaturalQuestions, Lost in the Middle, LongBench, LLMLingua, and LongLLMLingua resources in your own setup.

Useful entry points:

- `qkp_reorder/run_innovation_stage3_ours_compressor.py`
- `qkp_reorder/run_innovation_stage4_nq_100.py`
- `qkp_reorder/run_innovation_stage5_position.py`
- `qkp_reorder/run_innovation_stage6_longbench_small.py`
- `qkp_reorder/run_innovation_stage7_ablation.py`
- `qkp_reorder/run_innovation_stage7_budget_sensitivity.py`

## What Is Not Included

- Real API keys or `.env` files
- Raw benchmark datasets
- Local run logs
- Browser automation profiles and caches
- Virtual environments
- IDE metadata
- Python bytecode caches
- Large per-sample prediction dumps

These files are excluded intentionally to keep the release safe, small, and redistributable.

## Security

If an API key was ever committed to a Git history, revoke or rotate it immediately. Deleting the file in a later commit is not enough. See `SECURITY.md` for the short publishing checklist used by this release.

## License

This repository is released under the MIT License. Third-party notices are listed in `NOTICE.md`.

## Citation

If you use this project in a report or paper, cite the original LLMLingua, LongLLMLingua, LongBench, Lost in the Middle, and dataset sources that apply to your experiment setup.

---

# 中文说明

QKP-Reorder 是一个面向长上下文问答的轻量级 prompt 压缩项目。它在 LLMLingua / LongLLMLingua 风格复现实验的基础上，加入了一个可解释的改进方法：根据问题选择证据、保护关键词/实体/数字信息，并把高价值片段移动到压缩上下文的前部和后部，以缓解 `lost-in-the-middle` 现象。

```text
QKP-Reorder = 中英文混合分句
            + 问题感知 BM25 打分
            + 关键词/实体/数字保护
            + 中间位置感知重排
            + token budget 截断
```

## 项目特点

- 对比 `Original`、`Truncate`、`BM25`、`LLMLingua`、`LongLLMLingua` 和三个 QKP 变体。
- QKP 方法可以本地运行，不需要 API。
- 可选接入 DeepSeek 兼容的 OpenAI SDK 接口进行答案生成。
- 已清理真实密钥、原始数据、大日志、虚拟环境、浏览器缓存和本地配置。
- 保留了摘要结果、报告说明和可直接用于展示的图表。

## 快速开始

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_qkp_reorder.py
```

macOS 或 Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/run_qkp_reorder.py
```

## 启动演示页面

```bash
pip install -r requirements.txt
python -m streamlit run qkp_reorder/demo_streamlit_app.py
```

演示页面可以本地运行压缩逻辑。如果需要调用 DeepSeek 生成答案，请复制 `.env.example` 为 `.env`，并填入自己的 API 配置：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

不要提交 `.env` 文件。

## 作为库使用

```python
from qkp_reorder.compressors import ContextCompressor

compressor = ContextCompressor()
result = compressor.compress_context(
    context="这里放入长上下文...",
    question="问题是什么？",
    method="ours_full",
    token_budget=512,
)

print(result["compressed_context"])
```

QKP 方法说明：

| 方法 | 说明 |
| --- | --- |
| `ours_bm25_only` | 中英文混合分句 + 问题感知 BM25。 |
| `ours_keyword` | 在 BM25 基础上加入关键词、实体和数字保护。 |
| `ours_full` | 在 `ours_keyword` 基础上加入中间位置感知重排，即 QKP-Reorder。 |

## 仓库结构

```text
qkp-reorder/
|-- qkp_reorder/              # 核心压缩器、baseline、评价、demo 和实验脚本
|-- examples/                 # 不依赖 API 的最小示例
|-- results/                  # 精简后的摘要结果和报告说明
|-- figures/                  # 最终实验和改进实验图表
|-- data/README.md            # 数据下载与再分发说明
|-- third_party/              # 第三方许可证说明
|-- requirements.txt          # 轻量运行依赖
|-- requirements-reproduction.txt
|-- pyproject.toml
`-- README.md
```

## 复现说明

完整实验环境可以从下面的依赖开始：

```bash
pip install -r requirements-reproduction.txt
```

本开源版保留代码、摘要结果和图表，但不直接分发大型原始数据集，也不包含完整逐样本预测输出。下载或发布衍生数据前，请先阅读 `data/README.md`，并分别检查 NaturalQuestions、Lost in the Middle、LongBench、LLMLingua 和 LongLLMLingua 相关资源的许可证与再分发要求。

## 未包含内容

- 真实 API key 或 `.env` 文件
- 原始 benchmark 数据集
- 本地运行日志
- 浏览器 profile 与缓存
- 虚拟环境
- IDE 配置
- Python 字节码缓存
- 大型逐样本预测结果

这些内容被有意排除，以保证仓库更安全、更轻量，也更适合公开发布。

## 许可证与引用

本仓库使用 MIT License。第三方说明见 `NOTICE.md`。

如果你在报告或论文中使用本项目，请根据实际实验设置引用原始 LLMLingua、LongLLMLingua、LongBench、Lost in the Middle 以及相关数据集来源。
