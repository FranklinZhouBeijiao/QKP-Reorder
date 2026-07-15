# QKP-Reorder

English | [简体中文](README.zh-CN.md)

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
