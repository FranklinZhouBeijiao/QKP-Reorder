from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qkp_reorder.run_innovation_chinese_mixed_eval as eval_module
from qkp_reorder.run_innovation_chinese_mixed_eval import (
    CHINESE_MIXED_DATASET,
    METHODS,
    calculate_keyword_coverage,
    contains_any_answer,
    generate_nonempty_answer_with_retries,
    run_api_eval,
)
from qkp_reorder.compressors import ContextCompressor


class _FakePromptCompressor:
    def compress_prompt(self, context: str, **_: object) -> dict:
        return {"compressed_prompt": "前置干扰片段只讨论上海团队和广州论坛。"}


class _FakeLongLLMLinguaCompressor(ContextCompressor):
    def _get_prompt_compressor(self):
        return _FakePromptCompressor()


def test_dataset_shape() -> None:
    assert len(CHINESE_MIXED_DATASET) == 30
    assert sum(1 for row in CHINESE_MIXED_DATASET if row["language_type"] == "zh") == 15
    assert sum(1 for row in CHINESE_MIXED_DATASET if row["language_type"] == "mixed") == 15
    assert len({row["sample_id"] for row in CHINESE_MIXED_DATASET}) == 30
    assert all(row["answers"] for row in CHINESE_MIXED_DATASET)


def test_methods_cover_reproduction_and_qkp_sets() -> None:
    assert METHODS == [
        "original",
        "truncate",
        "bm25",
        "llmlingua",
        "longllmlingua",
        "ours_bm25_only",
        "ours_keyword",
        "ours_full",
    ]
    assert len(METHODS) == 8


def test_longllmlingua_method_is_callable() -> None:
    compressor = _FakeLongLLMLinguaCompressor()
    result = compressor.compress_context(
        context="上海团队介绍了无关方案。2024年北京人工智能大会上，深度求索发布了DeepSeek-R1相关工具。",
        question="2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？",
        method="longllmlingua",
        token_budget=40,
    )
    assert result["method"] == "longllmlingua"
    assert isinstance(result["compressed_context"], str)


def test_longllmlingua_keeps_chinese_mixed_key_evidence() -> None:
    compressor = _FakeLongLLMLinguaCompressor()
    context = (
        "上海团队介绍了多模态检索系统，主要面向医学影像。"
        "广州论坛讨论了低空经济数据平台，未涉及大模型工具。"
        "2024年北京人工智能大会上，深度求索发布了DeepSeek-R1相关工具，重点展示中文推理能力。"
        "闭幕环节讨论了开源生态和模型安全。"
    )
    result = compressor.compress_context(
        context=context,
        question="2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？",
        method="longllmlingua",
        token_budget=60,
    )

    compressed = result["compressed_context"]
    assert "深度求索" in compressed
    assert "DeepSeek-R1" in compressed


def test_longllmlingua_handles_empty_context_and_question() -> None:
    compressor = _FakeLongLLMLinguaCompressor()
    result = compressor.compress_context(
        context="",
        question="",
        method="longllmlingua",
        token_budget=60,
    )
    assert result["compressed_context"] == ""
    assert result["compressed_tokens"] == 0


def test_contains_any_answer_handles_chinese_aliases() -> None:
    assert contains_any_answer("最终答案是星河数据库。", ["星河数据库", "GalaxyDB"])
    assert contains_any_answer("The answer is GalaxyDB.", ["星河数据库", "GalaxyDB"])
    assert not contains_any_answer("最终答案是另一个系统。", ["星河数据库", "GalaxyDB"])


def test_keyword_coverage_counts_chinese_and_mixed_terms() -> None:
    keywords = ["北京", "DeepSeek-R1", "2024"]
    text = "2024年北京人工智能大会发布了DeepSeek-R1工具。"
    assert calculate_keyword_coverage(text, keywords) == 1.0
    assert calculate_keyword_coverage("北京大会", keywords) == 1 / 3


def test_generate_nonempty_answer_retries_empty_response() -> None:
    calls = []

    def fake_generate(prompt: str, max_tokens: int) -> dict:
        calls.append((prompt, max_tokens))
        if len(calls) == 1:
            return {"answer": "", "answer_time": 0.1, "usage": {"total_tokens": 3}}
        return {"answer": "青岚实验室", "answer_time": 0.2, "usage": {"total_tokens": 4}}

    result = generate_nonempty_answer_with_retries(
        prompt="question",
        generator=fake_generate,
        sleep_fn=lambda _: None,
    )

    assert result["answer"] == "青岚实验室"
    assert result["retries"] == 1
    assert len(calls) == 2


def test_api_skip_reuses_existing_nonempty_results() -> None:
    old_env = os.environ.get("QKP_CHINESE_SKIP_API")
    old_api_result_out = eval_module.API_RESULT_OUT

    with tempfile.TemporaryDirectory() as tmpdir:
        existing_api_path = Path(tmpdir) / "existing_api_results.csv"
        existing = pd.DataFrame(
            [
                {
                    "sample_id": "cm_001",
                    "language_type": "zh",
                    "method": "ours_full",
                    "method_label": "QKP-Reorder",
                    "question": "问题",
                    "answers": '["答案"]',
                    "prediction": "答案",
                    "api_contains_answer": 1,
                    "empty_prediction": 0,
                    "error_message": "",
                    "answer_time": 0.1,
                    "api_total_tokens": 10,
                    "api_prompt_tokens": 8,
                    "api_completion_tokens": 2,
                    "retries": 0,
                }
            ]
        )
        existing.to_csv(existing_api_path, index=False, encoding="utf-8-sig")

        try:
            os.environ["QKP_CHINESE_SKIP_API"] = "1"
            eval_module.API_RESULT_OUT = existing_api_path
            reused = run_api_eval(pd.DataFrame())
        finally:
            if old_env is None:
                os.environ.pop("QKP_CHINESE_SKIP_API", None)
            else:
                os.environ["QKP_CHINESE_SKIP_API"] = old_env
            eval_module.API_RESULT_OUT = old_api_result_out

    assert len(reused) == 1
    assert reused.iloc[0]["method"] == "ours_full"
    assert reused.iloc[0]["api_contains_answer"] == 1


def main() -> None:
    test_dataset_shape()
    test_methods_cover_reproduction_and_qkp_sets()
    test_longllmlingua_method_is_callable()
    test_longllmlingua_keeps_chinese_mixed_key_evidence()
    test_longllmlingua_handles_empty_context_and_question()
    test_contains_any_answer_handles_chinese_aliases()
    test_keyword_coverage_counts_chinese_and_mixed_terms()
    test_generate_nonempty_answer_retries_empty_response()
    test_api_skip_reuses_existing_nonempty_results()
    print("Chinese mixed selfcheck passed.")


if __name__ == "__main__":
    main()

