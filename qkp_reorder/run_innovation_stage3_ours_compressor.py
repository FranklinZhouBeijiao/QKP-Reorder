from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import SUPPORTED_METHODS, ContextCompressor

LOG_PATH = PROJECT_ROOT / "logs" / "innovation" / "stage3_ours_compressor_test_log.txt"
DEBUG_CSV_PATH = (
    PROJECT_ROOT / "results" / "innovation" / "ours_compressor_debug.csv"
)

_ENGLISH_CONTEXT = """
Albert Einstein published the theory of special relativity in 1905.
This work changed how physicists understand space and time.
The Eiffel Tower is a famous landmark in Paris, France.
NASA launched the James Webb Space Telescope in 2021.
Many people enjoy eating pizza on Friday evenings.
Machine learning models sometimes struggle with long-context reasoning.
"""

_CHINESE_CONTEXT = """
北京是中国的首都，拥有故宫和长城等著名景点。
人工智能在自然语言处理领域取得了显著进展。
东京是日本的首都，以樱花和美食闻名。
深度学习模型需要大量计算资源进行训练。
上海是中国最大的城市，也是重要的经济中心。
"""

_MIXED_CONTEXT = """
NASA在2024年发布了关于long-context AI的最新研究报告。
The study shows that transformer models achieve 95.3% accuracy on QA tasks.
北京大学的团队在自然语言处理方面做出了重要贡献。
Regular exercise is recommended for maintaining good health.
深度学习模型DeepSeek-V4在multiple benchmarks上表现优异，得分提升12.5%。
"""

_ENGLISH_QUESTION = "What theory did Einstein publish and when?"
_CHINESE_QUESTION = "哪个城市是中国的首都？"
_MIXED_QUESTION = "NASA在2024年发布了什么报告，DeepSeek-V4表现如何？"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_result_fields(result: dict, method: str) -> None:
    _require(result["method"] == method, f"method field mismatch: {result['method']}")
    _require("compressed_context" in result, "missing compressed_context")
    _require(isinstance(result["original_tokens"], int), "original_tokens not int")
    _require(isinstance(result["compressed_tokens"], int), "compressed_tokens not int")
    _require(result["compression_ratio"] >= 0, "compression_ratio negative")
    _require(result["token_saving_ratio"] >= 0, "token_saving_ratio negative")
    _require(result["compression_time"] >= 0, "compression_time negative")
    _require("token_budget" in result, "missing token_budget")
    _require("fallback_used" in result, "missing fallback_used")
    _require(result["fallback_used"] is False, f"unexpected fallback for {method}")
    _require(result["compressor_model_name"] is None, f"compressor_model_name should be None for {method}")


def _check_token_budget(result: dict, budget: int) -> None:
    ct = result["compressed_tokens"]
    if ct > budget:
        budget_str = f"compressed_tokens ({ct}) exceeds token_budget ({budget})"
        raise AssertionError(budget_str)


def test_registered_in_supported_methods(log_lines: list[str]) -> None:
    for name in ["ours_bm25_only", "ours_keyword", "ours_full"]:
        _require(name in SUPPORTED_METHODS, f"{name} not in SUPPORTED_METHODS")
    log_lines.append("SUPPORTED_METHODS check passed.")


def test_ours_bm25_only_basic(compressor: ContextCompressor, log_lines: list[str]) -> dict:
    result = compressor.compress_context(
        context=_ENGLISH_CONTEXT,
        question=_ENGLISH_QUESTION,
        method="ours_bm25_only",
        token_budget=80,
    )
    _check_result_fields(result, "ours_bm25_only")
    _check_token_budget(result, 80)
    _require(
        "relativity" in result["compressed_context"].lower()
        or "1905" in result["compressed_context"],
        "ours_bm25_only should select evidence about Einstein's theory from English context",
    )
    log_lines.append(
        f"ours_bm25_only basic: OK "
        f"(original_tokens={result['original_tokens']}, "
        f"compressed_tokens={result['compressed_tokens']})"
    )
    return result


def test_ours_keyword_basic(compressor: ContextCompressor, log_lines: list[str]) -> dict:
    result = compressor.compress_context(
        context=_ENGLISH_CONTEXT,
        question=_ENGLISH_QUESTION,
        method="ours_keyword",
        token_budget=80,
    )
    _check_result_fields(result, "ours_keyword")
    _check_token_budget(result, 80)
    _require(
        "relativity" in result["compressed_context"].lower()
        or "1905" in result["compressed_context"],
        "ours_keyword should select evidence about Einstein's theory from English context",
    )
    log_lines.append(
        f"ours_keyword basic: OK "
        f"(original_tokens={result['original_tokens']}, "
        f"compressed_tokens={result['compressed_tokens']})"
    )
    return result


def test_ours_full_basic(compressor: ContextCompressor, log_lines: list[str]) -> dict:
    result = compressor.compress_context(
        context=_ENGLISH_CONTEXT,
        question=_ENGLISH_QUESTION,
        method="ours_full",
        token_budget=80,
    )
    _check_result_fields(result, "ours_full")
    _check_token_budget(result, 80)
    _require(
        "relativity" in result["compressed_context"].lower()
        or "1905" in result["compressed_context"],
        "ours_full should select evidence about Einstein's theory from English context",
    )
    log_lines.append(
        f"ours_full basic: OK "
        f"(original_tokens={result['original_tokens']}, "
        f"compressed_tokens={result['compressed_tokens']})"
    )
    return result


def test_empty_context(compressor: ContextCompressor, log_lines: list[str]) -> None:
    for method in ["ours_bm25_only", "ours_keyword", "ours_full"]:
        result = compressor.compress_context(
            context="",
            question="What is AI?",
            method=method,
            token_budget=50,
        )
        _require(result["compressed_context"] == "", f"{method} empty context should return empty")
        _require(result["original_tokens"] == 0, f"{method} empty context original_tokens should be 0")
        _require(result["compressed_tokens"] == 0, f"{method} empty context compressed_tokens should be 0")
    log_lines.append("empty context: OK (all three methods return empty)")


def test_empty_question(compressor: ContextCompressor, log_lines: list[str]) -> None:
    context = "Some sample text for testing purposes."
    for method in ["ours_bm25_only", "ours_keyword", "ours_full"]:
        result = compressor.compress_context(
            context=context,
            question="",
            method=method,
            token_budget=20,
        )
        _require(result["compressed_tokens"] <= 20, f"{method} empty question exceeded budget")
        _require(
            len(result["compressed_context"]) > 0,
            f"{method} empty question should produce non-empty output",
        )
    log_lines.append("empty question: OK (all three methods handle gracefully)")


def test_chinese_segmentation(compressor: ContextCompressor, log_lines: list[str]) -> None:
    result = compressor.compress_context(
        context=_CHINESE_CONTEXT,
        question=_CHINESE_QUESTION,
        method="ours_bm25_only",
        token_budget=150,
    )
    _check_token_budget(result, 150)
    _require(
        "北京" in result["compressed_context"] and "首都" in result["compressed_context"],
        "Chinese segmentation should retain Beijing-is-capital evidence",
    )
    log_lines.append(
        f"Chinese segmentation: OK "
        f"(original_tokens={result['original_tokens']}, "
        f"compressed_tokens={result['compressed_tokens']})"
    )


def test_mixed_language(compressor: ContextCompressor, log_lines: list[str]) -> None:
    result = compressor.compress_context(
        context=_MIXED_CONTEXT,
        question=_MIXED_QUESTION,
        method="ours_full",
        token_budget=200,
    )
    _check_token_budget(result, 200)
    compressed = result["compressed_context"]
    _require("NASA" in compressed, "mixed should keep English entity NASA")
    _require("2024" in compressed, "mixed should keep number 2024")
    _require("DeepSeek" in compressed or "deepseek" in compressed.lower(),
             "mixed should keep DeepSeek entity")
    _require(any(ord(c) > 127 for c in compressed),
             "mixed should retain some Chinese content")
    log_lines.append(
        f"Mixed language: OK "
        f"(original_tokens={result['original_tokens']}, "
        f"compressed_tokens={result['compressed_tokens']})"
    )


def test_varied_token_budgets(compressor: ContextCompressor, log_lines: list[str]) -> None:
    for budget in [30, 60, 120]:
        for method in ["ours_bm25_only", "ours_keyword", "ours_full"]:
            result = compressor.compress_context(
                context=_ENGLISH_CONTEXT,
                question=_ENGLISH_QUESTION,
                method=method,
                token_budget=budget,
            )
            ct = result["compressed_tokens"]
            _require(
                ct <= budget,
                f"{method} budget={budget}: compressed_tokens ({ct}) > budget",
            )
    log_lines.append("Varied token budgets: OK (30/60/120 all within budget)")


def test_lightweight_baseline_methods_smoke(
    compressor: ContextCompressor, log_lines: list[str]
) -> None:
    """Smoke-test only the lightweight baseline methods (no ML model required).

    llmlingua and longllmlingua are intentionally NOT tested here to avoid
    downloading / loading the LLMLingua compression model.  Their behaviour
    has not been modified in stage 3 and they remain callable through the
    same ContextCompressor interface.
    """
    context = (
        "The quick brown fox jumps over the lazy dog. "
        "AI research is advancing rapidly."
    )
    question = "What jumps over the dog?"

    for method in ["original", "truncate", "bm25"]:
        result = compressor.compress_context(
            context=context,
            question=question,
            method=method,
            token_budget=30,
        )
        _require(result["method"] == method, f"{method} method field mismatch")
        _require(len(result["compressed_context"]) > 0, f"{method} returned empty")

    log_lines.append(
        "Lightweight baseline methods smoke test (original/truncate/bm25): OK"
    )
    log_lines.append(
        "Note: llmlingua/longllmlingua not tested here "
        "(no ML model load required for stage 3; their behaviour is unchanged)"
    )


def build_debug_csv(all_results: list[dict[str, Any]]) -> None:
    DEBUG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "original_tokens",
        "compressed_tokens",
        "token_budget",
        "compression_ratio",
        "token_saving_ratio",
        "compression_time",
        "fallback_used",
        "compressor_model_name",
        "compressed_context_preview",
    ]
    with open(DEBUG_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            preview = row.get("compressed_context", "")[:200].replace("\n", " | ")
            writer.writerow({
                "method": row["method"],
                "original_tokens": row["original_tokens"],
                "compressed_tokens": row["compressed_tokens"],
                "token_budget": row.get("token_budget", ""),
                "compression_ratio": round(row["compression_ratio"], 4),
                "token_saving_ratio": round(row["token_saving_ratio"], 4),
                "compression_time": round(row["compression_time"], 6),
                "fallback_used": row["fallback_used"],
                "compressor_model_name": row.get("compressor_model_name") or "",
                "compressed_context_preview": preview,
            })


def main() -> None:
    """Run the stage 3 Ours compressor verification tests.

    Tests lightweight methods only (ours_*, original, truncate, bm25).
    llmlingua / longllmlingua are intentionally skipped to avoid model
    downloads; their code paths are unchanged by stage 3.

    No DeepSeek API calls are made.
    """
    log_lines: list[str] = [
        "Stage 3 Ours compressor verification",
        f"checked_at={_now_iso()}",
        "No DeepSeek API call is made by this script.",
        "llmlingua/longllmlingua are NOT loaded or tested here (ML models unchanged by stage 3).",
        f"SUPPORTED_METHODS={sorted(SUPPORTED_METHODS)}",
        "",
    ]

    compressor = ContextCompressor(device_map="cpu")
    all_results: list[dict[str, Any]] = []

    try:
        test_registered_in_supported_methods(log_lines)
        log_lines.append("")

        r1 = test_ours_bm25_only_basic(compressor, log_lines)
        all_results.append(r1)

        r2 = test_ours_keyword_basic(compressor, log_lines)
        all_results.append(r2)

        r3 = test_ours_full_basic(compressor, log_lines)
        all_results.append(r3)

        log_lines.append("")
        test_empty_context(compressor, log_lines)
        test_empty_question(compressor, log_lines)
        test_chinese_segmentation(compressor, log_lines)
        test_mixed_language(compressor, log_lines)
        test_varied_token_budgets(compressor, log_lines)
        test_lightweight_baseline_methods_smoke(compressor, log_lines)

        # Collect varied-budget results for CSV
        for budget in [30, 60, 120]:
            for method in ["ours_bm25_only", "ours_keyword", "ours_full"]:
                all_results.append(
                    compressor.compress_context(
                        context=_ENGLISH_CONTEXT,
                        question=_ENGLISH_QUESTION,
                        method=method,
                        token_budget=budget,
                    )
                )

        build_debug_csv(all_results)

        log_lines.append("")
        log_lines.append("All stage 3 verification tests PASSED.")
        log_lines.append(f"Debug CSV written to: {DEBUG_CSV_PATH}")

    except Exception:
        log_lines.append("")
        log_lines.append("STAGE 3 VERIFICATION FAILED.")
        import traceback
        log_lines.append(traceback.format_exc())

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print("\n".join(log_lines))
        raise

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()

