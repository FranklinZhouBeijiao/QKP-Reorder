from __future__ import annotations

import time
from typing import Any, Dict, Optional

from rank_bm25 import BM25Okapi

from qkp_reorder.baselines import (
    bm25_context,
    original_context,
    ours_bm25_only_context,
    ours_full_context,
    ours_keyword_context,
    truncate_context,
)
from qkp_reorder.ours_utils import (
    compute_protection_bonus,
    contains_cjk,
    extract_question_keywords,
    split_into_segments,
    tokenize_for_bm25,
)
from qkp_reorder.tokenizer_utils import count_tokens, truncate_by_tokens


DEFAULT_COMPRESSOR_MODEL = (
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)
SUPPORTED_METHODS = {
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_bm25_only",
    "ours_keyword",
    "ours_full",
}


class CompressionResult(dict):
    """Dictionary result type for readability."""


class ContextCompressor:
    """Unified compressor wrapper for reproduction experiments."""

    def __init__(
        self,
        compressor_model_name: str = DEFAULT_COMPRESSOR_MODEL,
        device_map: str = "cpu",
        use_llmlingua2: bool = True,
    ):
        self.compressor_model_name = compressor_model_name
        self.device_map = device_map
        self.use_llmlingua2 = use_llmlingua2
        self._prompt_compressor: Any = None

    def _get_prompt_compressor(self):
        """Lazily initialize the official PromptCompressor."""
        if self._prompt_compressor is None:
            from llmlingua import PromptCompressor  # lazy import

            self._prompt_compressor = PromptCompressor(
                model_name=self.compressor_model_name,
                device_map=self.device_map,
                use_llmlingua2=self.use_llmlingua2,
            )
        return self._prompt_compressor

    def compress_context(
        self,
        context: str,
        question: str,
        method: str,
        token_budget: Optional[int] = None,
        rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compress context with one of the supported methods."""
        method = method.lower().strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown compression method: {method}. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_METHODS))}."
            )

        context = context or ""
        question = question or ""
        original_tokens = count_tokens(context)

        if token_budget is None:
            if rate is not None and rate > 0:
                token_budget = max(1, int(original_tokens * rate))
            else:
                token_budget = max(1, original_tokens // 2)

        start_time = time.time()

        fallback_used = False

        if method == "original":
            compressed_context = original_context(context)
        elif method == "truncate":
            compressed_context = truncate_context(context, token_budget)
        elif method == "bm25":
            compressed_context = bm25_context(context, question, token_budget)
        elif method == "llmlingua":
            compressed_context, fallback_used = self._compress_with_llmlingua(
                context=context,
                question=question,
                token_budget=token_budget,
                longllmlingua=False,
            )
        elif method == "longllmlingua":
            compressed_context, fallback_used = self._compress_with_llmlingua(
                context=context,
                question=question,
                token_budget=token_budget,
                longllmlingua=True,
            )
        elif method == "ours_bm25_only":
            compressed_context = ours_bm25_only_context(
                context=context,
                question=question,
                token_budget=token_budget,
            )
        elif method == "ours_keyword":
            compressed_context = ours_keyword_context(
                context=context,
                question=question,
                token_budget=token_budget,
            )
        elif method == "ours_full":
            compressed_context = ours_full_context(
                context=context,
                question=question,
                token_budget=token_budget,
            )

        end_time = time.time()
        compressed_tokens = count_tokens(compressed_context)
        compression_ratio = (
            original_tokens / compressed_tokens if compressed_tokens > 0 else 0.0
        )
        token_saving_ratio = (
            1.0 - compressed_tokens / original_tokens if original_tokens > 0 else 0.0
        )

        return CompressionResult(
            method=method,
            compressed_context=compressed_context,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            token_saving_ratio=token_saving_ratio,
            compression_time=end_time - start_time,
            token_budget=token_budget,
            fallback_used=fallback_used,
            compressor_model_name=self.compressor_model_name
            if method in {"llmlingua", "longllmlingua"}
            else None,
        )

    def _compress_with_llmlingua(
        self,
        context: str,
        question: str,
        token_budget: int,
        longllmlingua: bool,
    ) -> tuple[str, bool]:
        """Call the official LLMLingua compressor and report fallback usage."""
        if not context:
            return "", False

        compressor = self._get_prompt_compressor()
        fallback_used = False

        if longllmlingua:
            try:
                result = compressor.compress_prompt(
                    context,
                    instruction="Answer the question based only on the given context.",
                    question=question,
                    target_token=token_budget,
                    rank_method="longllmlingua",
                    condition_in_question="after",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3,
                    condition_compare=True,
                    add_instruction=False,
                )
            except TypeError:
                fallback_used = True
                result = compressor.compress_prompt(
                    context,
                    question=question,
                    target_token=token_budget,
                )
        else:
            result = compressor.compress_prompt(
                context,
                question=question,
                target_token=token_budget,
            )

        if isinstance(result, dict):
            compressed = result.get("compressed_prompt", "")
        else:
            compressed = str(result)

        if longllmlingua and self._should_apply_cjk_evidence_protection(
            context=context,
            question=question,
        ):
            compressed = self._protect_cjk_evidence_for_longllmlingua(
                context=context,
                question=question,
                compressed_context=compressed,
                token_budget=token_budget,
            )

        return compressed, fallback_used

    def _should_apply_cjk_evidence_protection(
        self,
        context: str,
        question: str,
    ) -> bool:
        """Only adapt LongLLMLingua for Chinese or mixed Chinese/English inputs."""
        return contains_cjk(context) or contains_cjk(question)

    def _protect_cjk_evidence_for_longllmlingua(
        self,
        context: str,
        question: str,
        compressed_context: str,
        token_budget: int,
    ) -> str:
        """Post-protect question-relevant CJK evidence without using gold answers."""
        protected_context = self._select_protected_segments(
            context=context,
            question=question,
            token_budget=token_budget,
        )

        candidates = [
            item.strip()
            for item in [protected_context, compressed_context]
            if item and item.strip()
        ]
        if not candidates:
            return ""

        merged: list[str] = []
        for candidate in candidates:
            if any(candidate in existing or existing in candidate for existing in merged):
                continue
            merged.append(candidate)

        return truncate_by_tokens("\n".join(merged), token_budget)

    def _select_protected_segments(
        self,
        context: str,
        question: str,
        token_budget: int,
    ) -> str:
        """Select question-aware protected evidence from context only."""
        question_tokens = tokenize_for_bm25(question)
        if not question_tokens:
            return truncate_by_tokens(context, token_budget)

        segments = split_into_segments(context)
        if not segments:
            return ""

        tokenized_segments = [tokenize_for_bm25(segment) for segment in segments]
        bm25 = BM25Okapi(tokenized_segments)
        bm25_scores = bm25.get_scores(question_tokens)
        keywords = extract_question_keywords(question)

        combined_scores = [
            bm25_scores[index] + compute_protection_bonus(segments[index], keywords)
            for index in range(len(segments))
        ]
        ranked_indices = sorted(
            range(len(segments)),
            key=lambda index: (combined_scores[index], -index),
            reverse=True,
        )

        selected: list[tuple[int, str]] = []
        current_tokens = 0
        best_over_budget: Optional[str] = None
        for index in ranked_indices:
            segment = segments[index]
            segment_tokens = count_tokens(segment)
            if segment_tokens > token_budget:
                if best_over_budget is None:
                    best_over_budget = segment
                continue
            if current_tokens + segment_tokens > token_budget:
                continue
            selected.append((index, segment))
            current_tokens += segment_tokens

        if not selected:
            return truncate_by_tokens(best_over_budget or context, token_budget)

        selected.sort(key=lambda item: item[0])
        return truncate_by_tokens(
            "\n".join(segment for _, segment in selected),
            token_budget,
        )

