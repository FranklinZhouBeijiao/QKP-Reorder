from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from qkp_reorder.ours_utils import (
    compute_protection_bonus,
    extract_question_keywords,
    split_into_segments,
    tokenize_for_bm25,
)
from qkp_reorder.tokenizer_utils import count_tokens, truncate_by_tokens


def split_into_sentences(text: str) -> List[str]:
    """Split English debug text into simple sentence-like chunks."""
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 scoring."""
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_]+", text)


def original_context(context: str) -> str:
    """Return the original context without compression."""
    return context or ""


def truncate_context(context: str, token_budget: int) -> str:
    """Truncate context to a fixed token budget."""
    return truncate_by_tokens(context or "", token_budget)


def bm25_context(context: str, question: str, token_budget: int) -> str:
    """Select question-relevant sentences with BM25 under a token budget."""
    context = context or ""
    question = question or ""

    sentences = split_into_sentences(context)
    if not sentences:
        return ""

    tokenized_question = simple_tokenize(question)
    if not tokenized_question:
        return truncate_by_tokens(context, token_budget)

    tokenized_sentences = [simple_tokenize(sentence) for sentence in sentences]
    bm25 = BM25Okapi(tokenized_sentences)
    scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = []
    current_tokens = 0
    for index in ranked_indices:
        sentence = sentences[index]
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > token_budget:
            continue

        selected.append((index, sentence))
        current_tokens += sentence_tokens

        if current_tokens >= token_budget:
            break

    selected.sort(key=lambda item: item[0])
    compressed = "\n".join(sentence for _, sentence in selected)

    if not compressed:
        return truncate_by_tokens(context, token_budget)

    return compressed


def ours_bm25_only_context(context: str, question: str, token_budget: int) -> str:
    """CJK-compatible segmentation + question-aware BM25 selection under token budget.

    Uses split_into_segments and tokenize_for_bm25 for mixed Chinese/English
    support.  No keyword protection or middle-aware reordering.
    """
    context = context or ""
    question = question or ""

    if not context:
        return ""

    question_tokens = tokenize_for_bm25(question)
    if not question_tokens:
        return truncate_by_tokens(context, token_budget)

    segments = split_into_segments(context)
    if not segments:
        return ""

    tokenized_segments = [tokenize_for_bm25(seg) for seg in segments]
    bm25 = BM25Okapi(tokenized_segments)
    scores = bm25.get_scores(question_tokens)

    ranked = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)

    selected: list[tuple[int, str]] = []
    current_tokens = 0
    for idx in ranked:
        seg = segments[idx]
        seg_tokens = count_tokens(seg)
        if current_tokens + seg_tokens > token_budget:
            continue
        selected.append((idx, seg))
        current_tokens += seg_tokens

    if not selected:
        return truncate_by_tokens(context, token_budget)

    selected.sort(key=lambda x: x[0])
    compressed = "\n".join(seg for _, seg in selected)
    return truncate_by_tokens(compressed, token_budget)


def ours_keyword_context(context: str, question: str, token_budget: int) -> str:
    """ours_bm25_only + keyword, entity, and number protection bonuses.

    BM25 scores are augmented with compute_protection_bonus before ranking.
    Selected segments are returned in original order.
    """
    context = context or ""
    question = question or ""

    if not context:
        return ""

    question_tokens = tokenize_for_bm25(question)
    if not question_tokens:
        return truncate_by_tokens(context, token_budget)

    keywords = extract_question_keywords(question)

    segments = split_into_segments(context)
    if not segments:
        return ""

    tokenized_segments = [tokenize_for_bm25(seg) for seg in segments]
    bm25 = BM25Okapi(tokenized_segments)
    bm25_scores = bm25.get_scores(question_tokens)

    combined_scores = [
        bm25_scores[i] + compute_protection_bonus(segments[i], keywords)
        for i in range(len(segments))
    ]

    ranked = sorted(range(len(segments)), key=lambda i: combined_scores[i], reverse=True)

    selected: list[tuple[int, str]] = []
    current_tokens = 0
    for idx in ranked:
        seg = segments[idx]
        seg_tokens = count_tokens(seg)
        if current_tokens + seg_tokens > token_budget:
            continue
        selected.append((idx, seg))
        current_tokens += seg_tokens

    if not selected:
        return truncate_by_tokens(context, token_budget)

    selected.sort(key=lambda x: x[0])
    compressed = "\n".join(seg for _, seg in selected)
    return truncate_by_tokens(compressed, token_budget)


def ours_full_context(context: str, question: str, token_budget: int) -> str:
    """ours_keyword + middle-aware reordering.

    After selection under token budget, high-scoring segments are placed
    at the front and back of the compressed text while lower-scoring
    segments remain in original order in the middle.  This targets the
    lost-in-the-middle phenomenon.
    """
    context = context or ""
    question = question or ""

    if not context:
        return ""

    question_tokens = tokenize_for_bm25(question)
    if not question_tokens:
        return truncate_by_tokens(context, token_budget)

    keywords = extract_question_keywords(question)

    segments = split_into_segments(context)
    if not segments:
        return ""

    tokenized_segments = [tokenize_for_bm25(seg) for seg in segments]
    bm25 = BM25Okapi(tokenized_segments)
    bm25_scores = bm25.get_scores(question_tokens)

    combined_scores = [
        bm25_scores[i] + compute_protection_bonus(segments[i], keywords)
        for i in range(len(segments))
    ]

    ranked = sorted(range(len(segments)), key=lambda i: combined_scores[i], reverse=True)

    selected: list[tuple[int, str, float]] = []
    current_tokens = 0
    for idx in ranked:
        seg = segments[idx]
        seg_tokens = count_tokens(seg)
        if current_tokens + seg_tokens > token_budget:
            continue
        selected.append((idx, seg, combined_scores[idx]))
        current_tokens += seg_tokens

    if not selected:
        return truncate_by_tokens(context, token_budget)

    if len(selected) <= 2:
        compressed = "\n".join(seg for _, seg, _ in selected)
        return truncate_by_tokens(compressed, token_budget)

    # Sort selected by score descending for reordering
    by_score = sorted(selected, key=lambda x: x[2], reverse=True)
    high_count = max(2, len(selected) // 2)

    high_value = by_score[:high_count]
    low_value = by_score[high_count:]

    front: list[str] = []
    back: list[str] = []
    for i, (_, seg, _) in enumerate(high_value):
        if i % 2 == 0:
            front.append(seg)
        else:
            back.append(seg)
    back.reverse()

    middle = [seg for _, seg, _ in sorted(low_value, key=lambda x: x[0])]

    parts = front + middle + back
    compressed = "\n".join(parts)
    return truncate_by_tokens(compressed, token_budget)

