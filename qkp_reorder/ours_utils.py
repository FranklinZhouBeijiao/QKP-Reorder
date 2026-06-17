from __future__ import annotations

import re
from typing import Iterable


try:
    import jieba
except ImportError:  # pragma: no cover - depends on local environment
    jieba = None


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:[_-][A-Za-z0-9]+)*|"
    r"\d+(?:,\d{3})*(?:\.\d+)?%?|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9_])")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:\d+)?\b")
_MIXED_CAP_RE = re.compile(r"\b[A-Za-z]*[A-Z][a-z]+[A-Z][A-Za-z]*\b")
_JOINED_ENTITY_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b")
_BOOK_TITLE_RE = re.compile(r"《([^》]{1,50})》")
_QUOTED_CJK_RE = re.compile(r"[“\"']([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,20})[”\"']")

_SEGMENT_DELIMITERS = {
    ".",
    "!",
    "?",
    "。",
    "！",
    "？",
    "；",
    ";",
    "，",
    "、",
}

_ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}

_CHINESE_QUESTION_STOPWORDS = {
    "什么",
    "谁",
    "哪里",
    "哪儿",
    "哪",
    "哪一",
    "哪一年",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "多少",
    "几",
    "吗",
    "呢",
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
}

_KEYWORD_BONUS_WEIGHT = 0.6
_NUMBER_BONUS_WEIGHT = 0.25
_ENTITY_BONUS_WEIGHT = 0.4
_MAX_KEYWORD_BONUS = 3.0
_MAX_NUMBER_BONUS = 1.0
_MAX_ENTITY_BONUS = 1.6


def is_jieba_available() -> bool:
    """Return whether optional jieba tokenization is available."""
    return jieba is not None


def contains_cjk(text: str) -> bool:
    """Return True if text contains a CJK character."""
    return bool(text and _CJK_RE.search(text))


def split_into_segments(text: str) -> list[str]:
    """Split English, Chinese, or mixed text into sentence-like segments."""
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []
    current: list[str] = []

    for index, char in enumerate(normalized):
        if char == "\n":
            _append_segment(segments, current)
            continue

        current.append(char)
        if _should_split_at(normalized, index):
            _append_segment(segments, current)

    _append_segment(segments, current)
    return segments


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25-style scoring."""
    if not text:
        return []

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw_token = match.group(0)
        if contains_cjk(raw_token):
            tokens.extend(_tokenize_cjk_chunk(raw_token))
        else:
            tokens.append(raw_token.lower())

    return [token for token in tokens if token]


def extract_question_keywords(question: str) -> list[str]:
    """Extract question keywords without using gold answers or labels."""
    if not question:
        return []

    cleaned_question = question
    for stopword in sorted(_CHINESE_QUESTION_STOPWORDS, key=len, reverse=True):
        cleaned_question = cleaned_question.replace(stopword, " ")

    keywords: list[str] = []
    for token in tokenize_for_bm25(cleaned_question):
        normalized = token.lower() if token.isascii() else token
        if _is_stopword(normalized):
            continue
        if normalized.isascii() and len(normalized) <= 1 and not normalized.isdigit():
            continue
        keywords.append(normalized)

    return _deduplicate(keywords)


def extract_numbers(text: str) -> list[str]:
    """Extract years, integers, decimals, and percentages from text."""
    if not text:
        return []

    return _deduplicate(match.group(0) for match in _NUMBER_RE.finditer(text))


def extract_entity_like_terms(text: str) -> list[str]:
    """Extract lightweight entity-like strings from a segment."""
    if not text:
        return []

    entities: list[str] = []
    entities.extend(match.group(0) for match in _CAPITALIZED_PHRASE_RE.finditer(text))
    entities.extend(match.group(0) for match in _ACRONYM_RE.finditer(text))
    entities.extend(match.group(0) for match in _MIXED_CAP_RE.finditer(text))
    entities.extend(match.group(0) for match in _JOINED_ENTITY_RE.finditer(text))
    entities.extend(match.group(1) for match in _BOOK_TITLE_RE.finditer(text))
    entities.extend(match.group(1) for match in _QUOTED_CJK_RE.finditer(text))

    return _deduplicate(entities)


def compute_protection_bonus(segment: str, keywords: list[str]) -> float:
    """Compute an interpretable protection score for keyword/entity/number evidence."""
    if not segment:
        return 0.0

    keyword_hits = _count_keyword_hits(segment, keywords)
    numbers = extract_numbers(segment)
    entities = extract_entity_like_terms(segment)

    keyword_bonus = min(keyword_hits * _KEYWORD_BONUS_WEIGHT, _MAX_KEYWORD_BONUS)
    number_bonus = min(len(numbers) * _NUMBER_BONUS_WEIGHT, _MAX_NUMBER_BONUS)
    entity_bonus = min(len(entities) * _ENTITY_BONUS_WEIGHT, _MAX_ENTITY_BONUS)

    return round(keyword_bonus + number_bonus + entity_bonus, 4)


def _append_segment(segments: list[str], current: list[str]) -> None:
    segment = "".join(current).strip()
    current.clear()
    if segment:
        segments.append(segment)


def _should_split_at(text: str, index: int) -> bool:
    char = text[index]
    if char not in _SEGMENT_DELIMITERS:
        return False

    if char == ".":
        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if previous_char.isdigit() and next_char.isdigit():
            return False
        if previous_char.isupper() and next_char.isupper():
            return False

    return True


def _tokenize_cjk_chunk(chunk: str) -> list[str]:
    if jieba is not None:
        return [
            token.strip()
            for token in jieba.lcut(chunk)
            if token.strip() and not _is_stopword(token.strip())
        ]

    return [char for char in chunk if contains_cjk(char) and not _is_stopword(char)]


def _is_stopword(token: str) -> bool:
    return token in _ENGLISH_STOPWORDS or token in _CHINESE_QUESTION_STOPWORDS


def _count_keyword_hits(segment: str, keywords: list[str]) -> int:
    if not keywords:
        return 0

    segment_tokens = set(tokenize_for_bm25(segment))
    segment_lower = segment.lower()
    hits = 0

    for keyword in _deduplicate(keywords):
        normalized = keyword.lower() if keyword.isascii() else keyword
        if not normalized or _is_stopword(normalized):
            continue
        if normalized in segment_tokens:
            hits += 1
        elif contains_cjk(normalized) and normalized in segment:
            hits += 1
        elif normalized.isascii() and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
            segment_lower,
        ):
            hits += 1

    return hits


def _deduplicate(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return deduped

