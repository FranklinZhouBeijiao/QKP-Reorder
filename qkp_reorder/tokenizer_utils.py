from __future__ import annotations

import tiktoken


_DEFAULT_ENCODING = "cl100k_base"


def get_encoding(encoding_name: str = _DEFAULT_ENCODING):
    """Return a tiktoken encoding object."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """Count tokens for a given text using the project-wide encoding."""
    if text is None:
        return 0

    enc = get_encoding(encoding_name)
    return len(enc.encode(text))


def truncate_by_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = _DEFAULT_ENCODING,
) -> str:
    """Truncate text to at most max_tokens tokens."""
    if text is None:
        return ""

    if max_tokens is None or max_tokens <= 0:
        return text

    enc = get_encoding(encoding_name)
    token_ids = enc.encode(text)

    if len(token_ids) <= max_tokens:
        return text

    return enc.decode(token_ids[:max_tokens])

