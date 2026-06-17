from __future__ import annotations


def build_qa_prompt(context: str, question: str) -> str:
    """Build the unified QA prompt used by all compression methods."""
    context = context or ""
    question = question or ""

    return f"""You are given several documents. Answer the question based only on the given documents. Please answer as briefly as possible.

Documents:
{context}

Question:
{question}

Answer:
"""


def build_system_prompt() -> str:
    """Build the system prompt for the answer generation model."""
    return (
        "You are a helpful question-answering assistant. "
        "Answer based only on the provided documents."
    )

