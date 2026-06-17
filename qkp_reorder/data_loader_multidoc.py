from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_multidoc_qa(path: str | Path) -> List[Dict[str, Any]]:
    """Load and validate multi-document QA samples from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The data file must contain a list of samples.")

    for sample in data:
        validate_multidoc_sample(sample)

    return data


def validate_multidoc_sample(sample: Dict[str, Any]) -> None:
    """Validate one multi-document QA sample."""
    required_fields = [
        "sample_id",
        "question",
        "gold_answers",
        "documents",
    ]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing required field `{field}` in sample: {sample}")

    sample_id = sample["sample_id"]
    if not isinstance(sample["gold_answers"], list):
        raise ValueError(f"`gold_answers` must be a list in sample {sample_id}")

    documents = sample["documents"]
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"`documents` must be a non-empty list in sample {sample_id}")

    for doc in documents:
        if "text" not in doc:
            raise ValueError(f"Each document must contain `text` in sample {sample_id}")


def format_documents_as_context(documents: List[Dict[str, Any]]) -> str:
    """Convert document objects into a single multi-document context string."""
    parts = []
    for index, doc in enumerate(documents, start=1):
        title = doc.get("title", f"Document {index}")
        text = doc.get("text", "")
        doc_id = doc.get("doc_id", f"doc_{index}")
        parts.append(f"[Document {index}]\nTitle: {title}\nID: {doc_id}\n{text}")

    return "\n\n".join(parts)


def get_answer_position(sample: Dict[str, Any]) -> Optional[int]:
    """Return the 1-based answer document position when present."""
    return sample.get("answer_position")

