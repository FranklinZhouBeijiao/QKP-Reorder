from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "stage4"
    / "raw"
    / "lost-in-the-middle"
    / "qa_data"
    / "10_total_documents"
    / "nq-open-10_total_documents_gold_at_4.jsonl.gz"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "stage5" / "processed" / "nq_multidoc_small_20.json"

MAX_SAMPLES = 20
MAX_DOCS_PER_SAMPLE = 10
SOURCE_NAME = "lost-in-the-middle/nq-open-10_total_documents_gold_at_4"


def load_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    """Load records from a .jsonl.gz file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    records = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_question(record: Dict[str, Any]) -> str:
    """Extract question text from a raw record."""
    for key in ["question", "query", "input"]:
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def extract_answers(record: Dict[str, Any]) -> List[str]:
    """Extract gold answers from a raw record."""
    for key in ["answers", "gold_answers", "answer"]:
        if key not in record:
            continue

        value = record[key]
        if isinstance(value, list):
            answers = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            answers = [value.strip()] if value.strip() else []
        else:
            answers = []

        if answers:
            return answers

    nq_gold = record.get("nq_annotated_gold")
    if isinstance(nq_gold, dict):
        short_answers = nq_gold.get("short_answers")
        if isinstance(short_answers, list):
            return [str(item).strip() for item in short_answers if str(item).strip()]

    return []


def extract_documents(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract documents from a raw Lost-in-the-Middle/NQ record."""
    docs_raw = None
    for key in ["contexts", "ctxs", "documents", "docs"]:
        value = record.get(key)
        if isinstance(value, list):
            docs_raw = value
            break

    if docs_raw is None:
        return []

    documents = []
    for index, doc in enumerate(docs_raw[:MAX_DOCS_PER_SAMPLE], start=1):
        if isinstance(doc, str):
            title = f"Document {index}"
            text = doc
            doc_id = f"doc_{index:02d}"
        elif isinstance(doc, dict):
            title = (
                doc.get("title")
                or doc.get("document_title")
                or doc.get("wikipedia_title")
                or doc.get("name")
                or f"Document {index}"
            )
            text = (
                doc.get("text")
                or doc.get("contents")
                or doc.get("passage")
                or doc.get("paragraph")
                or doc.get("content")
                or ""
            )
            doc_id = doc.get("id") or doc.get("doc_id") or f"doc_{index:02d}"
        else:
            continue

        if not str(text).strip():
            continue

        documents.append(
            {
                "doc_id": str(doc_id),
                "title": str(title),
                "text": str(text),
            }
        )

    return documents


def infer_answer_position(
    record: Dict[str, Any],
    documents: List[Dict[str, str]],
    gold_answers: List[str],
) -> Optional[int]:
    """Infer 1-based answer position from raw flags or answer string matching."""
    ctxs = record.get("ctxs")
    if isinstance(ctxs, list):
        for index, doc in enumerate(ctxs[: len(documents)], start=1):
            if isinstance(doc, dict) and doc.get("isgold") is True:
                return index

        for index, doc in enumerate(ctxs[: len(documents)], start=1):
            if isinstance(doc, dict) and doc.get("hasanswer") is True:
                return index

    answers = [answer.lower() for answer in gold_answers if answer.strip()]
    for index, doc in enumerate(documents, start=1):
        text = doc.get("text", "").lower()
        if any(answer and answer in text for answer in answers):
            return index

    return None


def convert_record(record: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Convert one raw record to the project-wide multi-document QA format."""
    question = extract_question(record)
    gold_answers = extract_answers(record)
    documents = extract_documents(record)

    if not question or not gold_answers or len(documents) < MAX_DOCS_PER_SAMPLE:
        return None

    answer_position = (
        record.get("answer_position")
        or record.get("gold_doc_position")
        or record.get("answer_doc_position")
        or infer_answer_position(record, documents, gold_answers)
        or 5
    )

    return {
        "sample_id": f"nq_small_{index:03d}",
        "question": question,
        "gold_answers": gold_answers,
        "answer_position": answer_position,
        "documents": documents,
        "source": SOURCE_NAME,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_records = load_jsonl_gz(RAW_DATA_PATH)
    converted = []

    for raw_record in raw_records:
        sample = convert_record(raw_record, index=len(converted) + 1)
        if sample is None:
            continue

        converted.append(sample)
        if len(converted) >= MAX_SAMPLES:
            break

    if len(converted) < MAX_SAMPLES:
        print(f"Warning: only converted {len(converted)} samples.")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(converted)} samples to: {OUTPUT_PATH}")
    for sample in converted[:5]:
        print(
            sample["sample_id"],
            "docs:",
            len(sample["documents"]),
            "answer_position:",
            sample["answer_position"],
            "question:",
            sample["question"][:100],
        )


if __name__ == "__main__":
    main()

