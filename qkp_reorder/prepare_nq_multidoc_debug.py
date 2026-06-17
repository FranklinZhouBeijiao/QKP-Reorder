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
OUTPUT_PATH = PROJECT_ROOT / "data" / "stage4" / "processed" / "nq_multidoc_debug.json"

MAX_SAMPLES = 3
MAX_DOCS_PER_SAMPLE = 10


def open_text(path: Path):
    """Open plain or gzip-compressed UTF-8 text files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_raw_records(path: Path) -> List[Dict[str, Any]]:
    """Load raw records from JSON, JSONL, JSON.GZ, or JSONL.GZ."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
        records = []
        with open_text(path) as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.name.endswith(".json") or path.name.endswith(".json.gz"):
        with open_text(path) as file:
            obj = json.load(file)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for key in ["data", "examples", "samples", "records"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]

    raise ValueError(f"Unsupported raw data format: {path}")


def normalize_answers(raw_record: Dict[str, Any]) -> List[str]:
    """Extract gold answers from a raw Lost-in-the-Middle/NQ record."""
    for key in ["answers", "gold_answers", "answer"]:
        if key not in raw_record:
            continue

        value = raw_record[key]
        if isinstance(value, list):
            answers = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            answers = [value.strip()] if value.strip() else []
        else:
            answers = []

        if answers:
            return answers

    nq_gold = raw_record.get("nq_annotated_gold")
    if isinstance(nq_gold, dict):
        short_answers = nq_gold.get("short_answers")
        if isinstance(short_answers, list):
            return [str(item).strip() for item in short_answers if str(item).strip()]

    return []


def normalize_question(raw_record: Dict[str, Any]) -> str:
    """Extract question text from a raw record."""
    for key in ["question", "query", "input"]:
        value = raw_record.get(key)
        if value:
            return str(value).strip()
    return ""


def normalize_documents(raw_record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract documents from raw ctxs/documents fields."""
    docs_raw = None
    for key in ["ctxs", "documents", "docs", "contexts", "paragraphs"]:
        value = raw_record.get(key)
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
                or f"Document {index}"
            )
            text = (
                doc.get("text")
                or doc.get("contents")
                or doc.get("passage")
                or doc.get("paragraph")
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
    raw_record: Dict[str, Any],
    documents: List[Dict[str, str]],
    gold_answers: List[str],
) -> Optional[int]:
    """Infer a 1-based answer position from isgold/hasanswer flags or text match."""
    ctxs = raw_record.get("ctxs")
    if isinstance(ctxs, list):
        for index, doc in enumerate(ctxs[: len(documents)], start=1):
            if isinstance(doc, dict) and doc.get("isgold") is True:
                return index

        for index, doc in enumerate(ctxs[: len(documents)], start=1):
            if isinstance(doc, dict) and doc.get("hasanswer") is True:
                return index

    normalized_answers = [answer.lower() for answer in gold_answers if answer.strip()]
    for index, doc in enumerate(documents, start=1):
        text = doc.get("text", "").lower()
        if any(answer and answer in text for answer in normalized_answers):
            return index

    return None


def convert_record(raw_record: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Convert one raw record to the project-wide sample format."""
    question = normalize_question(raw_record)
    gold_answers = normalize_answers(raw_record)
    documents = normalize_documents(raw_record)

    if not question or not gold_answers or not documents:
        return None

    answer_position = (
        raw_record.get("answer_position")
        or raw_record.get("gold_doc_position")
        or raw_record.get("answer_doc_position")
        or infer_answer_position(raw_record, documents, gold_answers)
    )

    return {
        "sample_id": f"nq_debug_{index:03d}",
        "question": question,
        "gold_answers": gold_answers,
        "answer_position": answer_position,
        "documents": documents,
        "source": {
            "dataset": "lost-in-the-middle nq-open",
            "raw_file": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
        },
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_records(RAW_DATA_PATH)

    converted = []
    for raw_record in raw_records:
        sample = convert_record(raw_record, index=len(converted) + 1)
        if sample is None:
            continue

        converted.append(sample)
        if len(converted) >= MAX_SAMPLES:
            break

    if not converted:
        raise RuntimeError(
            "No samples were converted. Inspect the raw fields and update mappings."
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(converted)} samples to: {OUTPUT_PATH}")
    for sample in converted:
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

