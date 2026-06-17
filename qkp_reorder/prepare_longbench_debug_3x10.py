from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "stage7" / "processed" / "longbench_debug_3x10.json"
)

DATASET_CANDIDATES = [
    "THUDM/LongBench",
    "zai-org/LongBench",
]

TASKS = {
    "hotpotqa": {
        "config": "hotpotqa",
        "display_name": "HotpotQA",
    },
    "2wikimqa": {
        "config": "2wikimqa",
        "display_name": "2WikiMultihopQA",
    },
    "multifieldqa_en": {
        "config": "multifieldqa_en",
        "display_name": "MultiFieldQA-en",
    },
}

SAMPLES_PER_TASK = 10
SPLIT = "test"


def load_longbench_task(config_name: str):
    """Load a LongBench task from a candidate Hugging Face repository."""
    last_error = None

    for dataset_name in DATASET_CANDIDATES:
        try:
            print(f"Trying to load {dataset_name} / {config_name} / {SPLIT}")
            try:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=SPLIT,
                    trust_remote_code=True,
                )
            except TypeError:
                dataset = load_dataset(dataset_name, config_name, split=SPLIT)
            print(f"Loaded from {dataset_name}: {len(dataset)} rows")
            return dataset, dataset_name
        except Exception as exc:
            print(f"Failed to load from {dataset_name}: {repr(exc)}")
            last_error = exc

    raise RuntimeError(
        f"Failed to load LongBench config {config_name} from all candidates."
    ) from last_error


def normalize_answers(value: Any) -> List[str]:
    """Convert LongBench answers to a non-empty list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)] if str(value).strip() else []


def convert_record(
    record: Dict[str, Any],
    task_key: str,
    display_name: str,
    dataset_source: str,
    index: int,
) -> Dict[str, Any]:
    """Convert one LongBench record to the project unified QA format."""
    question = str(record.get("input", "") or "")
    context = str(record.get("context", "") or "")
    answers = normalize_answers(record.get("answers", []))
    raw_id = record.get("_id", f"{task_key}_{index:03d}")

    return {
        "task_name": task_key,
        "task_display_name": display_name,
        "sample_id": f"{task_key}_{index:03d}",
        "raw_id": str(raw_id),
        "question": question,
        "gold_answers": answers,
        "context": context,
        "documents": [
            {
                "doc_id": "doc_01",
                "title": f"{display_name} Context",
                "text": context,
            }
        ],
        "length": record.get("length"),
        "dataset": record.get("dataset", task_key),
        "language": record.get("language", ""),
        "source": dataset_source,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_samples: list[Dict[str, Any]] = []

    for task_key, task_info in TASKS.items():
        dataset, dataset_source = load_longbench_task(task_info["config"])
        count = 0

        for record in dataset:
            if count >= SAMPLES_PER_TASK:
                break

            sample = convert_record(
                record=dict(record),
                task_key=task_key,
                display_name=task_info["display_name"],
                dataset_source=dataset_source,
                index=count + 1,
            )
            if not sample["question"] or not sample["context"] or not sample["gold_answers"]:
                continue

            all_samples.append(sample)
            count += 1

        print(f"Converted {count} samples for task {task_key}.")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_samples, file, ensure_ascii=False, indent=2)

    task_counts: dict[str, int] = {}
    for sample in all_samples:
        task_counts[sample["task_name"]] = task_counts.get(sample["task_name"], 0) + 1

    print("\n" + "=" * 100)
    print(f"Saved {len(all_samples)} samples to: {OUTPUT_PATH}")
    print("Task counts:", task_counts)
    for sample in all_samples[:5]:
        print(
            sample["sample_id"],
            sample["task_name"],
            "answers:",
            sample["gold_answers"][:2],
            "context_chars:",
            len(sample["context"]),
        )


if __name__ == "__main__":
    main()

