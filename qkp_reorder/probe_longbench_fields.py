from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset


DATASET_NAME = "THUDM/LongBench"
TASKS = {
    "hotpotqa": "HotpotQA",
    "2wikimqa": "2WikiMultihopQA",
    "multifieldqa_en": "MultiFieldQA-en",
}
OUTPUT_PATH = Path("logs/longbench_field_probe.json")


def get_configs(dataset_name: str) -> list[str]:
    try:
        return get_dataset_config_names(dataset_name, trust_remote_code=True)
    except TypeError:
        return get_dataset_config_names(dataset_name)


def load_longbench_task(config_name: str):
    try:
        return load_dataset(DATASET_NAME, config_name, trust_remote_code=True)
    except TypeError:
        return load_dataset(DATASET_NAME, config_name)


def preview(value: Any, max_chars: int = 220) -> Any:
    if isinstance(value, str):
        text = value.replace("\n", " ")
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    if isinstance(value, Mapping):
        return {key: preview(item, max_chars=120) for key, item in list(value.items())[:5]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [preview(item, max_chars=120) for item in list(value)[:3]]
    return value


def describe_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key, value in sample.items():
        fields[key] = {
            "type": type(value).__name__,
            "preview": preview(value),
        }
    return fields


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    configs = get_configs(DATASET_NAME)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Available configs: {len(configs)}")

    results: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "available_configs": configs,
        "tasks": {},
    }

    for config_name, display_name in TASKS.items():
        print("\n" + "=" * 100)
        print(f"Task: {display_name} ({config_name})")

        if config_name not in configs:
            print(f"Config not found: {config_name}")
            results["tasks"][config_name] = {
                "display_name": display_name,
                "found": False,
            }
            continue

        dataset_dict = load_longbench_task(config_name)
        split_names = list(dataset_dict.keys())
        split_name = "test" if "test" in dataset_dict else split_names[0]
        dataset = dataset_dict[split_name]
        sample = dataset[0]

        print(f"Splits: {split_names}")
        print(f"Selected split: {split_name}")
        print(f"Rows: {len(dataset)}")
        print(f"Column names: {dataset.column_names}")
        print("First sample field summary:")

        field_summary = describe_sample(sample)
        for key, info in field_summary.items():
            print(f"- {key}: {info['type']} -> {info['preview']}")

        results["tasks"][config_name] = {
            "display_name": display_name,
            "found": True,
            "splits": split_names,
            "selected_split": split_name,
            "num_rows": len(dataset),
            "column_names": dataset.column_names,
            "first_sample_fields": field_summary,
        }

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 100)
    print(f"Saved probe result to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

