from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "stage4" / "raw" / "lost-in-the-middle"


def open_text(path: Path):
    """Open plain or gzip-compressed text files as UTF-8."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def candidate_files(root: Path) -> Iterable[Path]:
    """Yield candidate structured data files under the raw data root."""
    patterns = ["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz", "*.csv", "*.tsv"]
    for pattern in patterns:
        yield from root.rglob(pattern)


def inspect_jsonl(path: Path, max_lines: int = 2) -> None:
    print(f"\nInspecting JSONL: {path}")
    with open_text(path) as file:
        shown = 0
        for line in file:
            if shown >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:4000])
            shown += 1


def inspect_json(path: Path) -> None:
    print(f"\nInspecting JSON: {path}")
    with open_text(path) as file:
        obj = json.load(file)

    print("type:", type(obj))
    if isinstance(obj, list):
        print("len:", len(obj))
        if obj:
            print(json.dumps(obj[0], indent=2, ensure_ascii=False)[:4000])
    elif isinstance(obj, dict):
        print("keys:", list(obj.keys()))
        print(json.dumps(obj, indent=2, ensure_ascii=False)[:4000])
    else:
        print("Unsupported JSON root type.")


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {RAW_ROOT}")

    files = sorted(candidate_files(RAW_ROOT))
    print(f"Found {len(files)} candidate structured data files.")
    for path in files:
        print(path)

    print("\n" + "=" * 100)
    print("Inspecting first candidate JSON/JSONL files...")
    print("=" * 100)

    inspected = 0
    for path in files:
        if ".json" not in path.name:
            continue
        try:
            if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
                inspect_jsonl(path)
            elif path.name.endswith(".json") or path.name.endswith(".json.gz"):
                inspect_json(path)
            inspected += 1
        except Exception as exc:
            print(f"Failed to inspect {path}: {exc}")

        if inspected >= 20:
            break


if __name__ == "__main__":
    main()

