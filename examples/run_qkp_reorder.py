from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import ContextCompressor


CONTEXT = """
Long-context question answering often gives a model many documents at once.
Some documents discuss evaluation data, while only a few contain the evidence
needed for the current question.

Simple truncation can remove evidence when the answer appears in the middle of
the context. QKP-Reorder scores segments with question-aware BM25, protects
keywords, entities, years, percentages, and numbers, then moves high-value
evidence toward the front and back of the compressed context.
"""

QUESTION = "What problem does QKP-Reorder try to reduce?"


def main() -> None:
    compressor = ContextCompressor()
    result = compressor.compress_context(
        context=CONTEXT,
        question=QUESTION,
        method="ours_full",
        token_budget=80,
    )

    print("method:", result["method"])
    print("original_tokens:", result["original_tokens"])
    print("compressed_tokens:", result["compressed_tokens"])
    print("token_saving_ratio:", round(result["token_saving_ratio"], 3))
    print()
    print(result["compressed_context"])


if __name__ == "__main__":
    main()
