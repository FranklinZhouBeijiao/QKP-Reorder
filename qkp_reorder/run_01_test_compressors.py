from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import ContextCompressor


def main():
    context = """
    LongLLMLingua is a prompt compression method designed for long-context scenarios.
    It aims to reduce the number of input tokens while preserving the information needed to answer a question.

    In long-context question answering, a user may provide many documents to a large language model.
    However, many of these documents may be irrelevant to the current question.
    Feeding all documents into the model can increase cost, latency, and the risk of distraction.

    LongLLMLingua uses question-aware compression to identify important parts of the context.
    It also considers document reordering to mitigate the lost in the middle problem.
    The lost in the middle problem means that large language models may pay less attention to information placed in the middle of a long context.

    By compressing and reordering the prompt, LongLLMLingua can reduce inference cost and sometimes improve answer accuracy.
    """

    question = "What problem does LongLLMLingua try to mitigate in long-context scenarios?"
    methods = [
        "original",
        "truncate",
        "bm25",
        "llmlingua",
        "longllmlingua",
    ]

    compressor = ContextCompressor(device_map="cpu")

    for method in methods:
        print("\n" + "=" * 80)
        print(f"Method: {method}")

        result = compressor.compress_context(
            context=context,
            question=question,
            method=method,
            token_budget=100,
        )

        print("original_tokens:", result["original_tokens"])
        print("compressed_tokens:", result["compressed_tokens"])
        print("compression_ratio:", round(result["compression_ratio"], 3))
        print("token_saving_ratio:", round(result["token_saving_ratio"], 3))
        print("compression_time:", round(result["compression_time"], 3))
        print("fallback_used:", result["fallback_used"])
        print("compressor_model_name:", result["compressor_model_name"])

        print("\nCompressed context:")
        print(result["compressed_context"])


if __name__ == "__main__":
    main()

