import time

from llmlingua import PromptCompressor


def main():
    print("Initializing PromptCompressor...")

    # The default PromptCompressor model is a 7B causal LM on CUDA, which is not
    # suitable for this Windows CPU stage-0 check. Use the official LLMLingua-2
    # compression model instead.
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device_map="cpu",
        use_llmlingua2=True,
    )

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

    print("Running LongLLMLingua compression...")

    start_time = time.time()

    try:
        compressed = compressor.compress_prompt(
            context,
            instruction="Answer the question based only on the given context.",
            question=question,
            target_token=120,
            rank_method="longllmlingua",
            condition_in_question="after",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            add_instruction=True,
        )
    except TypeError as exc:
        print(f"Advanced LongLLMLingua parameters are not compatible: {exc}")
        print("Retrying with the minimal stage-0 compression call...")
        compressed = compressor.compress_prompt(
            context,
            question=question,
            target_token=120,
        )

    end_time = time.time()

    print("\n========== Raw returned object ==========")
    print(compressed)

    print("\n========== Key fields ==========")
    print("origin_tokens:", compressed.get("origin_tokens"))
    print("compressed_tokens:", compressed.get("compressed_tokens"))
    print("ratio:", compressed.get("ratio"))

    print("\n========== Compressed prompt ==========")
    print(compressed.get("compressed_prompt", ""))

    print("\n========== Compression time ==========")
    print(f"{end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

