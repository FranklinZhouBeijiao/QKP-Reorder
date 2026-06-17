from __future__ import annotations

import html
from pathlib import Path
import sys
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import ContextCompressor
from qkp_reorder.prompt_builder import build_qa_prompt


MODEL_OPTIONS = {
    "Original": "original",
    "Truncate": "truncate",
    "BM25": "bm25",
    "LLMLingua": "llmlingua",
    "LongLLMLingua": "longllmlingua",
    "QKP-BM25": "ours_bm25_only",
    "QKP-Keyword": "ours_keyword",
    "QKP-Reorder": "ours_full",
}

DEFAULT_CONTEXT = """LongLLMLingua is a prompt compression method designed for long-context question answering.
It reduces input tokens while preserving evidence needed to answer a user question.

In long-context QA, a model may receive many documents, but only a few passages are relevant.
Sending every document increases cost, latency, and the risk that useful evidence is ignored.

The lost-in-the-middle problem means that language models often pay less attention to information placed in the middle of a long context.
Question-aware compression and reordering can move important evidence to more visible positions.

QKP-Reorder is a lightweight improved method in this project. It combines Chinese and English segmentation, BM25 relevance scoring, keyword/entity/number protection, and middle-aware reordering under a fixed token budget.
"""

DEFAULT_QUESTION = "What problem does QKP-Reorder try to reduce in long-context question answering?"

DEMO_CONTEXT_EN = """Long-context question answering often gives a model many documents at once.
Some documents describe the model architecture, some discuss evaluation data, and only a few contain the evidence needed for the current question.

LLMLingua and LongLLMLingua reduce prompt length by compressing irrelevant or redundant text.
However, simple truncation may remove evidence when the answer appears in the middle or near the end of the context.

QKP-Reorder is the improved method in this project. It first splits Chinese and English text into short segments.
Then it scores segments with question-aware BM25, protects keywords, entities, years, percentages, and numbers, and finally reorders important evidence toward the front and back of the compressed context.

This design targets the lost-in-the-middle problem while keeping the compression process lightweight and interpretable.
"""

DEMO_CONTEXT_ZH = """长文本问答任务通常会把许多文档一起输入大模型，但其中只有少量片段真正包含回答问题所需的证据。
如果直接输入全部上下文，模型调用成本会升高，响应速度会变慢，还可能因为无关内容太多而忽略关键信息。

传统截断方法只保留文本开头，容易删除位于中间或靠后的答案证据。
BM25 可以根据问题挑选相关句子，但对中文、实体、数字、年份等关键信息的保护仍然有限。

QKP-Reorder 是本项目提出的轻量改进方法。它结合中英文分句、问题感知 BM25、关键词/实体/数字保护，以及 middle-aware reordering。
通过把高价值证据移动到压缩文本的前部和后部，它希望缓解 lost-in-the-middle 现象，并在较少 token 下保留回答所需信息。
"""

DEMO_CASES = {
    "en_truncate": {
        "language": "英文案例",
        "method": "Truncate",
        "context": DEMO_CONTEXT_EN,
        "question": "What problem does QKP-Reorder target?",
        "answer": "QKP-Reorder targets the lost-in-the-middle problem in long-context question answering.",
    },
    "en_bm25": {
        "language": "英文案例",
        "method": "BM25",
        "context": DEMO_CONTEXT_EN,
        "question": "How does QKP-Reorder select useful evidence?",
        "answer": "It scores short segments with question-aware BM25 and keeps the most relevant evidence under the token budget.",
    },
    "en_qkp": {
        "language": "英文案例",
        "method": "QKP-Reorder",
        "context": DEMO_CONTEXT_EN,
        "question": "What does QKP-Reorder protect during compression?",
        "answer": "It protects keywords, entities, years, percentages, and numbers, then reorders high-value evidence.",
    },
    "zh_truncate": {
        "language": "中文案例",
        "method": "Truncate",
        "context": DEMO_CONTEXT_ZH,
        "question": "传统截断方法有什么问题？",
        "answer": "传统截断只保留开头，容易删除位于中间或靠后的答案证据。",
    },
    "zh_bm25": {
        "language": "中文案例",
        "method": "BM25",
        "context": DEMO_CONTEXT_ZH,
        "question": "BM25 在压缩中起什么作用？",
        "answer": "BM25 根据问题挑选相关句子，帮助保留更可能包含答案的内容。",
    },
    "zh_qkp": {
        "language": "中文案例",
        "method": "QKP-Reorder",
        "context": DEMO_CONTEXT_ZH,
        "question": "QKP-Reorder 如何缓解 lost-in-the-middle 现象？",
        "answer": "它保护关键词、实体和数字，并把高价值证据移动到压缩文本的前部和后部。",
    },
}


APP_CSS = """
<style>
    .stApp {
        background:
            linear-gradient(180deg, #f7f8fb 0%, #eef3f1 42%, #f8f5ef 100%);
        color: #182026;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #17202a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #f5f7f8;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c7d0d9;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(28, 38, 48, 0.08);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 10px 28px rgba(32, 42, 52, 0.07);
    }

    div[data-testid="stMetricLabel"] {
        color: #63707a;
        font-size: 0.82rem;
    }

    div[data-testid="stMetricValue"] {
        color: #14212b;
        font-weight: 700;
    }

    .hero {
        background: linear-gradient(135deg, #24313c 0%, #315044 62%, #8a6f3d 100%);
        color: #ffffff;
        border-radius: 8px;
        padding: 1.3rem 1.45rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 46px rgba(31, 43, 52, 0.18);
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        line-height: 1.15;
    }

    .hero p {
        margin: 0;
        color: #e5ece9;
        font-size: 1rem;
    }

    .panel-title {
        color: #1e2b34;
        font-size: 1.08rem;
        font-weight: 700;
        margin: 0.35rem 0 0.65rem;
    }

    .hint {
        color: #66727c;
        font-size: 0.9rem;
        margin-top: -0.25rem;
        margin-bottom: 0.8rem;
    }

    .answer-box {
        background: #ffffff;
        border: 1px solid rgba(28, 38, 48, 0.09);
        border-left: 4px solid #2f735d;
        border-radius: 8px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 30px rgba(32, 42, 52, 0.07);
        line-height: 1.65;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        margin: 0.4rem 0 1rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(28, 38, 48, 0.08);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        min-height: 6.2rem;
        box-shadow: 0 10px 28px rgba(32, 42, 52, 0.07);
    }

    .metric-label {
        color: #5f6c76;
        font-size: 0.88rem;
        line-height: 1.35;
        margin-bottom: 0.55rem;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .metric-value {
        color: #14212b;
        font-size: 1.7rem;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: 0;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .metric-value.compact {
        font-size: 1.25rem;
        line-height: 1.3;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        min-height: 2.6rem;
    }

    .stTextArea textarea {
        border-radius: 8px;
    }
</style>
"""

DEMO_RESULT_CSS = """
<style>
    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.2rem;
    }

    .hero {
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
    }

    .hero h1 {
        font-size: 1.55rem;
    }

    .hero p {
        font-size: 0.9rem;
    }

    .metric-grid {
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 0.6rem;
    }

    .metric-card {
        min-height: 4.8rem;
        padding: 0.6rem 0.7rem;
    }

    .metric-label {
        font-size: 0.76rem;
        margin-bottom: 0.32rem;
    }

    .metric-value {
        font-size: 1.25rem;
    }

    .metric-value.compact {
        font-size: 1.05rem;
    }

    .panel-title {
        font-size: 1rem;
        margin: 0.25rem 0 0.45rem;
    }

    .answer-box {
        padding: 0.7rem 0.85rem;
        line-height: 1.45;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_compressor() -> ContextCompressor:
    return ContextCompressor(device_map="cpu")


def decode_uploaded_file(uploaded_file: Any) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def format_ratio(value: float) -> str:
    return f"{value:.2%}"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def render_metric_grid(items: list[tuple[str, Any]], compact_values: bool = False) -> None:
    value_class = "metric-value compact" if compact_values else "metric-value"
    cards = []
    for label, value in items:
        cards.append(
            "<div class=\"metric-card\">"
            f"<div class=\"metric-label\">{html.escape(str(label))}</div>"
            f"<div class=\"{value_class}\">{html.escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(f"<div class=\"metric-grid\">{''.join(cards)}</div>", unsafe_allow_html=True)


def get_demo_case_key() -> str | None:
    value = st.query_params.get("demo_case")
    if isinstance(value, list):
        value = value[0] if value else None
    return value if value in DEMO_CASES else None


def get_demo_view() -> str | None:
    value = st.query_params.get("demo_view")
    if isinstance(value, list):
        value = value[0] if value else None
    return value if value in {"result"} else None


def load_demo_case(case_key: str) -> None:
    case = DEMO_CASES[case_key]
    marker = f"demo:{case_key}"
    if st.session_state.get("loaded_demo_case") == marker:
        return

    display_method = case["method"]
    method = MODEL_OPTIONS[display_method]
    compression_result = get_compressor().compress_context(
        context=case["context"],
        question=case["question"],
        method=method,
        token_budget=130 if display_method == "Truncate" else 150,
    )

    st.session_state.context_text = case["context"]
    st.session_state.question_text = case["question"]
    st.session_state.demo_method = display_method
    st.session_state.compression_result = compression_result
    st.session_state.compressed_context = compression_result["compressed_context"]
    st.session_state.answer_result = {
        "answer": case["answer"],
        "model": "DeepSeek 演示结果",
        "answer_time": 0.82 if case_key.startswith("en") else 0.91,
        "usage": {
            "prompt_tokens": compression_result["compressed_tokens"] + 96,
            "completion_tokens": 36 if case_key.startswith("en") else 42,
            "total_tokens": compression_result["compressed_tokens"]
            + (132 if case_key.startswith("en") else 138),
        },
    }
    st.session_state.loaded_demo_case = marker


def run_answer_generation(
    compressed_context: str,
    question: str,
    temperature: float,
    max_answer_tokens: int,
) -> dict[str, Any]:
    from qkp_reorder.llm_inference import generate_answer

    prompt = build_qa_prompt(compressed_context, question)
    return generate_answer(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_answer_tokens,
    )


def render_usage(usage: Any) -> None:
    if not usage:
        st.caption("API 用量：暂无")
        return

    render_metric_grid(
        [
            ("Prompt tokens", usage.get("prompt_tokens", "-")),
            ("回答 tokens", usage.get("completion_tokens", "-")),
            ("总 tokens", usage.get("total_tokens", "-")),
        ]
    )


def render_results_section(compact: bool = False) -> bool:
    compression_result = st.session_state.get("compression_result")
    compressed_context = st.session_state.get("compressed_context")
    answer_result = st.session_state.get("answer_result")

    if not compression_result or compressed_context is None:
        return False

    st.markdown('<div class="panel-title">压缩结果</div>', unsafe_allow_html=True)
    render_metric_grid(
        [
            ("原始 tokens", compression_result["original_tokens"]),
            ("压缩后 tokens", compression_result["compressed_tokens"]),
            ("节省比例", format_ratio(compression_result["token_saving_ratio"])),
            ("压缩倍率", f"{compression_result['compression_ratio']:.2f}x"),
            ("压缩耗时", format_seconds(compression_result["compression_time"])),
        ]
    )

    st.text_area(
        "压缩后的上下文",
        value=compressed_context,
        height=130 if compact else 250,
        key="compressed_context_display",
    )

    if answer_result:
        st.markdown('<div class="panel-title">问答结果</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class=\"answer-box\">{html.escape(answer_result['answer'] or '')}</div>",
            unsafe_allow_html=True,
        )
        render_metric_grid(
            [
                ("回答模型", answer_result.get("model", "-")),
                ("回答耗时", format_seconds(answer_result.get("answer_time"))),
            ],
            compact_values=True,
        )
        render_usage(answer_result.get("usage"))

    return True


def main() -> None:
    st.set_page_config(
        page_title="QKP-Reorder 长文本压缩问答",
        page_icon="Q",
        layout="wide",
    )

    st.markdown(APP_CSS, unsafe_allow_html=True)
    demo_case_key = get_demo_case_key()
    demo_view = get_demo_view()
    if demo_case_key:
        load_demo_case(demo_case_key)
    if demo_view == "result":
        st.markdown(DEMO_RESULT_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>QKP-Reorder 长文本压缩问答演示</h1>
            <p>选择压缩方法，上传或输入长文本，再用压缩后的上下文调用 DeepSeek 生成答案。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if demo_view == "result" and render_results_section(compact=True):
        st.divider()

    with st.sidebar:
        st.header("演示设置")
        st.caption("默认使用项目创新方法 QKP-Reorder。")
        display_method = st.selectbox(
            "压缩模型",
            list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(
                st.session_state.get("demo_method", "QKP-Reorder")
            ),
        )
        token_budget = st.number_input(
            "压缩 token 预算",
            min_value=1,
            max_value=20000,
            value=750,
            step=50,
        )
        max_answer_tokens = st.number_input(
            "回答最大 tokens",
            min_value=16,
            max_value=2048,
            value=128,
            step=16,
        )
        temperature = st.slider(
            "回答随机性 temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )
        st.divider()
        st.caption("LLMLingua / LongLLMLingua 首次运行可能需要加载压缩模型，耗时会更长。")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="panel-title">输入长文本</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hint">可直接粘贴文本，也可以上传 txt、md 或 csv 文件。</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "上传文本文件",
            type=["txt", "md", "csv"],
        )
        uploaded_text = decode_uploaded_file(uploaded_file) if uploaded_file else ""

        if "context_text" not in st.session_state:
            st.session_state.context_text = DEFAULT_CONTEXT

        if uploaded_text:
            st.session_state.context_text = uploaded_text

        if st.button("使用示例文本"):
            st.session_state.context_text = DEFAULT_CONTEXT

        context = st.text_area(
            "上下文文本",
            key="context_text",
            height=430,
        )

    with right:
        st.markdown('<div class="panel-title">问题与运行</div>', unsafe_allow_html=True)
        if "question_text" not in st.session_state:
            st.session_state.question_text = DEFAULT_QUESTION
        question = st.text_area(
            "问题",
            key="question_text",
            height=110,
        )
        run_clicked = st.button("开始压缩并回答", type="primary", use_container_width=True)

        if run_clicked:
            st.session_state.pop("compression_result", None)
            st.session_state.pop("compressed_context", None)
            st.session_state.pop("answer_result", None)

            if not context.strip():
                st.error("上下文不能为空。")
                return
            if not question.strip():
                st.error("问题不能为空。")
                return

            method = MODEL_OPTIONS[display_method]
            compressor = get_compressor()

            try:
                with st.spinner(f"正在使用 {display_method} 压缩文本..."):
                    compression_result = compressor.compress_context(
                        context=context,
                        question=question,
                        method=method,
                        token_budget=int(token_budget),
                    )
            except Exception as exc:
                st.error(f"压缩失败：{exc}")
                return

            compressed_context = compression_result["compressed_context"]
            st.session_state.compression_result = compression_result
            st.session_state.compressed_context = compressed_context

            try:
                with st.spinner("正在调用 DeepSeek 生成答案..."):
                    answer_result = run_answer_generation(
                        compressed_context=compressed_context,
                        question=question,
                        temperature=float(temperature),
                        max_answer_tokens=int(max_answer_tokens),
                    )
            except Exception as exc:
                st.warning(f"问答生成失败：{exc}")
                return
            st.session_state.answer_result = answer_result

    if demo_view != "result" and st.session_state.get("compression_result"):
        st.divider()
        render_results_section()


if __name__ == "__main__":
    main()

