from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results" / "stage6" / "exp1_nq_100_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "stage6"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]


def load_summary() -> pd.DataFrame:
    """Load stage-6 method summary in a stable display order."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("method")


def save_bar(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    rotate: int = 20,
) -> None:
    """Save one bar chart from a summary metric."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    x_labels = df["method"].astype(str)
    ax.bar(x_labels, df[y_col])
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=rotate)

    for index, value in enumerate(df[y_col]):
        if pd.notna(value):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_latency_figure(df: pd.DataFrame) -> None:
    """Save grouped compression/answer latency chart."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [index - width / 2 for index in x],
        df["avg_compression_time"],
        width,
        label="Compression Time",
    )
    ax.bar(
        [index + width / 2 for index in x],
        df["avg_answer_time"],
        width,
        label="Answer Time",
    )

    ax.set_xlabel("Method")
    ax.set_ylabel("Average Time (s)")
    ax.set_title("Average Compression and Answer Time")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"].astype(str), rotation=20)
    ax.legend()

    fig.tight_layout()
    out_path = FIGURE_DIR / "fig_exp1_latency.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    df = load_summary()

    save_bar(
        df=df,
        y_col="contains_answer_rate",
        ylabel="Contains Answer Rate",
        title="NQ Multi-document QA: Contains Answer Rate",
        filename="fig_exp1_contains_answer.png",
    )
    save_bar(
        df=df,
        y_col="avg_f1",
        ylabel="Average F1",
        title="NQ Multi-document QA: Average F1",
        filename="fig_exp1_f1.png",
    )
    save_bar(
        df=df,
        y_col="avg_compressed_tokens",
        ylabel="Average Compressed Tokens",
        title="NQ Multi-document QA: Average Context Tokens",
        filename="fig_exp1_avg_tokens.png",
    )
    save_bar(
        df=df,
        y_col="avg_token_saving_ratio",
        ylabel="Average Token Saving Ratio",
        title="NQ Multi-document QA: Token Saving Ratio",
        filename="fig_exp1_token_saving.png",
    )
    save_latency_figure(df)


if __name__ == "__main__":
    main()

