from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_nq_100_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "innovation"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_bm25_only",
    "ours_keyword",
    "ours_full",
]


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    df["method"] = pd.Categorical(df["method"], categories=present, ordered=True)
    return df.sort_values("method")


def save_bar(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_labels = df["method"].astype(str)
    ax.bar(x_labels, df[y_col])
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

    for index, value in enumerate(df[y_col]):
        if pd.notna(value):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    df = load_summary()

    f1_col = "f1_avg" if "f1_avg" in df.columns else "avg_f1"
    save_bar(
        df=df,
        y_col=f1_col,
        ylabel="Average F1",
        title="NQ Multi-document QA: Average Simple Token F1",
        filename="fig_ours_nq_100_f1.png",
    )

    tk_col = "compressed_tokens_avg" if "compressed_tokens_avg" in df.columns else "avg_compressed_tokens"
    save_bar(
        df=df,
        y_col=tk_col,
        ylabel="Average Compressed Tokens",
        title="NQ Multi-document QA: Average Context Tokens",
        filename="fig_ours_nq_100_tokens.png",
    )

    ts_col = "token_saving_ratio_avg" if "token_saving_ratio_avg" in df.columns else "avg_token_saving_ratio"
    save_bar(
        df=df,
        y_col=ts_col,
        ylabel="Average Token Saving Ratio",
        title="NQ Multi-document QA: Token Saving Ratio",
        filename="fig_ours_nq_100_token_saving.png",
    )

    if "nq_style_f1" in df.columns:
        save_bar(
            df=df,
            y_col="nq_style_f1",
            ylabel="NQ-style F1",
            title="NQ Multi-document QA: NQ-style Normalized F1",
            filename="fig_ours_nq_100_nq_style_f1.png",
        )

    if "contains_answer_avg" in df.columns:
        save_bar(
            df=df,
            y_col="contains_answer_avg",
            ylabel="Contains Answer Rate",
            title="NQ Multi-document QA: Contains Answer Rate",
            filename="fig_ours_nq_100_contains_answer.png",
        )

    print("\nAll figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()

