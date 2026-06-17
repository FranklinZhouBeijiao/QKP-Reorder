from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_DIR = PROJECT_ROOT / "results" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"

OVERALL_PATH = FINAL_DIR / "cost_effectiveness_overall.csv"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]


def load_overall() -> pd.DataFrame:
    if not OVERALL_PATH.exists():
        raise FileNotFoundError(f"Missing cost-effectiveness file: {OVERALL_PATH}")

    df = pd.read_csv(OVERALL_PATH)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("method")


def save_bar(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    digits: int = 2,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    x_labels = df["method"].astype(str)
    ax.bar(x_labels, df[y_col])

    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

    for idx, value in enumerate(df[y_col]):
        if pd.notna(value):
            ax.text(
                idx,
                value,
                f"{value:.{digits}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figure: {out_path}")


def main() -> None:
    df = load_overall()

    save_bar(
        df=df,
        y_col="avg_api_total_tokens",
        ylabel="Average API Total Tokens",
        title="Cost Proxy: Average API Total Tokens",
        filename="fig_cost_avg_total_tokens.png",
        digits=0,
    )
    save_bar(
        df=df,
        y_col="avg_f1",
        ylabel="Average F1",
        title="Effectiveness: Average F1",
        filename="fig_cost_avg_f1.png",
        digits=3,
    )
    save_bar(
        df=df,
        y_col="f1_per_1k_tokens",
        ylabel="F1 per 1k Tokens",
        title="Cost-Effectiveness: F1 per 1k Tokens",
        filename="fig_cost_f1_per_1k_tokens.png",
        digits=3,
    )
    save_bar(
        df=df,
        y_col="tokens_per_correct_answer",
        ylabel="Tokens per Correct Answer",
        title="Cost-Effectiveness: Tokens per Correct Answer",
        filename="fig_cost_tokens_per_correct_answer.png",
        digits=0,
    )


if __name__ == "__main__":
    main()

