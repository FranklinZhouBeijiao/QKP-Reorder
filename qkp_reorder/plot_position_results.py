from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results" / "stage6_position" / "exp2_position_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "stage6_position"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]
POSITION_ORDER = [1, 5, 10, 15, 20]


def load_summary() -> pd.DataFrame:
    """Load position summary with stable position/method ordering."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    df["answer_position"] = pd.Categorical(
        df["answer_position"],
        categories=POSITION_ORDER,
        ordered=True,
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["answer_position", "method"])


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot one position-sensitivity metric as method-wise lines."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        ax.plot(
            sub["answer_position"].astype(int),
            sub[metric_col],
            marker="o",
            label=method,
        )

    ax.set_xlabel("Answer Document Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(POSITION_ORDER)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    df = load_summary()

    plot_metric(
        df=df,
        metric_col="contains_answer_rate",
        ylabel="Contains Answer Rate",
        title="Position Sensitivity: Contains Answer Rate",
        filename="fig_exp2_position_contains_answer.png",
    )
    plot_metric(
        df=df,
        metric_col="avg_f1",
        ylabel="Average F1",
        title="Position Sensitivity: Average F1",
        filename="fig_exp2_position_f1.png",
    )
    plot_metric(
        df=df,
        metric_col="avg_compressed_tokens",
        ylabel="Average Compressed Tokens",
        title="Position Sensitivity: Average Tokens",
        filename="fig_exp2_position_tokens.png",
    )
    plot_metric(
        df=df,
        metric_col="avg_answer_time",
        ylabel="Average Answer Time (s)",
        title="Position Sensitivity: Answer Time",
        filename="fig_exp2_position_latency.png",
    )


if __name__ == "__main__":
    main()

