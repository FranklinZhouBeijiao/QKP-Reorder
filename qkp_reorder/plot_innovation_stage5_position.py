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

SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_position_summary.csv"
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
POSITION_ORDER = [1, 5, 10, 15, 20]

COLORS: dict[str, str] = {
    "original": "#999999",
    "truncate": "#E69F00",
    "bm25": "#56B4E9",
    "llmlingua": "#009E73",
    "longllmlingua": "#F0E442",
    "ours_bm25_only": "#CC79A7",
    "ours_keyword": "#D55E00",
    "ours_full": "#0072B2",
}


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    df["answer_position"] = pd.Categorical(
        df["answer_position"], categories=POSITION_ORDER, ordered=True
    )
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    df["method"] = pd.Categorical(df["method"], categories=present, ordered=True)
    return df.sort_values(["answer_position", "method"])


def plot_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
    highlight_ours: bool = True,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    for method in methods_present:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        is_ours = method.startswith("ours")
        linewidth = 2.5 if (highlight_ours and is_ours) else 1.2
        linestyle = "-" if is_ours else "--"
        alpha = 0.95 if (highlight_ours and is_ours) else 0.6
        marker = "D" if is_ours else "o"
        markersize = 6 if (highlight_ours and is_ours) else 4

        color = COLORS.get(method, None)

        ax.plot(
            sub["answer_position"].astype(int),
            sub[metric_col],
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=method,
            color=color,
        )

    ax.set_xlabel("Answer Document Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(POSITION_ORDER)
    ax.legend(fontsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Add vertical line at position 10 to mark "middle" region
    ax.axvline(x=10, color="red", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    df = load_summary()

    plot_metric(
        df=df,
        metric_col="contains_answer_avg",
        ylabel="Contains Answer Rate",
        title="Position Sensitivity: Contains Answer Rate\n(ours_full vs baselines)",
        filename="fig_ours_position_contains_answer.png",
    )

    plot_metric(
        df=df,
        metric_col="f1_avg",
        ylabel="Average F1",
        title="Position Sensitivity: Simple Token F1\n(ours_full vs baselines)",
        filename="fig_ours_position_f1.png",
    )

    if "nq_style_f1_avg" in df.columns:
        plot_metric(
            df=df,
            metric_col="nq_style_f1_avg",
            ylabel="NQ-style F1",
            title="Position Sensitivity: NQ-style Normalized F1\n(ours_full vs baselines)",
            filename="fig_ours_position_nq_style_f1.png",
        )

    if "compressed_tokens_avg" in df.columns:
        plot_metric(
            df=df,
            metric_col="compressed_tokens_avg",
            ylabel="Average Compressed Tokens",
            title="Position Sensitivity: Compressed Tokens",
            filename="fig_ours_position_tokens.png",
            highlight_ours=False,
        )

    if "token_saving_ratio_avg" in df.columns:
        plot_metric(
            df=df,
            metric_col="token_saving_ratio_avg",
            ylabel="Average Token Saving Ratio",
            title="Position Sensitivity: Token Saving Ratio",
            filename="fig_ours_position_token_saving.png",
            highlight_ours=False,
        )

    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

