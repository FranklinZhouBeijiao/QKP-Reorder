from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_ablation_summary_by_method.csv"
POS_SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_ablation_summary_by_position.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "innovation"

METHOD_ORDER = ["ours_bm25_only", "ours_keyword", "ours_full"]
POSITION_ORDER = [1, 5, 10, 15, 20]

COLORS: dict[str, str] = {
    "ours_bm25_only": "#CC79A7",
    "ours_keyword": "#D55E00",
    "ours_full": "#0072B2",
}
METHOD_LABELS: dict[str, str] = {
    "ours_bm25_only": "BM25 Only",
    "ours_keyword": "+ Keyword",
    "ours_full": "+ Middle-Aware",
}


def load_method_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    df["method"] = pd.Categorical(df["method"], categories=present, ordered=True)
    return df.sort_values("method")


def load_position_summary() -> pd.DataFrame:
    if not POS_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Position summary file not found: {POS_SUMMARY_PATH}")
    df = pd.read_csv(POS_SUMMARY_PATH)
    df["answer_position"] = pd.Categorical(
        df["answer_position"], categories=POSITION_ORDER, ordered=True
    )
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    df["method"] = pd.Categorical(df["method"], categories=present, ordered=True)
    return df.sort_values(["answer_position", "method"])


def plot_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = [m for m in METHOD_ORDER if m in df["method"].values]
    x = np.arange(len(methods))
    values = []
    for m in methods:
        v = df[df["method"] == m][metric_col].values
        values.append(v[0] if len(v) > 0 else 0)

    bars = ax.bar(
        x, values,
        color=[COLORS.get(m, "#999999") for m in methods],
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_position_comparison(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    for method in methods_present:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        color = COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)

        ax.plot(
            sub["answer_position"].astype(int),
            sub[metric_col],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=label,
            color=color,
        )

    ax.set_xlabel("Answer Document Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(POSITION_ORDER)
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axvline(x=10, color="red", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_side_by_side_bars(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Grouped bar chart comparing methods at high positions (10, 15, 20)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    target_positions = [10, 15, 20]
    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    x = np.arange(len(target_positions))
    width = 0.25
    n_methods = len(methods_present)

    for i, method in enumerate(methods_present):
        sub = df[(df["method"] == method) & (df["answer_position"].isin(target_positions))]
        values = []
        for pos in target_positions:
            v = sub[sub["answer_position"] == pos][metric_col].values
            values.append(v[0] if len(v) > 0 else 0)

        offset = (i - (n_methods - 1) / 2) * width
        color = COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)

        bars = ax.bar(
            x + offset, values, width,
            color=color, label=label,
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Position {p}" for p in target_positions])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Overall bar charts ──
    method_df = load_method_summary()

    plot_bar_chart(
        df=method_df,
        metric_col="f1_avg",
        ylabel="Average F1",
        title="Ablation: Simple Token F1 by Method\n(Token Budget = 1500, Position Dataset)",
        filename="fig_ours_ablation_f1.png",
    )

    plot_bar_chart(
        df=method_df,
        metric_col="contains_answer_avg",
        ylabel="Contains Answer Rate",
        title="Ablation: Contains Answer Rate by Method\n(Token Budget = 1500, Position Dataset)",
        filename="fig_ours_ablation_contains_answer.png",
    )

    plot_bar_chart(
        df=method_df,
        metric_col="nq_style_f1_avg",
        ylabel="NQ-style F1",
        title="Ablation: NQ-style Normalized F1 by Method\n(Token Budget = 1500, Position Dataset)",
        filename="fig_ours_ablation_nq_style_f1.png",
    )

    plot_bar_chart(
        df=method_df,
        metric_col="token_saving_ratio_avg",
        ylabel="Token Saving Ratio",
        title="Ablation: Token Saving Ratio by Method",
        filename="fig_ours_ablation_token_saving.png",
    )

    plot_bar_chart(
        df=method_df,
        metric_col="answer_time_avg",
        ylabel="Avg Answer Time (s)",
        title="Ablation: Average Answer Time by Method",
        filename="fig_ours_ablation_answer_time.png",
    )

    # ── Position comparison charts ──
    pos_df = load_position_summary()

    plot_position_comparison(
        df=pos_df,
        metric_col="f1_avg",
        ylabel="Average F1",
        title="Ablation × Position: Simple Token F1",
        filename="fig_ours_ablation_position_f1.png",
    )

    plot_position_comparison(
        df=pos_df,
        metric_col="contains_answer_avg",
        ylabel="Contains Answer Rate",
        title="Ablation × Position: Contains Answer Rate",
        filename="fig_ours_ablation_position_contains_answer.png",
    )

    plot_position_comparison(
        df=pos_df,
        metric_col="nq_style_f1_avg",
        ylabel="NQ-style F1",
        title="Ablation × Position: NQ-style Normalized F1",
        filename="fig_ours_ablation_position_nq_style_f1.png",
    )

    # ── Side-by-side for high positions ──
    plot_side_by_side_bars(
        df=pos_df,
        metric_col="f1_avg",
        ylabel="Average F1",
        title="Ablation: F1 at Deep Positions (10/15/20)",
        filename="fig_ours_ablation_deep_position_f1.png",
    )

    plot_side_by_side_bars(
        df=pos_df,
        metric_col="contains_answer_avg",
        ylabel="Contains Answer Rate",
        title="Ablation: Contains Answer at Deep Positions (10/15/20)",
        filename="fig_ours_ablation_deep_position_contains_answer.png",
    )

    print(f"\nAll ablation figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

