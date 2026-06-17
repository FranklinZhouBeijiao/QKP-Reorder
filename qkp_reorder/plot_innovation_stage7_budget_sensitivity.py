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

SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_budget_sensitivity_summary.csv"
POS_SUMMARY_PATH = PROJECT_ROOT / "results" / "innovation" / "ours_budget_sensitivity_summary_by_position.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "innovation"

BUDGET_ORDER = [500, 750, 1000, 1500, 2000]
POSITION_ORDER = [1, 5, 10, 15, 20]

COLOR_QUALITY = "#0072B2"
COLOR_COST = "#D55E00"


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    present = [b for b in BUDGET_ORDER if b in df["token_budget"].values]
    df["token_budget"] = pd.Categorical(
        df["token_budget"], categories=present, ordered=True
    )
    return df.sort_values("token_budget")


def load_position_summary() -> pd.DataFrame:
    if not POS_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Position summary file not found: {POS_SUMMARY_PATH}")
    df = pd.read_csv(POS_SUMMARY_PATH)
    df["answer_position"] = pd.Categorical(
        df["answer_position"], categories=POSITION_ORDER, ordered=True
    )
    present = [b for b in BUDGET_ORDER if b in df["token_budget"].values]
    df["token_budget"] = pd.Categorical(
        df["token_budget"], categories=present, ordered=True
    )
    return df.sort_values(["answer_position", "token_budget"])


def plot_quality_cost_curve(df: pd.DataFrame) -> None:
    """Dual-axis plot: F1 (quality) and total_tokens (cost) vs budget."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    budgets = df["token_budget"].astype(int).values
    f1 = df["f1_avg"].values
    tokens = df["api_total_tokens_avg"].values

    ax1.plot(budgets, f1, "o-", color=COLOR_QUALITY, linewidth=2, markersize=8, label="Avg F1")
    ax2.plot(budgets, tokens, "s--", color=COLOR_COST, linewidth=2, markersize=8, label="Avg API Total Tokens")

    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Average F1", color=COLOR_QUALITY)
    ax2.set_ylabel("Avg API Total Tokens", color=COLOR_COST)
    ax1.tick_params(axis="y", labelcolor=COLOR_QUALITY)
    ax2.tick_params(axis="y", labelcolor=COLOR_COST)

    ax1.set_title("Quality–Cost Curve: ours_full at Different Token Budgets\n(Position Dataset)")
    ax1.set_xticks(budgets)
    ax1.grid(True, linestyle="--", alpha=0.35)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8.5)

    # Annotate F1 values
    for x, y in zip(budgets, f1):
        ax1.annotate(
            f"{y:.3f}", (x, y),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, color=COLOR_QUALITY,
        )

    fig.tight_layout()
    out_path = FIGURE_DIR / "fig_ours_budget_quality_cost_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_token_saving(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    budgets = df["token_budget"].astype(int).values
    token_saving = df["token_saving_ratio_avg"].values
    compressed = df["compressed_tokens_avg"].values

    ax.plot(budgets, token_saving, "o-", color=COLOR_QUALITY, linewidth=2, markersize=8)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Token Saving Ratio", color=COLOR_QUALITY)
    ax.tick_params(axis="y", labelcolor=COLOR_QUALITY)
    ax.set_title("Token Saving Ratio vs Budget\n(ours_full, Position Dataset)")

    # Annotate
    for x, y in zip(budgets, token_saving):
        ax.annotate(
            f"{y:.3f}", (x, y),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, color=COLOR_QUALITY,
        )

    ax.set_xticks(budgets)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path = FIGURE_DIR / "fig_ours_budget_token_saving.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_budget_position_heatmap(df: pd.DataFrame, metric_col: str, filename: str) -> None:
    """Position × Budget heatmap for a given metric."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    budgets_present = sorted(df["token_budget"].unique())
    positions_present = sorted(df["answer_position"].unique())

    pivot = df.pivot_table(
        index="answer_position", columns="token_budget", values=metric_col
    )
    pivot = pivot.reindex(index=positions_present, columns=budgets_present)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(budgets_present)))
    ax.set_xticklabels([str(int(b)) for b in budgets_present])
    ax.set_yticks(range(len(positions_present)))
    ax.set_yticklabels([str(int(p)) for p in positions_present])
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Answer Position")
    ax.set_title(f"Budget Sensitivity × Position: {metric_col}\n(ours_full)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_col)

    # Annotate cells
    for i in range(len(positions_present)):
        for j in range(len(budgets_present)):
            val = pivot.values[i, j]
            text_color = "white" if val is not None and val < 0.5 else "black"
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_budget_f1_by_position_line(df: pd.DataFrame) -> None:
    """Line chart showing F1 at each budget, grouped by answer position."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    positions_present = sorted(df["answer_position"].unique())

    for pos in positions_present:
        sub = df[df["answer_position"] == pos]
        budgets = sub["token_budget"].astype(int).values
        f1 = sub["f1_avg"].values

        alpha = 0.4 + 0.15 * (pos / max(positions_present))
        linestyle = "-" if pos <= 10 else "--"

        ax.plot(
            budgets, f1,
            marker="o", markersize=5, linewidth=1.5,
            alpha=alpha, linestyle=linestyle,
            label=f"Position {int(pos)}",
        )

    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Average F1")
    ax.set_title("Budget Sensitivity × Position: F1\n(ours_full)")
    ax.set_xticks(sorted(df["token_budget"].unique()))
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = FIGURE_DIR / "fig_ours_budget_position_f1.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()

    plot_quality_cost_curve(df)
    plot_token_saving(df)

    # Position-level analysis
    pos_df = load_position_summary()

    plot_budget_position_heatmap(
        pos_df, "f1_avg", "fig_ours_budget_position_f1_heatmap.png"
    )
    plot_budget_position_heatmap(
        pos_df, "contains_answer_avg", "fig_ours_budget_position_contains_answer_heatmap.png"
    )
    plot_budget_f1_by_position_line(pos_df)

    print(f"\nAll budget sensitivity figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

