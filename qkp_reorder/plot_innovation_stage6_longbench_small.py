from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "innovation"
    / "ours_longbench_official_eval_comparison.csv"
)
SIMPLIFIED_COMPARISON_CSV = (
    PROJECT_ROOT / "results" / "innovation" / "ours_longbench_small_comparison.csv"
)
FIG_DIR = PROJECT_ROOT / "figures" / "innovation"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_full",
]
TASK_NAMES = ["hotpotqa", "2wikimqa", "multifieldqa_en"]
TASK_LABELS = ["HotpotQA", "2WikiMQA", "MultiFieldQA-en"]

COLORS = {
    "original": "#7f7f7f",
    "truncate": "#d62728",
    "bm25": "#1f77b4",
    "llmlingua": "#ff7f0e",
    "longllmlingua": "#2ca02c",
    "ours_full": "#9467bd",
}


def setup_figure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def plot_official_score(df: pd.DataFrame) -> None:
    if "official_score" not in df.columns or df["official_score"].isna().all():
        print("[SKIP] No official_score data for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, task, label in zip(axes, TASK_NAMES, TASK_LABELS):
        task_df = df[df["task_name"] == task].copy()
        if task_df.empty:
            ax.set_title(label)
            continue

        methods_present = [m for m in METHOD_ORDER if m in task_df["method"].values]
        scores = []
        bar_colors = []
        for m in methods_present:
            row = task_df[task_df["method"] == m]
            if row.empty:
                continue
            val = row["official_score"].values[0]
            scores.append(val if not pd.isna(val) else 0)
            bar_colors.append(COLORS.get(m, "#999999"))

        x = np.arange(len(methods_present))
        bars = ax.bar(x, scores, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Official Score")
        ax.set_ylim(0, max(scores) * 1.25 if scores else 80)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.suptitle("LongBench Official Scores by Task", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_DIR / "fig_ours_longbench_official_score.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def plot_f1(df: pd.DataFrame) -> None:
    if "avg_f1" not in df.columns or df["avg_f1"].isna().all():
        print("[SKIP] No avg_f1 data for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, task, label in zip(axes, TASK_NAMES, TASK_LABELS):
        task_df = df[df["task_name"] == task].copy()
        if task_df.empty:
            ax.set_title(label)
            continue

        methods_present = [m for m in METHOD_ORDER if m in task_df["method"].values]
        scores = []
        bar_colors = []
        for m in methods_present:
            row = task_df[task_df["method"] == m]
            if row.empty:
                continue
            val = row["avg_f1"].values[0]
            scores.append(val if not pd.isna(val) else 0)
            bar_colors.append(COLORS.get(m, "#999999"))

        x = np.arange(len(methods_present))
        bars = ax.bar(x, scores, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg Simple Token F1")
        ax.set_ylim(0, max(scores) * 1.25 if scores else 1.0)

    fig.suptitle("Simple Token F1 by Task", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_DIR / "fig_ours_longbench_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def plot_tokens(df: pd.DataFrame) -> None:
    if "avg_compressed_tokens" not in df.columns:
        print("[SKIP] No avg_compressed_tokens data for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, task, label in zip(axes, TASK_NAMES, TASK_LABELS):
        task_df = df[df["task_name"] == task].copy()
        if task_df.empty:
            ax.set_title(label)
            continue

        methods_present = [m for m in METHOD_ORDER if m in task_df["method"].values]
        tokens = []
        bar_colors = []
        for m in methods_present:
            row = task_df[task_df["method"] == m]
            if row.empty:
                continue
            val = row["avg_compressed_tokens"].values[0]
            tokens.append(val if not pd.isna(val) else 0)
            bar_colors.append(COLORS.get(m, "#999999"))

        x = np.arange(len(methods_present))
        bars = ax.bar(x, tokens, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, tokens):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg Compressed Tokens")
        if tokens:
            ax.set_ylim(0, max(tokens) * 1.2)

    fig.suptitle("Average Compressed Tokens by Task", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_DIR / "fig_ours_longbench_tokens.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def plot_token_saving(df: pd.DataFrame) -> None:
    if "avg_token_saving_ratio" not in df.columns:
        print("[SKIP] No avg_token_saving_ratio data for plotting.")
        return

    # Only compress methods (exclude original)
    compress_methods = [m for m in METHOD_ORDER if m != "original"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, task, label in zip(axes, TASK_NAMES, TASK_LABELS):
        task_df = df[df["task_name"] == task].copy()
        if task_df.empty:
            ax.set_title(label)
            continue

        methods_present = [m for m in compress_methods if m in task_df["method"].values]
        ratios = []
        bar_colors = []
        for m in methods_present:
            row = task_df[task_df["method"] == m]
            if row.empty:
                continue
            val = row["avg_token_saving_ratio"].values[0]
            ratios.append(val if not pd.isna(val) else 0)
            bar_colors.append(COLORS.get(m, "#999999"))

        x = np.arange(len(methods_present))
        bars = ax.bar(x, ratios, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg Token Saving Ratio")
        ax.set_ylim(0, 1.1)

    fig.suptitle("Token Saving Ratio by Task", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_DIR / "fig_ours_longbench_token_saving.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def plot_contains_answer(df: pd.DataFrame) -> None:
    if "contains_answer_rate" not in df.columns:
        print("[SKIP] No contains_answer_rate data for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, task, label in zip(axes, TASK_NAMES, TASK_LABELS):
        task_df = df[df["task_name"] == task].copy()
        if task_df.empty:
            ax.set_title(label)
            continue

        methods_present = [m for m in METHOD_ORDER if m in task_df["method"].values]
        rates = []
        bar_colors = []
        for m in methods_present:
            row = task_df[task_df["method"] == m]
            if row.empty:
                continue
            val = row["contains_answer_rate"].values[0]
            rates.append(val if not pd.isna(val) else 0)
            bar_colors.append(COLORS.get(m, "#999999"))

        x = np.arange(len(methods_present))
        bars = ax.bar(x, rates, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Contains Answer Rate")
        ax.set_ylim(0, 1.1)

    fig.suptitle("Contains Answer Rate by Task", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_DIR / "fig_ours_longbench_contains_answer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def main() -> None:
    setup_figure_dir()
    set_style()

    # Try official eval comparison first
    if COMPARISON_CSV.exists():
        df = pd.read_csv(COMPARISON_CSV)
        print(f"Loaded official comparison: {COMPARISON_CSV}")
    elif SIMPLIFIED_COMPARISON_CSV.exists():
        df = pd.read_csv(SIMPLIFIED_COMPARISON_CSV)
        print(f"Loaded simplified comparison: {SIMPLIFIED_COMPARISON_CSV}")
    else:
        print("[WARN] No comparison data available for plotting.")
        return

    plot_official_score(df)
    plot_f1(df)
    plot_tokens(df)
    plot_token_saving(df)
    plot_contains_answer(df)

    print("\nAll plots generated in:", FIG_DIR)


if __name__ == "__main__":
    main()

