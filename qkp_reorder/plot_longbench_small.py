from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results" / "stage7" / "longbench_small_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "stage7"

METHOD_ORDER = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
]

TASK_ORDER = [
    "hotpotqa",
    "2wikimqa",
    "multifieldqa_en",
]


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    df["task_name"] = pd.Categorical(
        df["task_name"],
        categories=TASK_ORDER,
        ordered=True,
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["task_name", "method"])


def plot_grouped_bar(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot(index="task_name", columns="method", values=metric_col)
    pivot = pivot.reindex(index=TASK_ORDER, columns=METHOD_ORDER)

    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Method", fontsize=8)
    ax.tick_params(axis="x", rotation=0)

    fig = ax.get_figure()
    fig.tight_layout()

    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    df = load_summary()

    plot_grouped_bar(
        df=df,
        metric_col="contains_answer_rate",
        ylabel="Contains Answer Rate",
        title="LongBench Small: Contains Answer Rate",
        filename="fig_longbench_small_contains_answer.png",
    )
    plot_grouped_bar(
        df=df,
        metric_col="avg_f1",
        ylabel="Average F1",
        title="LongBench Small: Average F1",
        filename="fig_longbench_small_f1.png",
    )
    plot_grouped_bar(
        df=df,
        metric_col="avg_compressed_tokens",
        ylabel="Average Compressed Tokens",
        title="LongBench Small: Average Tokens",
        filename="fig_longbench_small_tokens.png",
    )
    plot_grouped_bar(
        df=df,
        metric_col="avg_answer_time",
        ylabel="Average Answer Time (s)",
        title="LongBench Small: Average Answer Time",
        filename="fig_longbench_small_latency.png",
    )


if __name__ == "__main__":
    main()

