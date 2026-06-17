from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "results" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"

EXP1_SUMMARY = FINAL_DIR / "nq_style_eval_exp1_summary.csv"
EXP2_SUMMARY = FINAL_DIR / "nq_style_eval_exp2_summary.csv"

METHOD_ORDER = ["original", "truncate", "bm25", "llmlingua", "longllmlingua"]
POSITION_ORDER = [1, 5, 10, 15, 20]


def load_exp1() -> pd.DataFrame:
    df = pd.read_csv(EXP1_SUMMARY)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("method")


def load_exp2() -> pd.DataFrame:
    df = pd.read_csv(EXP2_SUMMARY)
    df["answer_position"] = pd.Categorical(df["answer_position"], categories=POSITION_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["answer_position", "method"])


def save_bar(df: pd.DataFrame, y_col: str, ylabel: str, title: str, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["method"].astype(str), df[y_col])
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

    for idx, value in enumerate(df[y_col]):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figure: {out_path}")


def save_position_curve(df: pd.DataFrame, y_col: str, ylabel: str, title: str, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        ax.plot(
            sub["answer_position"].astype(int),
            sub[y_col],
            marker="o",
            label=method,
        )

    ax.set_xlabel("Answer Document Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(POSITION_ORDER)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figure: {out_path}")


def main() -> None:
    exp1 = load_exp1()
    exp2 = load_exp2()

    save_bar(
        df=exp1,
        y_col="nq_style_em",
        ylabel="NQ-style EM",
        title="NQ 100: Normalized Exact Match",
        filename="fig_nq_style_exp1_em.png",
    )

    save_bar(
        df=exp1,
        y_col="nq_style_f1",
        ylabel="NQ-style F1",
        title="NQ 100: Normalized Token F1",
        filename="fig_nq_style_exp1_f1.png",
    )

    save_position_curve(
        df=exp2,
        y_col="nq_style_em",
        ylabel="NQ-style EM",
        title="Position Sensitivity: Normalized Exact Match",
        filename="fig_nq_style_exp2_position_em.png",
    )

    save_position_curve(
        df=exp2,
        y_col="nq_style_f1",
        ylabel="NQ-style F1",
        title="Position Sensitivity: Normalized Token F1",
        filename="fig_nq_style_exp2_position_f1.png",
    )


if __name__ == "__main__":
    main()

