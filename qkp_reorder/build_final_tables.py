from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FINAL_DIR = PROJECT_ROOT / "results" / "final"
FIGURE_FINAL_DIR = PROJECT_ROOT / "figures" / "final"

SUMMARY_FILES = {
    "final_exp1_nq_summary.csv": PROJECT_ROOT
    / "results"
    / "stage6"
    / "exp1_nq_100_summary.csv",
    "final_exp2_position_summary.csv": PROJECT_ROOT
    / "results"
    / "stage6_position"
    / "exp2_position_summary.csv",
    "final_exp3_longbench_summary.csv": PROJECT_ROOT
    / "results"
    / "stage7"
    / "longbench_small_summary.csv",
}

FIGURE_FILES = {
    "exp1_fig_exp1_contains_answer.png": PROJECT_ROOT
    / "figures"
    / "stage6"
    / "fig_exp1_contains_answer.png",
    "exp1_fig_exp1_f1.png": PROJECT_ROOT
    / "figures"
    / "stage6"
    / "fig_exp1_f1.png",
    "exp1_fig_exp1_avg_tokens.png": PROJECT_ROOT
    / "figures"
    / "stage6"
    / "fig_exp1_avg_tokens.png",
    "exp1_fig_exp1_token_saving.png": PROJECT_ROOT
    / "figures"
    / "stage6"
    / "fig_exp1_token_saving.png",
    "exp2_fig_exp2_position_contains_answer.png": PROJECT_ROOT
    / "figures"
    / "stage6_position"
    / "fig_exp2_position_contains_answer.png",
    "exp2_fig_exp2_position_f1.png": PROJECT_ROOT
    / "figures"
    / "stage6_position"
    / "fig_exp2_position_f1.png",
    "exp3_fig_longbench_small_contains_answer.png": PROJECT_ROOT
    / "figures"
    / "stage7"
    / "fig_longbench_small_contains_answer.png",
    "exp3_fig_longbench_small_f1.png": PROJECT_ROOT
    / "figures"
    / "stage7"
    / "fig_longbench_small_f1.png",
    "exp3_fig_longbench_small_tokens.png": PROJECT_ROOT
    / "figures"
    / "stage7"
    / "fig_longbench_small_tokens.png",
    "exp3_fig_longbench_small_latency.png": PROJECT_ROOT
    / "figures"
    / "stage7"
    / "fig_longbench_small_latency.png",
}


def copy_summary_files() -> None:
    """Copy final experiment summary tables into results/final."""
    RESULT_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, src_path in SUMMARY_FILES.items():
        if not src_path.exists():
            print(f"[WARN] Missing summary file: {src_path}")
            continue

        df = pd.read_csv(src_path)
        out_path = RESULT_FINAL_DIR / out_name
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] Copied summary: {src_path} -> {out_path}")


def copy_figure_files() -> None:
    """Copy representative report figures into figures/final."""
    FIGURE_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, src_path in FIGURE_FILES.items():
        if not src_path.exists():
            print(f"[WARN] Missing figure file: {src_path}")
            continue

        out_path = FIGURE_FINAL_DIR / out_name
        shutil.copy2(src_path, out_path)
        print(f"[OK] Copied figure: {src_path} -> {out_path}")


def build_artifact_index() -> None:
    """Write an index for final summary tables and copied figures."""
    index_path = RESULT_FINAL_DIR / "final_artifact_index.md"

    lines = [
        "# Final Artifact Index",
        "",
        "This file lists the final summarized artifacts for the reproduction project.",
        "",
        "## Final Summary Tables",
        "",
    ]
    lines.extend(f"- `results/final/{out_name}`" for out_name in SUMMARY_FILES)
    lines.extend(["", "## Final Figures", ""])
    lines.extend(f"- `figures/final/{out_name}`" for out_name in FIGURE_FILES)
    lines.extend(["", "## Source Summary Files", ""])

    for src_path in SUMMARY_FILES.values():
        lines.append(f"- `{src_path.relative_to(PROJECT_ROOT)}`")

    lines.extend(
        [
            "",
            "## Source Figure Directories",
            "",
            "- `figures/stage6/`",
            "- `figures/stage6_position/`",
            "- `figures/stage7/`",
            "",
        ]
    )

    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote artifact index: {index_path}")


def main() -> None:
    copy_summary_files()
    copy_figure_files()
    build_artifact_index()


if __name__ == "__main__":
    main()

