#!/usr/bin/env python3
"""Plot descriptive means of all 18 frozen-checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.diagnostics / "summary.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(args.diagnostics.glob("fold_*_seed_*.json"))]
    if len(runs) != 18 or not summary["complete_18_runs"]:
        raise ValueError("Requires all 18 formal frozen-checkpoint diagnostics.")
    colors = {"old": "#66788a", "new": "#d66c36"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for variant in ("old", "new"):
        selected = [r for r in runs if r["variant"] == variant]
        for mode, style in (("raw", "-"), ("teacher", "--")):
            curve = np.mean(
                [
                    [h["proposal_feature_mse"] for h in r["interventions"]["full"][mode]["horizons"]]
                    for r in selected
                ],
                axis=0,
            )
            axes[0].plot(
                range(1, 16),
                curve,
                style,
                color=colors[variant],
                label=f"{variant}: {'recursive' if mode == 'raw' else 'true history'}",
            )
    axes[0].axvline(7.5, color="#b4b8bd", linewidth=1)
    axes[0].text(7.8, 0.48, "input > 8 frames", fontsize=8, color="#59636c")
    axes[0].set(
        xlabel="Prediction horizon (starts after 2 true frames)",
        ylabel="Feature MSE",
        title="Recursive error grows despite better one-step fit",
    )
    axes[0].legend(fontsize=8, loc="upper left")
    metric = summary["overall"]["full/raw"]["mean_15"]
    parts = [
        ("bias_mse", "True-history residual", "#b5c3d1"),
        ("feedback_shift_mse", "Feedback shift", "#506e88"),
        ("bias_feedback_cross_term", "Cross term", "#df936d"),
    ]
    bottoms = np.zeros(2)
    for key, label, color in parts:
        values = np.array([metric[key][v] for v in ("old", "new")])
        axes[1].bar([0, 1], values, bottom=bottoms, color=color, label=label, width=0.55)
        bottoms += values
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Old", "Corrected"],
        ylabel="Mean 15-step feature MSE",
        title="Exact feature-error decomposition",
    )
    axes[1].legend(fontsize=8, loc="upper left")
    for index, total in enumerate(bottoms):
        axes[1].text(index, total + 0.003, f"{total:.5f}", ha="center", fontsize=9)
    diag_key = "physics_kernel_diagonal_mse"
    cross_key = "physics_kernel_offdiagonal_mse"
    # These keys encode an exact quadratic decomposition, not variance estimates.
    diagonal = np.array([metric[diag_key][v] for v in ("old", "new")])
    cross = np.array([metric[cross_key][v] for v in ("old", "new")])
    axes[2].bar([0, 1], diagonal, color="#8eb2a6", width=0.55, label="Weighted defect squares")
    axes[2].bar([0, 1], cross, bottom=diagonal, color="#c87545", width=0.55, label="Cross-time products")
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Old", "Corrected"],
        ylabel="Mean 15-step position MSE",
        title="More error alignment outweighs smaller defects",
    )
    axes[2].legend(fontsize=8, loc="upper left")
    for index, total in enumerate(diagonal + cross):
        axes[2].text(index, total + 0.002, f"{total:.5f}", ha="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[1].set_ylim(top=0.29)
    axes[2].set_ylim(top=0.155)
    fig.suptitle(
        "LeRobot tiny predictor / same 18 CUDA-trained checkpoints / post-hoc CPU diagnostics", fontsize=13
    )
    fig.text(
        0.5,
        -0.04,
        "Descriptive paired means over 3 folds and 3 initializations; synthetic features; no new training or significance claim.",
        ha="center",
        fontsize=9,
        color="#59636c",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "rollout-diagnosis.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.out / "rollout-diagnosis.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
