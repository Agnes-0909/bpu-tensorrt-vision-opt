#!/usr/bin/env python3
"""Generate the root README overview chart for the optimization timeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures" / "optimization_stage_timing.png"

# Values are the published mean timings from the corresponding stage reports.
# They include only the three application-side segments shown in the chart.
STAGES = [
    ("FP32\nTensorRT", (6.8551, 26.4743, 0.0711)),
    ("Full INT8\n(failed accuracy)", (6.7124, 11.3143, 0.0703)),
    ("FlashAttention\n(PyTorch SDPA)", (6.60, 8.80, 0.07)),
    ("CUDA preprocess\n+ FlashAttention", (0.22, 8.13, 0.06)),
]
SEGMENTS = ("Preprocess", "Inference", "Postprocess")
COLORS = ("#F4A261", "#2A9D8F", "#8D99AE")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labels = [stage[0] for stage in STAGES]
    values = np.asarray([stage[1] for stage in STAGES], dtype=float)
    totals = values.sum(axis=1)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    x = np.arange(len(STAGES))
    bottom = np.zeros(len(STAGES))
    bars = []
    for index, (name, _) in enumerate(STAGES):
        segment_bars = []
        for segment_index, segment in enumerate(SEGMENTS):
            height = values[index, segment_index]
            bar = ax.bar(
                x[index], height, bottom=bottom[index], width=0.64,
                color=COLORS[segment_index], edgecolor="white", linewidth=0.8,
                label=segment if index == 0 else None,
            )
            segment_bars.extend(bar)
            share = height / totals[index] * 100.0
            # Keep the tiny postprocess label readable without exaggerating it.
            if height >= 0.18:
                ax.text(
                    x[index], bottom[index] + height / 2,
                    f"{height:.2f} ms\n{share:.1f}%",
                    ha="center", va="center", fontsize=8.2,
                    color="white" if segment_index == 1 else "#20242a",
                    fontweight="bold",
                )
            bottom[index] += height
        bars.append(segment_bars)
        ax.text(
            x[index], totals[index] + 0.55,
            f"Total {totals[index]:.2f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Explicitly call out the shares when the postprocess segment is too small
    # to carry a readable in-bar label.
    for index in range(len(STAGES)):
        pre_share = values[index, 0] / totals[index] * 100.0
        infer_share = values[index, 1] / totals[index] * 100.0
        ax.text(
            x[index], -2.0,
            f"pre {pre_share:.1f}%  |  infer {infer_share:.1f}%  |  post {100.0 - pre_share - infer_share:.1f}%",
            ha="center", va="top", fontsize=8.2, color="#3d4650",
        )

    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Mean latency (ms)", fontsize=11)
    ax.set_title(
        "RF-DETR XLarge end-to-end optimization timeline",
        fontsize=14, fontweight="bold", pad=18,
    )
    ax.text(
        0.5, 1.015,
        "RTX 5060 Ti · application timing = preprocess + TensorRT inference + postprocess",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#56616d",
    )
    ax.legend(loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-3.5, max(totals) + 4.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5, 0.015,
        "Numbers are means from the stage reports; the Full INT8 stage is speed-valid but accuracy-invalid.\n"
        "The FlashAttention and CUDA-preprocess stages use the persistent runtime path; see README caveat.",
        ha="center", va="bottom", fontsize=8.5, color="#56616d",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
