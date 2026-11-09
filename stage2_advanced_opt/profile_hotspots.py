#!/usr/bin/env python3
"""Summarize TensorRT profile hotspots for the stage-2 optimization loop.

The input is the compact CSV emitted by
``tensorrt_quant_decoder/compute_graph_analysis/analyze_compute_distribution.py``.
The script deliberately works without pandas so it can run in the TensorRT
environment with only the Python standard library.  Matplotlib is optional;
when it is available the script also writes two small diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


def classify(name: str, stage: str) -> str:
    n = name.lower()
    if stage == "encoder" and "maxrsubexpsumdivmul" in n:
        return "encoder_softmax_fused"
    if stage == "encoder" and "/attention/attention/matmul_1_" in n:
        return "encoder_attention_av"
    if stage == "encoder" and "/attention/attention/matmul_myl0_" in n:
        return "encoder_attention_qkt"
    if stage == "encoder" and "/attention/attention/" in n and "matmul" in n:
        return "encoder_attention_projection"
    if "layernorm" in n or "mean" in n and "sqrt" in n:
        return "normalization"
    if "grid" in n or "gather" in n or "sample" in n:
        return "sampling_or_gather"
    if "matmul" in n or "gemm" in n or "conv" in n:
        return "dense_or_conv"
    return "other"


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["averageMs"] = float(row["averageMs"])
            row["percentage"] = float(row["percentage"])
            row["stage"] = row.get("stage", "unknown")
            row["category"] = classify(str(row["name"]), str(row["stage"]))
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = sum(float(r["averageMs"]) for r in rows)
    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    for r in rows:
        key = str(r["category"])
        item = grouped.setdefault(key, {"category": key, "layers": 0, "averageMs": 0.0})
        item["layers"] = int(item["layers"]) + 1
        item["averageMs"] = float(item["averageMs"]) + float(r["averageMs"])
    output = list(grouped.values())
    for item in output:
        item["share"] = 100.0 * float(item["averageMs"]) / total if total else 0.0
    return sorted(output, key=lambda x: float(x["averageMs"]), reverse=True)


def write_plots(summary: list[dict[str, object]], rows: list[dict[str, object]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out.mkdir(parents=True, exist_ok=True)
    top = sorted(rows, key=lambda x: float(x["averageMs"]), reverse=True)[:20]
    labels = [str(x["name"])[:58] for x in top][::-1]
    values = [float(x["averageMs"]) for x in top][::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, values, color="#2f6f9f")
    ax.set_xlabel("Mean kernel time (ms)")
    ax.set_title("TensorRT profile Top-20 layers")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "top20_layers.png", dpi=160)
    plt.close(fig)

    labels = [str(x["category"]) for x in summary][::-1]
    values = [float(x["averageMs"]) for x in summary][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color="#d46b38")
    ax.set_xlabel("Mean time (ms)")
    ax.set_title("Profile hotspot categories")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "hotspot_categories.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("../tensorrt_quant_decoder/compute_graph_analysis/tensorrt_profile_layers.csv"),
        help="CSV exported from the TensorRT profile parser",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()
    rows = read_rows(args.profile_csv)
    summary = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "hotspot_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "layers", "averageMs", "share"])
        writer.writeheader()
        writer.writerows(summary)
    (args.out_dir / "hotspot_summary.json").write_text(
        json.dumps({"profile": str(args.profile_csv), "layers": len(rows), "totalMs": sum(float(x["averageMs"]) for x in rows), "categories": summary}, indent=2),
        encoding="utf-8",
    )
    write_plots(summary, rows, args.out_dir)
    print(f"profile layers: {len(rows)}")
    print(f"total average time: {sum(float(x['averageMs']) for x in rows):.4f} ms")
    for item in summary[:12]:
        print(f"{item['category']:30s} {item['averageMs']:8.4f} ms {item['share']:6.2f}% ({item['layers']} layers)")


if __name__ == "__main__":
    main()
