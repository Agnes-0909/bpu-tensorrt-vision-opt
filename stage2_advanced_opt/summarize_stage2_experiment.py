#!/usr/bin/env python3
"""Summarize the stage-2 attention and runtime experiments.

The script reads only text/JSON artifacts produced by trtexec, the C++
benchmark, and TensorRT detailed profiling.  It writes CSV summaries and
plots into an output directory so the numerical conclusion can be reviewed
without manually copying values from logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRT_RE = re.compile(
    r"Latency: min = (?P<min>[0-9.]+) ms, max = (?P<max>[0-9.]+) ms, "
    r"mean = (?P<mean>[0-9.]+) ms, median = (?P<median>[0-9.]+) ms, "
    r"percentile\(90%\) = (?P<p90>[0-9.]+) ms, "
    r"percentile\(95%\) = (?P<p95>[0-9.]+) ms, "
    r"percentile\(99%\) = (?P<p99>[0-9.]+) ms"
)
THROUGHPUT_RE = re.compile(r"Throughput: (?P<qps>[0-9.]+) qps")
GRAPH_RE = re.compile(r"CUDA Graph: (?P<value>Enabled|Disabled)")
APP_RE = re.compile(
    r"(?P<name>preprocess|inference|postprocess)\s+mean=(?P<mean>[0-9.]+)\s+ms "
    r"P95=(?P<p95>[0-9.]+)\s+ms P99=(?P<p99>[0-9.]+)\s+ms"
)


def parse_trt(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    latency = TRT_RE.search(text)
    throughput = THROUGHPUT_RE.search(text)
    graph = GRAPH_RE.search(text)
    if not latency or not throughput:
        raise RuntimeError(f"cannot parse trtexec log: {path}")
    result: dict[str, object] = {k: float(v) for k, v in latency.groupdict().items()}
    result["throughput_qps"] = float(throughput.group("qps"))
    result["cuda_graph"] = graph.group("value") if graph else "unknown"
    return result


def parse_app(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for match in APP_RE.finditer(path.read_text(errors="replace")):
        result[match.group("name")] = {
            "mean": float(match.group("mean")),
            "p95": float(match.group("p95")),
            "p99": float(match.group("p99")),
        }
    if set(result) != {"preprocess", "inference", "postprocess"}:
        raise RuntimeError(f"cannot parse C++ benchmark log: {path}")
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_summary(profile_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in ("baseline", "fused"):
        data = json.loads((profile_dir / f"profile_{variant}.json").read_text())
        layers = data[1:] if data and "count" in data[0] else data

        def total(predicate):
            return sum(float(x["averageMs"]) for x in layers if predicate(x["name"]))

        # In the explicit graph QK^T and AV are the two unnamed MatMul nodes
        # directly under attention/attention. The ONNX Attention rewrite turns
        # those into TensorRT internal __fc GEMMs; the profile does not retain
        # the original semantic names, so they are reported as one fused-GEMM
        # category and its layer count is retained separately.
        qkt = total(lambda n: "/attention/attention/MatMul_myl" in n)
        av = total(lambda n: "/attention/attention/MatMul_1_myl" in n)
        softmax = total(lambda n: "MaxrSubExpSumDiv" in n)
        internal_gemm = total(lambda n: n.startswith("__fc"))
        rows.extend(
            [
                {"variant": variant, "metric": "profile_total_ms", "value": total(lambda n: True), "count": len(layers)},
                {"variant": variant, "metric": "explicit_qkt_ms", "value": qkt, "count": sum("/attention/attention/MatMul_myl" in x["name"] for x in layers)},
                {"variant": variant, "metric": "explicit_av_ms", "value": av, "count": sum("/attention/attention/MatMul_1_myl" in x["name"] for x in layers)},
                {"variant": variant, "metric": "softmax_reduction_ms", "value": softmax, "count": sum("MaxrSubExpSumDiv" in x["name"] for x in layers)},
                {"variant": variant, "metric": "internal_attention_gemm_ms", "value": internal_gemm, "count": sum(x["name"].startswith("__fc") for x in layers)},
            ]
        )
    return rows


def plot_trt(records: list[dict[str, object]], out: Path) -> None:
    groups = [("baseline", "no_graph"), ("fused", "no_graph"), ("baseline", "cudagraph"), ("fused", "cudagraph")]
    labels = ["baseline\nno graph", "IAttention\nno graph", "baseline\nCUDA Graph", "IAttention\nCUDA Graph"]
    means, errors, p95 = [], [], []
    for model, mode in groups:
        values = [r["mean"] for r in records if r["model"] == model and r["mode"] == mode]
        p95_values = [r["p95"] for r in records if r["model"] == model and r["mode"] == mode]
        m, s = mean_std(values)
        means.append(m); errors.append(s); p95.append(float(np.mean(p95_values)))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, means, yerr=errors, capsize=4, color=["#4c78a8", "#f58518", "#4c78a8", "#f58518"], alpha=0.9)
    ax.plot(x, p95, "k^--", label="P95 mean")
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_xticks(x, labels)
    ax.set_title("TensorRT enqueue benchmark: explicit vs ONNX Attention")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_app(app_rows: list[dict[str, object]], out: Path) -> None:
    labels = ["baseline\nper-call", "baseline\npersistent", "IAttention\nper-call", "IAttention\npersistent"]
    keys = [("baseline", "per_call"), ("baseline", "persistent"), ("fused", "per_call"), ("fused", "persistent")]
    means, p95, errors = [], [], []
    for model, mode in keys:
        rows = [r for r in app_rows if r["model"] == model and r["mode"] == mode and r["stage"] == "inference"]
        values = [float(r["mean_ms"]) for r in rows]
        m, s = mean_std(values); means.append(m); errors.append(s); p95.append(float(np.mean([r["p95_ms"] for r in rows])))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, means, yerr=errors, capsize=4, color=["#4c78a8", "#59a14f", "#f58518", "#eeca3b"], alpha=0.9)
    ax.plot(x, p95, "k^--", label="P95 mean")
    ax.set_ylabel("Inference latency (ms, lower is better)")
    ax.set_xticks(x, labels)
    ax.set_title("Application benchmark: persistent buffers remove per-call allocation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def plot_breakdown(app_rows: list[dict[str, object]], out: Path) -> None:
    variants = [("baseline", "persistent"), ("fused", "persistent")]
    stages = ["preprocess", "inference", "postprocess"]
    x = np.arange(len(variants)); bottom = np.zeros(len(variants))
    colors = {"preprocess": "#bab0ab", "inference": "#4c78a8", "postprocess": "#59a14f"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for stage in stages:
        values = []
        for model, mode in variants:
            rows = [r for r in app_rows if r["model"] == model and r["mode"] == mode and r["stage"] == stage]
            values.append(float(np.mean([r["mean_ms"] for r in rows])))
        ax.bar(x, values, bottom=bottom, label=stage, color=colors[stage]); bottom += values
    ax.set_ylabel("Mean time per image (ms)")
    ax.set_xticks(x, ["baseline", "IAttention"])
    ax.set_title("End-to-end application timing breakdown")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).parent / "experiments/attention_fp16_fusion")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "analysis_outputs/attention_fp16_fusion")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trt_records: list[dict[str, object]] = []
    for path in sorted(args.experiment_dir.glob("*_no_graph_run*.txt")) + sorted(args.experiment_dir.glob("*_cudagraph_run*.txt")):
        match = re.match(r"(baseline|fused)_(no_graph|cudagraph)_run(\d+)\.txt", path.name)
        if not match: continue
        row = parse_trt(path)
        row.update(model=match.group(1), mode=match.group(2), run=int(match.group(3)), file=path.name)
        trt_records.append(row)
    write_csv(args.out_dir / "trtexec_benchmark.csv", trt_records)

    app_records: list[dict[str, object]] = []
    for path in sorted(args.experiment_dir.glob("app_*.txt")):
        match = re.match(r"app_(baseline|fused)_(per_call|persistent)(?:_run(\d+)_300)?_500\.txt$", path.name)
        if not match:
            match = re.match(r"app_(baseline|fused)_(per_call|persistent)_run(\d+)_300\.txt$", path.name)
        if not match: continue
        model, mode, run = match.group(1), match.group(2), int(match.group(3) or 1)
        # The one-shot *_500 persistent logs are kept as a sanity check, but
        # the repeated *_runN_300 measurements are the stable sample used for
        # the persistent-bar error bars.
        if mode == "persistent" and path.name.endswith("_500.txt"):
            continue
        for stage, values in parse_app(path).items():
            app_records.append({"model": model, "mode": mode, "run": run, "stage": stage, "mean_ms": values["mean"], "p95_ms": values["p95"], "p99_ms": values["p99"], "file": path.name})
    write_csv(args.out_dir / "application_benchmark.csv", app_records)

    profile_rows = profile_summary(args.experiment_dir)
    write_csv(args.out_dir / "profile_summary.csv", profile_rows)
    plot_trt(trt_records, args.out_dir / "trtexec_latency.png")
    plot_app(app_records, args.out_dir / "application_inference_latency.png")
    plot_breakdown(app_records, args.out_dir / "application_breakdown.png")
    print(f"wrote {len(trt_records)} TensorRT runs, {len(app_records)} application rows, {len(profile_rows)} profile rows to {args.out_dir}")


if __name__ == "__main__":
    main()
