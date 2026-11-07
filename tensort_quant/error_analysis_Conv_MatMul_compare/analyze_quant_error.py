#!/usr/bin/env python3
"""Use Polygraphy/ONNX Runtime to locate Conv+MatMul INT8 error growth.

The script is intentionally an analysis tool rather than a calibration tool.
It runs the FP32 and Q/DQ ONNX graphs on exactly the same saved input(s),
marks the common Conv/MatMul outputs, and calculates a metric for every
marked tensor.  It also compares the two already-built TensorRT engines at
their public outputs and writes a small audit of the Q/DQ scales.

The recommended invocation is from the ``TensorRT`` conda environment::

    conda activate TensorRT
    python analyze_quant_error.py

The TensorRT environment in this workspace does not include the ``onnx``
wheel, while Polygraphy needs it to mark intermediate outputs.  The script
therefore tries the matching ``torch`` environment's site-packages as a
read-only fallback.  TensorRT, Polygraphy, ONNX Runtime, and NumPy still come
from the active ``TensorRT`` environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _import_onnx_with_fallback() -> Any:
    """Import ONNX, adding the local compatible wheel only when necessary."""

    try:
        import onnx  # type: ignore

        return onnx
    except ModuleNotFoundError as first_error:
        candidates = [
            Path("/home/rton/miniconda3/envs/torch/lib/python3.12/site-packages"),
            Path(sys.executable).resolve().parent.parent / "lib/python3.12/site-packages",
        ]
        for candidate in candidates:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                try:
                    import onnx  # type: ignore

                    print(
                        f"[info] using ONNX wheel from {candidate} for graph inspection",
                        file=sys.stderr,
                    )
                    return onnx
                except ModuleNotFoundError:
                    continue
        raise RuntimeError(
            "Polygraphy needs the Python package 'onnx' to mark intermediate "
            "outputs. Install it in the TensorRT environment, or make the "
            "workspace's torch environment available."
        ) from first_error


onnx = _import_onnx_with_fallback()

import numpy as np  # noqa: E402
from onnx import numpy_helper  # noqa: E402
from polygraphy.backend.common import BytesFromPath  # noqa: E402
from polygraphy.backend.onnx import ModifyOutputs  # noqa: E402
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx  # noqa: E402
from polygraphy.backend.trt import EngineFromBytes, TrtRunner  # noqa: E402
from polygraphy.comparator import Comparator, SimpleCompareFunc, StreamingDataLoader  # noqa: E402
from polygraphy.logger import G_LOGGER  # noqa: E402

try:  # Matplotlib is optional; the numerical CSV/JSON report remains usable without it.
    import matplotlib  # type: ignore  # noqa: E402

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover - depends on the local conda environment
    plt = None


# Intermediate tensors can be very large (the attention score tensors are
# [1, 12, 1226, 1226]).  Keep Polygraphy's verbose tensor dump out of the
# terminal; the useful information is written to CSV/JSON below.
G_LOGGER.module_severity = G_LOGGER.WARNING


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    repo = root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp32-onnx",
        type=Path,
        default=repo / "python_infer/output/rfdetr-xlarge.onnx",
        help="FP32 source ONNX model",
    )
    parser.add_argument(
        "--int8-onnx",
        type=Path,
        default=repo / "tensort_quant/artifacts_xlarge/rfdetr-xlarge-int8.onnx",
        help="full Conv+MatMul INT8 Q/DQ ONNX model",
    )
    parser.add_argument(
        "--fp32-engine",
        type=Path,
        default=repo / "model_xlarge.engine",
        help="FP32 TensorRT engine used for final-output comparison",
    )
    parser.add_argument(
        "--int8-engine",
        type=Path,
        default=repo / "tensort_quant/artifacts_xlarge/rfdetr-xlarge-int8.engine",
        help="full Conv+MatMul INT8 TensorRT engine",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=root.parent / "polygraphy_analysis/.inputs.json",
        help="Polygraphy JSON input file; all runners consume the same file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root,
        help="directory for summary.json, layerwise_errors.csv and qdq_scales.csv",
    )
    parser.add_argument(
        "--op-types",
        default="Conv,MatMul",
        help="comma-separated ONNX op types whose outputs are compared",
    )
    parser.add_argument(
        "--algorithm-label",
        default="static PTQ QDQ; MinMax activation calibration; symmetric INT8 activation; per-channel INT8 weights",
        help="algorithm description written into summary.json",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="limit analysis to the first N common tensors (0 means all)",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="skip the TensorRT engine final-output comparison",
    )
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required file(s): " + ", ".join(missing))


def node_specs(model: Any, op_types: set[str]) -> list[dict[str, Any]]:
    """Return common graph output names in deterministic topological order."""

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in op_types:
            continue
        for output in node.output:
            if output and output not in seen:
                seen.add(output)
                specs.append(
                    {
                        "tensor": output,
                        "node_name": node.name or "<unnamed>",
                        "op_type": node.op_type,
                    }
                )
    return specs


def graph_output_specs(model: Any, common: set[str]) -> list[dict[str, Any]]:
    return [
        {"tensor": output.name, "node_name": "<graph-output>", "op_type": "GraphOutput"}
        for output in model.graph.output
        if output.name in common
    ]


def make_marked_model(model: Any, outputs: list[str]) -> Any:
    # copy=True is important: the original model is reused to inspect node and
    # initializer metadata after this call.
    return ModifyOutputs(model, outputs=outputs, copy=True)()


def run_onnxrt(model: Any, output_names: list[str], inputs: Path, label: str) -> dict[str, np.ndarray]:
    marked = make_marked_model(model, output_names)
    runner = OnnxrtRunner(
        SessionFromOnnx(marked.SerializeToString(), providers=["cpu"]), name=label
    )
    results = Comparator.run(
        [runner], data_loader=StreamingDataLoader(str(inputs)), warm_up=0
    )
    return dict(results[label][0])


def finite(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    got = np.asarray(candidate, dtype=np.float32)
    if ref.shape != got.shape:
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(got.shape),
        }
    delta = got - ref
    abs_delta = np.abs(delta)
    ref_abs = np.abs(ref)
    ref_mean_abs = float(np.mean(ref_abs))
    delta_mean_abs = float(np.mean(abs_delta))
    ref_norm = float(np.linalg.norm(ref.ravel()))
    got_norm = float(np.linalg.norm(got.ravel()))
    cosine = float(np.dot(ref.ravel(), got.ravel()) / max(ref_norm * got_norm, 1e-12))
    relative_l1 = delta_mean_abs / max(ref_mean_abs, 1e-12)
    return {
        "shape_match": True,
        "shape": list(ref.shape),
        "numel": int(ref.size),
        "reference_mean": finite(np.mean(ref)),
        "candidate_mean": finite(np.mean(got)),
        "reference_mean_abs": finite(ref_mean_abs),
        "candidate_mean_abs": finite(np.mean(np.abs(got))),
        "reference_max_abs": finite(np.max(ref_abs)),
        "candidate_max_abs": finite(np.max(np.abs(got))),
        "mean_abs": finite(delta_mean_abs),
        "max_abs": finite(np.max(abs_delta)),
        "rmse": finite(np.sqrt(np.mean(delta * delta))),
        "relative_l1": finite(relative_l1),
        "cosine_similarity": finite(cosine),
        "p99_abs": finite(np.percentile(abs_delta, 99.0)),
        "nonfinite_reference": int(np.size(ref) - np.count_nonzero(np.isfinite(ref))),
        "nonfinite_candidate": int(np.size(got) - np.count_nonzero(np.isfinite(got))),
    }


def qdq_audit(model: Any) -> dict[str, Any]:
    """Summarize Q/DQ coverage and scale ranges without dumping weights."""

    initializers = {item.name: item for item in model.graph.initializer}
    scale_rows: list[dict[str, Any]] = []
    qdq_counts = Counter(node.op_type for node in model.graph.node)
    quantized_output_names = {
        node.input[0]
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input
    }
    quantized_ops: dict[str, dict[str, Any]] = {}
    for op_type in ("Conv", "MatMul"):
        nodes = [node for node in model.graph.node if node.op_type == op_type]
        quantized = [
            node for node in nodes if any(output in quantized_output_names for output in node.output)
        ]
        quantized_ops[op_type] = {
            "total": len(nodes),
            "with_output_quantize": len(quantized),
            "without_output_quantize": len(nodes) - len(quantized),
            "without_output_quantize_names": [node.name for node in nodes if node not in quantized],
        }
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            continue
        scale = np.asarray(numpy_helper.to_array(initializers[node.input[1]]), dtype=np.float32)
        axis = next((int(attr.i) for attr in node.attribute if attr.name == "axis"), None)
        scale_rows.append(
            {
                "node": node.name,
                "op_type": node.op_type,
                "tensor": node.input[0],
                "scale_name": node.input[1],
                "scale_shape": list(scale.shape),
                "scale_min": finite(np.min(scale)),
                "scale_max": finite(np.max(scale)),
                "scale_mean": finite(np.mean(scale)),
                "axis": axis,
                "per_channel": bool(scale.size > 1),
            }
        )
    return {
        "node_counts": dict(qdq_counts),
        "quantized_ops": quantized_ops,
        "quantize_linear_count": int(qdq_counts["QuantizeLinear"]),
        "dequantize_linear_count": int(qdq_counts["DequantizeLinear"]),
        "scale_count": len(scale_rows),
        "per_channel_scale_count": sum(row["per_channel"] for row in scale_rows),
        "scalar_scale_count": sum(not row["per_channel"] for row in scale_rows),
        "scale_rows": scale_rows,
    }


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def engine_compare(
    fp32_engine: Path, int8_engine: Path, inputs: Path, plot_dir: Path
) -> dict[str, Any]:
    """Compare final outputs from the serialized engines on the same input."""

    fp_runner = TrtRunner(
        EngineFromBytes(BytesFromPath(str(fp32_engine))), name="fp32_engine"
    )
    int8_runner = TrtRunner(
        EngineFromBytes(BytesFromPath(str(int8_engine))), name="int8_engine"
    )
    results = Comparator.run(
        [fp_runner, int8_runner],
        data_loader=StreamingDataLoader(str(inputs)),
        warm_up=0,
    )
    # This is Polygraphy's own error-metrics plotter. It produces one PNG per
    # public output and complements the custom graph-level plots below.
    polygraphy_plot_dir = plot_dir / "polygraphy_error_metrics"
    Comparator.compare_accuracy(
        results,
        compare_func=SimpleCompareFunc(
            check_shapes=True,
            rtol=1e-2,
            atol=1e-2,
            save_error_metrics_plot=str(polygraphy_plot_dir),
        ),
    )
    fp = results["fp32_engine"][0]
    iq = results["int8_engine"][0]
    output_rows = {}
    for name in sorted(set(fp) & set(iq)):
        output_rows[name] = tensor_metrics(fp[name], iq[name])
    return {
        "outputs": output_rows,
        "input_iterations": len(results["fp32_engine"]),
        "polygraphy_error_metrics_plot_dir": str(polygraphy_plot_dir),
    }


def make_plots(rows: list[dict[str, Any]], audit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Create compact PNGs from the metrics, without retaining model tensors."""

    if plt is None:
        return {"available": False, "reason": "matplotlib is not installed"}

    valid = [row for row in rows if row.get("relative_l1") is not None]
    if not valid:
        return {"available": False, "reason": "no finite layerwise metrics"}
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Conv": "#2a9d8f", "MatMul": "#e76f51", "GraphOutput": "#264653"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for op_type in sorted({row["op_type"] for row in valid}):
        subset = [row for row in valid if row["op_type"] == op_type]
        axes[0].scatter(
            [row["order"] for row in subset],
            [row["relative_l1"] for row in subset],
            s=20,
            alpha=0.8,
            label=op_type,
            color=colors.get(op_type, "#457b9d"),
        )
        axes[1].scatter(
            [row["order"] for row in subset],
            [row["cosine_similarity"] for row in subset],
            s=20,
            alpha=0.8,
            label=op_type,
            color=colors.get(op_type, "#457b9d"),
        )
    axes[0].axhline(0.5, color="#666", linestyle="--", linewidth=1, label="relative L1 = 0.5")
    axes[0].set_ylabel("relative L1 error")
    axes[0].set_yscale("symlog", linthresh=0.05)
    axes[0].set_title("Layerwise quantization error propagation")
    axes[0].grid(alpha=0.2)
    axes[1].axhline(1.0, color="#666", linestyle="--", linewidth=1)
    axes[1].set_xlabel("topological order")
    axes[1].set_ylabel("cosine similarity")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=4, loc="lower left")
    fig.savefig(output_dir / "layerwise_error_propagation.png", dpi=160)
    plt.close(fig)

    top = sorted(valid, key=lambda row: row["relative_l1"], reverse=True)[:15]
    top.reverse()
    fig, axis = plt.subplots(figsize=(13, 8), constrained_layout=True)
    axis.barh(
        range(len(top)),
        [row["relative_l1"] for row in top],
        color=[colors.get(row["op_type"], "#457b9d") for row in top],
    )
    axis.set_yticks(range(len(top)))
    axis.set_yticklabels(
        [f"{row['order']}: {row['node_name'].split('/')[-1]}" for row in top], fontsize=8
    )
    axis.set_xlabel("relative L1 error")
    axis.set_title("Top 15 tensors by relative L1 error")
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(output_dir / "top15_relative_l1.png", dpi=160)
    plt.close(fig)

    grouped = {
        op_type: [row["relative_l1"] for row in valid if row["op_type"] == op_type]
        for op_type in sorted({row["op_type"] for row in valid})
    }
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = list(grouped)
    try:
        axis.boxplot(
            [grouped[label] for label in labels],
            tick_labels=labels,
            showfliers=False,
        )
    except TypeError:  # Matplotlib < 3.9 used the ``labels`` keyword.
        axis.boxplot(
            [grouped[label] for label in labels],
            labels=labels,
            showfliers=False,
        )
    axis.set_ylabel("relative L1 error")
    axis.set_title("Error distribution by operator type")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "error_by_op_type.png", dpi=160)
    plt.close(fig)

    files = [
        str(output_dir / "layerwise_error_propagation.png"),
        str(output_dir / "top15_relative_l1.png"),
        str(output_dir / "error_by_op_type.png"),
    ]
    scales = [row["scale_min"] for row in audit["scale_rows"] if row.get("scale_min", 0) > 0]
    if scales:
        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        axis.hist(np.log10(scales), bins=40, color="#457b9d", alpha=0.85)
        axis.set_xlabel("log10(scale_min)")
        axis.set_ylabel("number of Q/DQ tensors")
        axis.set_title("Q/DQ scale distribution")
        axis.grid(axis="y", alpha=0.2)
        fig.savefig(output_dir / "qdq_scale_distribution.png", dpi=160)
        plt.close(fig)
        files.append(str(output_dir / "qdq_scale_distribution.png"))

    return {
        "available": True,
        "files": files,
    }


def main() -> int:
    args = parse_args()
    op_types = {item.strip() for item in args.op_types.split(",") if item.strip()}
    if not op_types:
        raise ValueError("--op-types must contain at least one ONNX op type")
    if args.max_tensors < 0:
        raise ValueError("--max-tensors must be >= 0")
    require_files([args.fp32_onnx, args.int8_onnx, args.inputs])
    if not args.skip_engine:
        require_files([args.fp32_engine, args.int8_engine])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading ONNX graphs ({', '.join(sorted(op_types))})", flush=True)
    fp_model = onnx.load(str(args.fp32_onnx), load_external_data=True)
    int8_model = onnx.load(str(args.int8_onnx), load_external_data=True)
    fp_specs = node_specs(fp_model, op_types)
    int8_outputs = {
        output for node in int8_model.graph.node if node.op_type in op_types for output in node.output
    }
    fp_output_names = {spec["tensor"] for spec in fp_specs}
    common = fp_output_names & int8_outputs
    # Include public graph outputs in the layerwise table so the same report
    # connects the first bad intermediate tensor to the final model result.
    common |= {output.name for output in fp_model.graph.output} & {
        output.name for output in int8_model.graph.output
    }
    specs = [spec for spec in fp_specs if spec["tensor"] in common]
    specs.extend(graph_output_specs(fp_model, common))
    if args.max_tensors:
        specs = specs[: args.max_tensors]
    output_names = [spec["tensor"] for spec in specs]
    if not output_names:
        raise RuntimeError("no common outputs found; check the two ONNX models")
    spec_by_name = {spec["tensor"]: spec for spec in specs}
    print(f"[2/5] comparing {len(output_names)} common tensors with ONNX Runtime", flush=True)
    start = time.perf_counter()
    fp_outputs = run_onnxrt(fp_model, output_names, args.inputs, "fp32_onnx")
    int8_outputs_data = run_onnxrt(int8_model, output_names, args.inputs, "int8_onnx")
    rows: list[dict[str, Any]] = []
    for order, name in enumerate(output_names):
        metrics = tensor_metrics(fp_outputs[name], int8_outputs_data[name])
        rows.append(
            {
                "order": order,
                "op_type": spec_by_name[name]["op_type"],
                "node_name": spec_by_name[name]["node_name"],
                "tensor": name,
                **metrics,
            }
        )
    del fp_outputs, int8_outputs_data
    elapsed = time.perf_counter() - start
    rows_with_shape = [row for row in rows if row.get("shape_match")]
    ranked = sorted(
        rows_with_shape,
        key=lambda row: float(row.get("relative_l1") or 0.0),
        reverse=True,
    )
    first_severe = next(
        (row for row in rows if row.get("relative_l1") is not None and row["relative_l1"] >= 0.5),
        None,
    )
    type_summary: dict[str, dict[str, Any]] = {}
    for op_type in sorted({row["op_type"] for row in rows}):
        subset = [row for row in rows_with_shape if row["op_type"] == op_type]
        if subset:
            type_summary[op_type] = {
                "count": len(subset),
                "mean_relative_l1": float(np.mean([row["relative_l1"] for row in subset])),
                "max_relative_l1": max(row["relative_l1"] for row in subset),
                "max_tensor": max(subset, key=lambda row: row["relative_l1"])["tensor"],
            }

    print(f"[3/5] writing {args.output_dir}", flush=True)
    write_csv(args.output_dir / "layerwise_errors.csv", rows)
    audit = qdq_audit(int8_model)
    (args.output_dir / "qdq_scales.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    # A CSV is convenient for sorting in a spreadsheet without opening the
    # larger JSON audit.
    write_csv(
        args.output_dir / "qdq_scales.csv",
        audit["scale_rows"],
        fieldnames=[
            "node",
            "op_type",
            "tensor",
            "scale_name",
            "scale_shape",
            "scale_min",
            "scale_max",
            "scale_mean",
            "axis",
            "per_channel",
        ],
    )
    visualization = make_plots(rows, audit, args.output_dir / "plots")

    engine_result: dict[str, Any] | None = None
    if not args.skip_engine:
        print("[4/5] comparing serialized TensorRT engine outputs", flush=True)
        engine_result = engine_compare(
            args.fp32_engine,
            args.int8_engine,
            args.inputs,
            args.output_dir,
        )
    else:
        print("[4/5] TensorRT engine comparison skipped", flush=True)

    summary = {
        "fp32_onnx": str(args.fp32_onnx),
        "int8_onnx": str(args.int8_onnx),
        "fp32_engine": None if args.skip_engine else str(args.fp32_engine),
        "int8_engine": None if args.skip_engine else str(args.int8_engine),
        "inputs": str(args.inputs),
        "quantization_algorithm": args.algorithm_label,
        "op_types": sorted(op_types),
        "common_tensor_count": len(output_names),
        "onnxrt_analysis_seconds": elapsed,
        "first_relative_l1_at_least_0.5": first_severe,
        "top_relative_l1": ranked[:20],
        "by_op_type": type_summary,
        "qdq_audit": {key: value for key, value in audit.items() if key != "scale_rows"},
        "visualizations": visualization,
        "engine_final_output_comparison": engine_result,
        "interpretation": [
            "按 layerwise_errors.csv 的 order 从上到下查看误差首次明显增大的位置。",
            "relative_l1 是该张量平均绝对误差除以 FP32 平均绝对值；attention score 的大误差会被后续层放大。",
            "qdq_scales.csv 用于检查激活/权重 scale 是否出现极端范围、是否为 per-channel。",
            "TensorRT engine 对比只代表最终输出差异；逐层定位使用 ONNX Runtime 的共同中间输出。",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print("[5/5] analysis complete", flush=True)
    if first_severe:
        print(
            "first relative_l1 >= 0.5: "
            f"order={first_severe['order']} {first_severe['op_type']} "
            f"{first_severe['node_name']} ({first_severe['relative_l1']:.4g})"
        )
    print("top tensors by relative_l1:")
    for row in ranked[:10]:
        print(
            f"  {row['relative_l1']:.4g}  {row['op_type']:<8} "
            f"order={row['order']:<3} {row['node_name']}"
        )
    print(f"reports: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
