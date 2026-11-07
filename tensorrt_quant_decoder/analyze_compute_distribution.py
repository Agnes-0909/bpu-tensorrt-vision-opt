#!/usr/bin/env python3
"""Quantify encoder/decoder/head work from ONNX and TensorRT profile data.

The ONNX part reports operator FLOPs for Conv/MatMul/Gemm (the dominant dense
compute). The TensorRT part groups profiled fused layers into encoder,
decoder, and head. Since TensorRT fuses several neighboring ONNX nodes, the
profile grouping is deliberately based on graph order and explicit stage
markers rather than pretending that every fused kernel belongs to one ONNX
node.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import shape_inference


def shape_map(model: Any) -> dict[str, list[int | None]]:
    result: dict[str, list[int | None]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for item in values:
        tensor = item.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims: list[int | None] = []
        for dim in tensor.shape.dim:
            # PyTorch export commonly leaves the concrete batch as dim_value=0
            # in this fixed-shape graph. Runtime input is batch 1, so use 1 for
            # FLOP accounting while retaining None for genuinely symbolic dims.
            dims.append(1 if dim.dim_value == 0 else (int(dim.dim_value) if dim.dim_value else None))
        result[item.name] = dims
    for item in model.graph.initializer:
        result[item.name] = [int(x) for x in item.dims]
    return result


def numel(shape: list[int | None] | None) -> int | None:
    if not shape or any(x is None for x in shape):
        return None
    return int(math.prod(int(x) for x in shape))


def product_known(values: list[int | None]) -> int | None:
    return numel(values)


def node_flops(node: Any, shapes: dict[str, list[int | None]]) -> tuple[float, str, list[int | None] | None]:
    out_shape = shapes.get(node.output[0]) if node.output else None
    if node.op_type == "MatMul" and len(node.input) >= 2:
        a, b = shapes.get(node.input[0]), shapes.get(node.input[1])
        if a and b and len(a) >= 2 and len(b) >= 2 and out_shape:
            k = a[-1]
            out_n = numel(out_shape)
            if k is not None and out_n is not None and b[-1] is not None:
                return float(2 * out_n * k), "MatMul", out_shape
    if node.op_type == "Gemm" and len(node.input) >= 2:
        a, b = shapes.get(node.input[0]), shapes.get(node.input[1])
        if a and b and len(a) >= 2 and len(b) >= 2:
            m, k, n = a[-2], a[-1], b[-1]
            if m is not None and k is not None and n is not None:
                return float(2 * m * k * n), "Gemm", out_shape
    if node.op_type == "Conv" and len(node.input) >= 2 and out_shape:
        w = shapes.get(node.input[1])
        if w and len(w) >= 4 and numel(out_shape) is not None:
            groups = 1
            for attr in node.attribute:
                if attr.name == "group":
                    groups = int(attr.i)
            kernel = int(math.prod(int(x) for x in w[2:] if x is not None))
            cin_per_group = w[1]
            if cin_per_group is not None:
                return float(2 * numel(out_shape) * cin_per_group * kernel), "Conv", out_shape
    # For non-dense nodes retain element count as a small, clearly separate
    # proxy. It is not mixed into the headline GEMM/Conv FLOPs.
    return 0.0, node.op_type, out_shape


def onnx_stage(name: str) -> str:
    if "/transformer/decoder/" in name:
        return "decoder"
    if any(x in name for x in ("/transformer/enc_output", "/transformer/enc_out_", "/class_embed/", "/bbox_embed/")):
        return "head"
    if "/backbone/" in name:
        return "encoder"
    # ONNX nodes outside the named backbone/decoder/head paths are usually
    # shape/postprocess glue. Keep them visible instead of silently assigning
    # them to a neural-network stage.
    return "other"


def onnx_substage(stage: str, name: str) -> str:
    if stage == "encoder":
        if "/projector/" in name:
            return "feature projector"
        if "/attention/attention/query/" in name or "/attention/attention/key/" in name or "/attention/attention/value/" in name:
            return "attention QKV projection"
        if "/attention/attention/MatMul_1" in name:
            return "attention AV"
        if "/attention/attention/MatMul" in name:
            return "attention QK^T"
        if "/attention/output/" in name:
            return "attention output projection"
        if "/mlp/fc1/" in name:
            return "FFN fc1"
        if "/mlp/fc2/" in name:
            return "FFN fc2"
    elif stage == "decoder":
        if "/self_attn/" in name:
            return "self-attention"
        if "/cross_attn/" in name:
            return "deformable cross-attention"
        if "/linear1/" in name or "/linear2/" in name:
            return "FFN"
        if "/ref_point_head/" in name:
            return "reference-point head"
    elif stage == "head":
        if "/transformer/enc_output" in name:
            return "encoder output projection"
        if "/transformer/enc_out_class_embed/" in name:
            return "encoder class head"
        if "/transformer/enc_out_bbox_embed/" in name:
            return "encoder bbox head"
        if "/class_embed/" in name:
            return "decoder class head"
        if "/bbox_embed/" in name:
            return "decoder bbox head"
    return "other"


def profile_stage(index: int, first_head: int, first_decoder: int, first_final_head: int) -> str:
    if index < first_head:
        return "encoder"
    if index < first_decoder:
        return "head_predecoder"
    if index < first_final_head:
        return "decoder"
    return "head_postdecoder"


def load_profile(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [x for x in data if isinstance(x, dict) and "averageMs" in x and "name" in x]
    first_head = next((i for i, x in enumerate(rows) if "/transformer/enc_output" in x["name"]), len(rows))
    first_decoder = next((i for i, x in enumerate(rows) if "/transformer/decoder/layers." in x["name"]), len(rows))
    first_final_head = next((i for i, x in enumerate(rows) if "/class_embed/" in x["name"] or "/bbox_embed/" in x["name"]), len(rows))
    boundaries = {"first_head": first_head, "first_decoder": first_decoder, "first_final_head": first_final_head}
    for i, row in enumerate(rows):
        row["stage"] = profile_stage(i, first_head, first_decoder, first_final_head)
    return rows, boundaries


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def fmt_gflops(x: float) -> str:
    return f"{x / 1e9:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--onnx", type=Path, default=Path("/home/rton/rfdetr/tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx"))
    parser.add_argument("--profile", type=Path, default=root / "compute_graph_analysis/baseline_profile.json")
    parser.add_argument("--output-dir", type=Path, default=root / "compute_graph_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(args.onnx), load_external_data=True)
    # The exported graph has no value_info entries. Shape inference supplies
    # the output dimensions needed to count MatMul/Gemm/Conv operations.
    model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    shapes = shape_map(model)
    op_rows: list[dict[str, Any]] = []
    for order, node in enumerate(model.graph.node):
        flops, counted_as, out_shape = node_flops(node, shapes)
        op_rows.append({
            "order": order,
            "stage": onnx_stage(node.name),
            "op_type": node.op_type,
            "node_name": node.name,
            "output": node.output[0] if node.output else "",
            "output_shape": str(out_shape),
            "flops": flops,
            "gflops": flops / 1e9,
            "counted_operator": counted_as,
        })
    write_rows(args.output_dir / "onnx_compute_nodes.csv", op_rows)

    onnx_groups: dict[str, dict[str, Any]] = {}
    for stage in ("encoder", "decoder", "head", "other"):
        subset = [x for x in op_rows if x["stage"] == stage]
        counted = [x for x in subset if x["flops"] > 0]
        by_op = Counter(x["counted_operator"] for x in counted)
        onnx_groups[stage] = {
            "node_count": len(subset),
            "dense_operator_count": len(counted),
            "flops": sum(x["flops"] for x in subset),
            "gflops": sum(x["gflops"] for x in subset),
            "by_operator_gflops": {k: sum(x["gflops"] for x in counted if x["counted_operator"] == k) for k in by_op},
            "by_substage_gflops": {
                sub: sum(x["gflops"] for x in counted if onnx_substage(stage, x["node_name"]) == sub)
                for sub in sorted({onnx_substage(stage, x["node_name"]) for x in counted})
            },
        }

    profile_rows: list[dict[str, Any]] = []
    profile_groups: dict[str, dict[str, Any]] = {}
    boundaries: dict[str, Any] = {}
    if args.profile.exists():
        profile_rows, boundaries = load_profile(args.profile)
        write_rows(args.output_dir / "tensorrt_profile_layers.csv", profile_rows)
        for stage in ("encoder", "head_predecoder", "decoder", "head_postdecoder"):
            subset = [x for x in profile_rows if x["stage"] == stage]
            profile_groups[stage] = {
                "layer_count": len(subset),
                "profile_sum_ms": sum(float(x["averageMs"]) for x in subset),
                "profile_sum_pct": sum(float(x.get("percentage", 0.0)) for x in subset),
            }
        profile_groups["head"] = {
            "layer_count": profile_groups["head_predecoder"]["layer_count"] + profile_groups["head_postdecoder"]["layer_count"],
            "profile_sum_ms": profile_groups["head_predecoder"]["profile_sum_ms"] + profile_groups["head_postdecoder"]["profile_sum_ms"],
            "profile_sum_pct": profile_groups["head_predecoder"]["profile_sum_pct"] + profile_groups["head_postdecoder"]["profile_sum_pct"],
        }

    total_flops = sum(x["flops"] for x in op_rows)
    total_profile = sum(float(x["averageMs"]) for x in profile_rows)
    for group in onnx_groups.values():
        group["flops_pct"] = 100.0 * group["flops"] / total_flops if total_flops else 0.0
    summary = {
        "model": str(args.onnx),
        "input_shape": [1, 3, 700, 700],
        "stage_definition": {
            "encoder": "backbone DINOv2 + multi-scale feature projector",
            "head": "transformer enc_output/enc_out_* selection and class/bbox heads; head includes pre/post decoder regions in TensorRT profile",
            "decoder": "transformer.decoder.ref_point_head and 5 decoder layers",
            "other": "graph glue not attributed to named stages in ONNX FLOPs table",
        },
        "onnx_dense_compute": {"total_gflops": total_flops / 1e9, "stages": onnx_groups},
        "tensorrt_profile": {"profile_file": str(args.profile), "profile_sum_ms": total_profile, "stage_boundaries": boundaries, "stages": profile_groups},
        "limitations": [
            "ONNX FLOPs count multiply-add as two operations and only reports Conv/MatMul/Gemm, where static shapes are available.",
            "TensorRT fuses nodes; profile stage percentages are contiguous graph-region attribution and include fused glue kernels.",
            "Profile sum is not identical to ordinary trtexec latency because profiling changes execution and reports per-layer timing.",
            "The report is for batch 1, input 3x700x700, one stream, CUDA Graph enabled.",
        ],
    }
    (args.output_dir / "compute_distribution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = args.output_dir / "plots"; plot_dir.mkdir(exist_ok=True)
        stages = ["encoder", "decoder", "head", "other"]
        vals = [onnx_groups[s]["gflops"] for s in stages]
        fig, ax = plt.subplots(figsize=(8, 5)); bars=ax.bar(stages, vals, color=["#457b9d", "#e76f51", "#2a9d8f", "#a8a8a8"])
        ax.set_ylabel("Dense compute (GFLOPs; MAC=2 FLOPs)"); ax.set_title("ONNX compute distribution"); ax.grid(axis="y", alpha=.25)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.1f} ({100*v/(sum(vals) or 1):.1f}%)",ha="center",va="bottom",fontsize=9)
        fig.tight_layout(); fig.savefig(plot_dir/"onnx_flops_by_stage.png",dpi=170); plt.close(fig)
        if profile_groups:
            labels=["encoder","decoder","head"]
            pvals=[profile_groups[s]["profile_sum_ms"] for s in labels]
            fig, ax = plt.subplots(figsize=(8,5)); bars=ax.bar(labels,pvals,color=["#457b9d","#e76f51","#2a9d8f"])
            ax.set_ylabel("TensorRT profile sum (ms)"); ax.set_title("TensorRT profiled time distribution"); ax.grid(axis="y",alpha=.25)
            for b,v in zip(bars,pvals): ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f} ms",ha="center",va="bottom",fontsize=9)
            fig.tight_layout(); fig.savefig(plot_dir/"tensorrt_time_by_stage.png",dpi=170); plt.close(fig)
            fig, ax=plt.subplots(figsize=(8,5)); ax.pie(pvals,labels=labels,autopct="%1.1f%%",colors=["#457b9d","#e76f51","#2a9d8f"]); ax.set_title("TensorRT profiled time share")
            fig.tight_layout(); fig.savefig(plot_dir/"tensorrt_time_share.png",dpi=170); plt.close(fig)
    except ImportError:
        pass

    report = ["# RF-DETR XLarge 全计算图计算量分布", "", "## 1. 统计对象", "", f"模型：`{args.onnx}`；输入固定为 `1×3×700×700`。ONNX 统计以 Conv/MatMul/Gemm 为主，乘加按 2 FLOPs 计；TensorRT 统计来自 `trtexec --profilingVerbosity=detailed --dumpProfile`，并把融合 kernel 按计算图连续区域归属到 encoder、decoder、head。", "", "## 2. 结果", "", "| 阶段 | ONNX dense FLOPs | FLOPs 占比 | TensorRT profile 时间 | profile 时间占比 |", "|---|---:|---:|---:|---:|"]
    for stage in ("encoder", "decoder", "head"):
        og=onnx_groups[stage]; pg=profile_groups.get(stage, {})
        report.append(f"| {stage} | {og['gflops']:.2f} GFLOPs | {og['flops_pct']:.1f}% | {pg.get('profile_sum_ms', float('nan')):.3f} ms | {pg.get('profile_sum_pct', float('nan')):.1f}% |")
    report += ["", "## 3. encoder/decoder 内部构成", "", "| 阶段 | 子模块 | dense FLOPs | 阶段内占比 |", "|---|---|---:|---:|"]
    for stage in ("encoder", "decoder", "head"):
        total = onnx_groups[stage]["flops"] or 1.0
        for sub, value in sorted(onnx_groups[stage]["by_substage_gflops"].items(), key=lambda x: -x[1]):
            if value > 0:
                report.append(f"| {stage} | {sub} | {value:.2f} GFLOPs | {100 * value * 1e9 / total:.1f}% |")
    report += ["", "## 4. 如何解读", "", "- FLOPs 表示理论计算量，回答“需要做多少乘加”；TensorRT profile 表示实际 kernel 时间，回答“GPU 在这个版本上花了多少时间”。两者不会严格成正比，因为 TensorRT 会融合算子、使用不同精度 kernel，并受到访存、同步和 plugin 影响。", "- encoder 包含 DINOv2 backbone 和多尺度 feature projector；这里的 attention 是规则的大矩阵计算，重复 block 多，因此既有最大的 dense FLOPs，也有最大的实际耗时占比。", "- encoder 内部 FLOPs 最大的是 FFN fc1/fc2，其次是 QKᵀ、AV 和 QKV projection；这解释了为什么 encoder Linear/attention projection 的 INT8 化能够影响整体速度。", "- decoder 只有 5 层、query 数量较小，并且 deformable cross-attention 中包含采样、GridSample、Gather 和残差归一化；其中很多操作不是大规模规则 GEMM，所以 decoder FLOPs 和可被 INT8 加速的 FLOPs 都有限。", "- head 包含 encoder 输出筛选、TopK、reference/bbox/class 分支以及最终检测头。它的 dense FLOPs 很小，但包含后处理相关的图操作，不能只用 FLOPs 判断其时间。", "", "## 5. 与量化收益的关系", "", "当前 encoder SmoothQuant 的速度收益主要来自覆盖了计算图中占比最大的规则 dense encoder 计算；decoder 量化只覆盖较小的 GEMM 子集，Softmax、deformable sampling、GridSample、LayerNorm 和残差仍然保留，因此完整图推理收益自然有限。", "", "## 6. 产物", "", "- `compute_distribution_summary.json`：完整机器可读统计。", "- `onnx_compute_nodes.csv`：逐 ONNX 节点的 stage、算子、shape 和 FLOPs。", "- `tensorrt_profile_layers.csv`：逐 TensorRT 融合 layer 的平均时间和占比。", "- `plots/onnx_flops_by_stage.png`：理论 dense FLOPs 分布。", "- `plots/tensorrt_time_by_stage.png`、`tensorrt_time_share.png`：实际 profile 时间分布。", ""]
    (args.output_dir / "COMPUTE_DISTRIBUTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("wrote", args.output_dir / "COMPUTE_DISTRIBUTION_REPORT.md")
    print(json.dumps({"onnx": onnx_groups, "profile": profile_groups}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
