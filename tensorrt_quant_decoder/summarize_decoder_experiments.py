#!/usr/bin/env python3
"""Build a decoder sensitivity table, plots, and a readable experiment report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

SPECS = [
    ("baseline_encoder_sq256", "Baseline", "Encoder SmoothQuant SQ256; decoder FP32", "baseline"),
    ("d01_self_qkv_minmax", "D1 Self QKV", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; self-attention Q/K/V", "minmax"),
    ("d02_self_qkvo_minmax", "D2 Self QKVO", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; Q/K/V/output", "minmax"),
    ("d03_ffn_fc1_minmax", "D3 FFN FC1", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear1", "minmax"),
    ("d04_ffn_fc2_minmax", "D4 FFN FC2", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear2", "minmax"),
    ("d05_ffn_minmax", "D5 FFN", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear1/linear2", "minmax"),
    ("d06_cross_value_output_minmax", "D6 Cross Value/Output", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; value/output", "minmax"),
    ("d07_cross_attention_weights_minmax", "D7 Attention Weights", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; attention_weights", "minmax"),
    ("d08_cross_sampling_offsets_minmax", "D8 Sampling Offsets", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; sampling_offsets", "minmax"),
    ("d09_selective_minmax", "D9 Selective", "Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; QKV/QKVO/FFN/value/output", "minmax"),
    ("smoothquant_all_linear_256", "SQ Decoder all Linear", "ModelOpt SmoothQuant; encoder + decoder Linear; 256 images", "smoothquant"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    rows = []
    for directory, label, algorithm, kind in SPECS:
        folder = ART / directory
        full = read_json(folder / "coco_int8_full.json")
        short = read_json(folder / "coco_100.json")
        if directory == "baseline_encoder_sq256":
            full = read_json(ROOT.parent / "tensort_quant" / "artifacts_xlarge_smoothquant_256" / "coco_int8_full.json") or full
        report = full or short
        if not report:
            continue
        metrics = report.get("metrics", {})
        timing = report.get("timing_ms", {}).get("inference", {})
        engine = next(folder.glob("*.engine"), None)
        rows.append({
            "directory": directory,
            "label": label,
            "algorithm": algorithm,
            "kind": kind,
            "images": report.get("num_images"),
            "map_50_95": metrics.get("mAP_50_95"),
            "map_50": metrics.get("mAP_50"),
            "map_75": metrics.get("mAP_75"),
            "inference_ms": timing.get("mean_ms"),
            "engine_mib": engine.stat().st_size / 1024 / 1024 if engine else None,
            "map_100": (short or {}).get("metrics", {}).get("mAP_50_95"),
        })
    baseline = next((r for r in rows if r["kind"] == "baseline"), None)
    if baseline:
        for r in rows:
            r["delta_map"] = None if r["map_50_95"] is None else r["map_50_95"] - baseline["map_50_95"]
            r["delta_ms"] = None if r["inference_ms"] is None or baseline["inference_ms"] is None else r["inference_ms"] - baseline["inference_ms"]
    fields = list(rows[0]) if rows else []
    with (ROOT / "decoder_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    plot_files = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = ROOT / "plots"; plot_dir.mkdir(exist_ok=True)
        valid = [r for r in rows if r.get("map_50_95") is not None]
        labels = [r["label"] for r in valid]
        x = list(range(len(valid)))
        fig, ax = plt.subplots(figsize=(14, 6)); ax.plot(x, [r["map_50_95"] for r in valid], marker="o", label="full COCO mAP")
        ax.set_xticks(x, labels, rotation=38, ha="right"); ax.set_ylabel("mAP@[0.50:0.95]"); ax.set_title("Decoder scope sensitivity: full COCO accuracy")
        ax.grid(axis="y", alpha=.25); fig.tight_layout(); p=plot_dir/"decoder_map_sensitivity.png"; fig.savefig(p,dpi=170); plt.close(fig); plot_files.append(str(p))
        fig, ax = plt.subplots(figsize=(14, 6)); ax.plot(x, [r["inference_ms"] for r in valid], marker="o", color="#e76f51")
        ax.set_xticks(x, labels, rotation=38, ha="right"); ax.set_ylabel("TensorRT inference mean (ms)"); ax.set_title("Decoder scope sensitivity: inference latency")
        ax.grid(axis="y", alpha=.25); fig.tight_layout(); p=plot_dir/"decoder_latency_sensitivity.png"; fig.savefig(p,dpi=170); plt.close(fig); plot_files.append(str(p))
        groups = {"MinMax": [r for r in valid if r["kind"] in {"baseline", "minmax"}], "SmoothQuant": [r for r in valid if r["kind"] == "smoothquant"]}
        fig, ax = plt.subplots(figsize=(8, 5));
        for name, group in groups.items():
            if group: ax.scatter([r["inference_ms"] for r in group], [r["map_50_95"] for r in group], label=name, s=55)
        for r in valid: ax.annotate(r["label"], (r["inference_ms"], r["map_50_95"]), fontsize=7, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel("inference mean (ms)"); ax.set_ylabel("mAP@[0.50:0.95]"); ax.set_title("Accuracy/latency trade-off"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); p=plot_dir/"decoder_accuracy_latency.png"; fig.savefig(p,dpi=170); plt.close(fig); plot_files.append(str(p))
    except ImportError:
        pass

    lines = ["# Decoder 量化阶段实验报告", "", "本报告记录在固定 encoder SmoothQuant（SQ256）基础上，对 decoder 分组进行 INT8 敏感性实验。MinMax 组使用 ONNX Runtime static PTQ/QDQ：激活采用 MinMax 校准，权重采用 per-channel INT8；SmoothQuant 组使用 ModelOpt SmoothQuant，校准图片为 256 张。", "", "## 1. 结果表", "", "| 实验 | 算法与量化范围 | 图片数 | mAP@[0.50:0.95] | 相对基线 | 推理均值(ms) | 引擎(MiB) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['algorithm']} | {r.get('images','')} | {r.get('map_50_95',float('nan')):.6f} | {r.get('delta_map',float('nan')):+.6f} | {r.get('inference_ms',float('nan')):.3f} | {r.get('engine_mib',float('nan')):.1f} |")
    lines += ["", "完整 COCO 结果优先使用 5000 张验证集；只有尚未完成完整评估的实验才保留 100 张预筛选结果。`delta_map` 相对于 baseline_encoder_sq256 计算。", "", "## 2. 如何读取敏感性", "", "D1、D2、D3、D6、D7 在完整数据上与 baseline 接近，说明对应 decoder 子图的 INT8 误差仍能被后续网络吸收。D4 有可见但有限的下降，D5 和联合 D9 的下降明显，说明逐层单独可接受不代表联合量化一定可接受。最终应以完整 COCO 为准，100 张只用于快速筛选。", "", "D6 的额外诊断很关键：cross-attention 的 `value_proj` 输入来自 encoder memory，同时还被 `enc_out_bbox_embed` 等 encoder 输出分支共享。静态 QDQ 在该共享张量上插入 activation Quantize/Dequantize 后，虽然目标节点属于 decoder，encoder-to-decoder 接口及旁路 head 也会受到影响。因此 D6 的逐层表中可能在 `enc_out_bbox_embed` 处先出现误差；这不是权重被误改，而是共享激活边界被量化。若要把 D6 变成严格的“仅 decoder”实验，应先复制该共享输入或采用只量化权重/只在 decoder 分支量化 activation 的图改写。", "", "## 3. Polygraphy 逐层误差分析", "", "每个 D1、D3、D6、D9 目录下的 `error_analysis/` 由 Polygraphy `OnnxrtRunner` 在相同固定输入上导出共同 MatMul/Gemm 输出，并生成 `layerwise_errors.csv`、`summary.json`、`plots/layerwise_error_propagation.png`、`plots/top15_relative_l1.png`、`plots/error_by_op_type.png` 和 Q/DQ scale 审计。横轴 `order` 是 ONNX 拓扑顺序，纵轴 `relative_l1` 是平均绝对误差除以参考张量平均绝对值；它用于定位误差第一次明显上升的位置，不等于最终 mAP 损失。", "", "Polygraphy 的 TensorRT engine 对比还会输出 `polygraphy_error_metrics/`。它比较 baseline engine 和候选 engine 的 `dets`、`labels`，用于确认最终输出是否敏感；中间层定位仍以 ONNX Runtime 共同张量为主。", "", "## 4. SmoothQuant 说明", "", "`smoothquant_decoder.py` 保留 encoder SQ256 配置，同时只启用指定 decoder Linear 模块的输入和权重量化器。self-attention 的 packed `in_proj_weight`、LayerNorm、残差、softmax、deformable sampling 和检测头保持 FP32。ModelOpt 导出和 TensorRT 构建均已通过；完整 COCO 评估可在该目录继续运行。", "", "## 5. 产物目录", "", "- `artifacts/baseline_encoder_sq256/`：固定 encoder SmoothQuant、decoder FP32 基线。", "- `artifacts/d0*_minmax/`：D1-D9 的 decoder 分组 MinMax QDQ ONNX、TensorRT engine、校准 manifest、100/5000 图片评估和逐层误差分析。", "- `artifacts/smoothquant_all_linear_256/`：ModelOpt SmoothQuant decoder Linear 导出模型和 engine。", "- `calibration_images_256.txt`：256 张校准图片清单。", "- `decoder_sensitivity_summary.csv`：可被表格软件直接排序的统一结果。", "- `plots/`：mAP、延迟和精度/延迟散点图。", ""]
    (ROOT / "DECODER_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ROOT / 'DECODER_EXPERIMENT_REPORT.md'}")
    print(f"wrote {ROOT / 'decoder_sensitivity_summary.csv'}")
    for p in plot_files: print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
