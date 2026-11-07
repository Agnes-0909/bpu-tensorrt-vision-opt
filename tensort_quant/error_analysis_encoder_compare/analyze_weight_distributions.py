#!/usr/bin/env python3
"""Compare encoder weight distributions before/after MinMax, Percentile and SmoothQuant."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "error_analysis_Conv_MatMul_compare"))
from analyze_quant_error import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

ROOT = HERE.parent.parent
FP = ROOT / "python_infer/output/rfdetr-xlarge.onnx"
MODELS = {
    "MinMax": ROOT / "tensort_quant/artifacts_xlarge_encoder_minmax/rfdetr-xlarge-encoder-minmax-int8.onnx",
    "Percentile": ROOT / "tensort_quant/artifacts_xlarge_encoder_percentile/rfdetr-xlarge-encoder-percentile-int8.onnx",
    "SmoothQuant": ROOT / "tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx",
}
TARGETS = ("/attention/attention/query/MatMul", "/attention/attention/key/MatMul",
           "/attention/attention/value/MatMul", "/attention/output/dense/MatMul",
           "/mlp/fc1/MatMul", "/mlp/fc2/MatMul")


def values(model):
    out = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                out[node.output[0]] = numpy_helper.to_array(attr.t)
    return out


def attr_int(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def dq_tensor(model, tensor_name):
    """Return dequantized weight, quantized integer tensor and scale."""
    prod = {o: n for n in model.graph.node for o in n.output}
    init = values(model)
    node = prod.get(tensor_name)
    transposed = False
    if node is not None and node.op_type == "Transpose":
        transposed = True
        node = prod.get(node.input[0])
    if node is None or node.op_type != "DequantizeLinear":
        raise RuntimeError(f"cannot find weight DQ producer for {tensor_name}")
    qnode = prod.get(node.input[0])
    explicit_q = qnode is not None and qnode.op_type == "QuantizeLinear"
    if explicit_q:
        q_name = qnode.input[0]
    else:
        # ONNX Runtime QDQ stores static weights directly as an INT8
        # initializer feeding DequantizeLinear; ModelOpt may retain an
        # explicit QuantizeLinear node instead.
        q_name = node.input[0]
    if q_name not in init:
        raise RuntimeError(f"cannot find quantized weight for {tensor_name}: {q_name}")
    scale = np.asarray(init[node.input[1]], dtype=np.float32)
    zp = np.asarray(init[node.input[2]])
    axis = attr_int(node, "axis", 1)
    prequant = np.asarray(init[q_name], dtype=np.float32)
    # ModelOpt keeps the pre-quantized FP32 weight and an explicit Q node;
    # ORT QDQ keeps an INT8 initializer directly under DQ.
    if explicit_q:
        shape = [1] * prequant.ndim
        if scale.size > 1:
            shape[axis if axis >= 0 else prequant.ndim + axis] = scale.size
        sc = scale.reshape(shape) if scale.size > 1 else scale
        zpv = zp.astype(np.float32).reshape(shape) if scale.size > 1 else zp.astype(np.float32)
        q = np.clip(np.rint(prequant / sc + zpv), -127, 127).astype(np.int8)
    else:
        q = np.asarray(init[q_name])
    if scale.size == 1:
        deq = (q.astype(np.float32) - zp.astype(np.float32)) * scale.astype(np.float32)
    else:
        shape = [1] * q.ndim
        shape[axis if axis >= 0 else q.ndim + axis] = scale.size
        deq = (q.astype(np.float32) - zp.astype(np.float32).reshape(shape)) * scale.reshape(shape)
    if transposed:
        deq = deq.T
    if explicit_q:
        prequant_effective = prequant.T if transposed else prequant
    else:
        prequant_effective = None
    return deq, q, scale, zp, transposed, prequant_effective


def target_nodes(model):
    return {n.name: n for n in model.graph.node if n.op_type == "MatMul" and any(x in n.name for x in TARGETS)}


def stat(ref, got, q, scale, zp, algorithm, node_name):
    ref = np.asarray(ref, dtype=np.float32).ravel()
    got = np.asarray(got, dtype=np.float32).ravel()
    q = np.asarray(q).ravel()
    d = got - ref
    cosine = float(np.dot(ref, got) / max(np.linalg.norm(ref) * np.linalg.norm(got), 1e-12))
    return {
        "algorithm": algorithm, "node_name": node_name,
        "numel": int(ref.size), "shape": list(ref.shape),
        "fp32_mean": float(np.mean(ref)), "fp32_std": float(np.std(ref)),
        "fp32_min": float(np.min(ref)), "fp32_max": float(np.max(ref)),
        "fp32_abs_p99": float(np.percentile(np.abs(ref), 99)),
        "fp32_abs_p99_9": float(np.percentile(np.abs(ref), 99.9)),
        "fp32_abs_p99_99": float(np.percentile(np.abs(ref), 99.99)),
        "quant_mean": float(np.mean(got)), "quant_std": float(np.std(got)),
        "quant_min": float(np.min(got)), "quant_max": float(np.max(got)),
        "quant_abs_p99": float(np.percentile(np.abs(got), 99)),
        "quant_abs_p99_9": float(np.percentile(np.abs(got), 99.9)),
        "quant_abs_p99_99": float(np.percentile(np.abs(got), 99.99)),
        "weight_mse": float(np.mean(d * d)), "weight_mae": float(np.mean(np.abs(d))),
        "relative_l1": float(np.mean(np.abs(d)) / max(np.mean(np.abs(ref)), 1e-12)),
        "cosine": cosine, "max_abs_error": float(np.max(np.abs(d))),
        "scale_min": float(np.min(scale)), "scale_median": float(np.median(scale)),
        "scale_max": float(np.max(scale)), "scale_count": int(scale.size),
        "quant_min_int": int(np.min(q)), "quant_max_int": int(np.max(q)),
        "saturation_ratio": float(np.mean(np.abs(q.astype(np.int16)) >= 127)),
        "zero_point_min": int(np.min(zp)), "zero_point_max": int(np.max(zp)),
    }


def main():
    fp = onnx.load(str(FP), load_external_data=True)
    fpvals = values(fp); fpnodes = target_nodes(fp)
    rows = []
    samples = {a: [] for a in MODELS}
    for algorithm, path in MODELS.items():
        model = onnx.load(str(path), load_external_data=True)
        vals = values(model); nodes = target_nodes(model)
        for name, fn in fpnodes.items():
            if name not in nodes:
                continue
            src_weight = fpvals[fn.input[1]]
            mn = nodes[name]
            deq, q, scale, zp, transposed, prequant = dq_tensor(model, mn.input[1])
            if deq.shape != src_weight.shape:
                raise RuntimeError(f"shape mismatch {name}: {src_weight.shape} vs {deq.shape}")
            # For SmoothQuant, the exported FP32 weight is already smoothed
            # before its INT8 Q/DQ pair. Report both the intended quantization
            # error (smoothed FP32 -> dequantized) and the reparameterization
            # change (original FP32 -> smoothed FP32).
            quant_ref = prequant if prequant is not None else src_weight
            r = stat(quant_ref, deq, q, scale, zp, algorithm, name)
            if prequant is not None:
                tr = prequant.astype(np.float32).ravel() - src_weight.astype(np.float32).ravel()
                r["smoothquant_transform_relative_l1"] = float(np.mean(np.abs(tr)) / max(np.mean(np.abs(src_weight)), 1e-12))
                r["smoothquant_transform_cosine"] = float(np.dot(prequant.ravel(), src_weight.ravel()) / max(np.linalg.norm(prequant.ravel()) * np.linalg.norm(src_weight.ravel()), 1e-12))
                r["original_fp32_mean"] = float(np.mean(src_weight)); r["smoothed_fp32_mean"] = float(np.mean(prequant))
                r["original_fp32_abs_p99_99"] = float(np.percentile(np.abs(src_weight), 99.99)); r["smoothed_fp32_abs_p99_99"] = float(np.percentile(np.abs(prequant), 99.99))
            else:
                r["smoothquant_transform_relative_l1"] = 0.0; r["smoothquant_transform_cosine"] = 1.0
            r["layer"] = int(name.split("/layer.")[1].split("/")[0])
            r["projection"] = next(x for x in TARGETS if x in name).replace("/attention/attention/", "").replace("/attention/output/dense", "attn_out").replace("/mlp/", "")
            r["weight_transpose_in_graph"] = transposed
            rows.append(r)
            if len(samples[algorithm]) < 6:
                samples[algorithm].append((name, src_weight.ravel(), deq.ravel()))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / "weight_layerwise_distributions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {"scope": "DINOv2 encoder 72 projection MatMul weights", "source_model": str(FP), "rows": rows}
    (HERE / "weight_distribution_comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if plt is not None:
        colors = {"MinMax": "#4575b4", "Percentile": "#d73027", "SmoothQuant": "#1a9850"}
        # Representative histograms show raw FP32 versus effective dequantized weights.
        reps = [x for x in rows if x["algorithm"] == "MinMax" and x["layer"] in (0, 8, 10) and x["projection"] in ("query/MatMul", "fc2/MatMul", "attn_out/MatMul")]
        if not reps:
            reps = rows[:6]
        fig, axes = plt.subplots(len(reps), 3, figsize=(15, max(8, 2.5 * len(reps))), squeeze=False)
        for i, r in enumerate(reps):
            name = r["node_name"]
            fpw = fpvals[fpnodes[name].input[1]].astype(np.float32).ravel()
            for j, algorithm in enumerate(("MinMax", "Percentile", "SmoothQuant")):
                rr = next(x for x in rows if x["algorithm"] == algorithm and x["node_name"] == name)
                m = onnx.load(str(MODELS[algorithm]), load_external_data=True)
                deq, _, _, _, _, _ = dq_tensor(m, target_nodes(m)[name].input[1]); deq = deq.ravel()
                lo, hi = np.percentile(np.concatenate([fpw, deq]), [0.1, 99.9]); bins = np.linspace(lo, hi, 80)
                ax = axes[i, j]; ax.hist(fpw, bins=bins, density=True, histtype="step", lw=1.4, color="black", label="FP32")
                ax.hist(deq, bins=bins, density=True, histtype="step", lw=1.2, color=colors[algorithm], label="dequantized")
                ax.set_title(f"block {r['layer']} {r['projection']}\n{algorithm}"); ax.grid(alpha=.15)
                if i == 0 and j == 0: ax.legend(fontsize=8)
        fig.suptitle("Encoder weight distribution: FP32 vs effective dequantized INT8", y=1.0); fig.tight_layout(); fig.savefig(HERE / "weight_histograms_representative.png", dpi=160); plt.close(fig)
        # SmoothQuant has an additional reparameterization stage. Show the
        # original FP32 weight, the smoothed FP32 weight before INT8, and the
        # final dequantized weight separately so the two effects are not mixed.
        sq_reps = reps[:4]
        fig, axes = plt.subplots(len(sq_reps), 1, figsize=(11, max(7, 2.6 * len(sq_reps))), squeeze=False)
        sq_model = onnx.load(str(MODELS["SmoothQuant"]), load_external_data=True); sq_nodes = target_nodes(sq_model)
        for i, r in enumerate(sq_reps):
            name = r["node_name"]; original = fpvals[fpnodes[name].input[1]].astype(np.float32).ravel()
            deq, _, _, _, _, smooth = dq_tensor(sq_model, sq_nodes[name].input[1]); deq = deq.ravel(); smooth = smooth.ravel()
            lo, hi = np.percentile(np.concatenate([original, smooth, deq]), [0.1, 99.9]); bins = np.linspace(lo, hi, 100)
            ax = axes[i, 0]; ax.hist(original, bins=bins, density=True, histtype="step", lw=1.2, color="black", label="original FP32")
            ax.hist(smooth, bins=bins, density=True, histtype="step", lw=1.2, color="#fdae61", label="SmoothQuant FP32 before INT8")
            ax.hist(deq, bins=bins, density=True, histtype="step", lw=1.2, color="#1a9850", label="SmoothQuant dequantized INT8")
            ax.set_title(f"block {r['layer']} {r['projection']}"); ax.grid(alpha=.15)
            if i == 0: ax.legend(fontsize=8)
        fig.suptitle("SmoothQuant weight stages: original → smoothed → dequantized INT8", y=1.0); fig.tight_layout(); fig.savefig(HERE / "smoothquant_weight_stages.png", dpi=160); plt.close(fig)
        # Per-layer weight error and saturation.
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for algorithm in MODELS:
            rs = [r for r in rows if r["algorithm"] == algorithm]
            ax[0].plot(range(len(rs)), [r["relative_l1"] for r in rs], '.-', label=algorithm, color=colors[algorithm])
            ax[1].plot(range(len(rs)), [r["saturation_ratio"] * 100 for r in rs], '.-', label=algorithm, color=colors[algorithm])
        ax[0].set_title("Weight relative L1 by layer"); ax[0].set_ylabel("mean |Wq-W| / mean |W|")
        ax[1].set_title("INT8 weight saturation"); ax[1].set_ylabel("|q| = 127 (%)")
        for a in ax: a.set_xlabel("encoder projection order"); a.grid(alpha=.2); a.legend()
        fig.savefig(HERE / "weight_error_and_saturation.png", dpi=160); plt.close(fig)
        # Scale distributions.
        fig, ax = plt.subplots(figsize=(10, 5))
        for algorithm in MODELS:
            rs = [r for r in rows if r["algorithm"] == algorithm]
            ax.hist(np.log10([r["scale_median"] for r in rs]), bins=30, alpha=.5, label=algorithm, color=colors[algorithm])
        ax.set_xlabel("log10 median weight scale"); ax.set_ylabel("number of projection tensors"); ax.set_title("Per-layer weight scale distribution"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(HERE / "weight_scale_distribution.png", dpi=160); plt.close(fig)
    print(f"wrote {HERE / 'weight_distribution_comparison.json'} rows={len(rows)}")


if __name__ == "__main__":
    main()
