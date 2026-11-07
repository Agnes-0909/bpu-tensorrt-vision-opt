#!/usr/bin/env python3
"""Analyze encoder MatMul input activation distributions and Q/DQ error."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "error_analysis_Conv_MatMul_compare"))
from analyze_quant_error import onnx, run_onnxrt  # noqa: E402
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
INPUTS = ROOT / "tensort_quant/polygraphy_analysis/.inputs.json"


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


def target_nodes(model):
    return {n.name: n for n in model.graph.node if n.op_type == "MatMul" and any(x in n.name for x in TARGETS)}


def producer_map(model):
    return {o: n for n in model.graph.node for o in n.output}


def input_quant_info(model, matmul):
    """Find pre-quant activation, DQ output, scale/zp and axis."""
    prod = producer_map(model); init = values(model)
    dq = prod.get(matmul.input[0])
    if dq is None or dq.op_type != "DequantizeLinear":
        raise RuntimeError(f"no activation DQ before {matmul.name}")
    q = prod.get(dq.input[0])
    if q is not None and q.op_type == "QuantizeLinear":
        pre = q.input[0]
    else:
        pre = dq.input[0]
    scale = np.asarray(init[dq.input[1]], dtype=np.float32)
    zp = np.asarray(init[dq.input[2]])
    axis = attr_int(dq, "axis", None)
    return pre, dq.output[0], scale, zp, axis


def reshape_scale(scale, value_ndim, axis):
    if scale.size == 1:
        return scale
    if axis is None:
        axis = 0
    shape = [1] * value_ndim
    shape[axis if axis >= 0 else value_ndim + axis] = scale.size
    return scale.reshape(shape)


def metrics(ref, pre, post, scale, zp, algorithm, name, layer, projection):
    ref = np.asarray(ref, dtype=np.float32); pre = np.asarray(pre, dtype=np.float32); post = np.asarray(post, dtype=np.float32)
    s = reshape_scale(scale, pre.ndim, None)
    z = reshape_scale(zp.astype(np.float32), pre.ndim, None)
    q = np.clip(np.rint(pre / s + z), -127, 127).astype(np.int8)
    local = post - pre; total = post - ref; upstream = pre - ref
    def rel(d, base): return float(np.mean(np.abs(d)) / max(float(np.mean(np.abs(base))), 1e-12))
    def cos(a, b): return float(np.dot(a.ravel(), b.ravel()) / max(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()), 1e-12))
    return {
        "algorithm": algorithm, "node_name": name, "layer": layer, "projection": projection,
        "numel": int(ref.size), "ref_abs_mean": float(np.mean(np.abs(ref))),
        "ref_abs_p99": float(np.percentile(np.abs(ref), 99)), "ref_abs_p99_9": float(np.percentile(np.abs(ref), 99.9)),
        "ref_abs_p99_99": float(np.percentile(np.abs(ref), 99.99)),
        "pre_abs_mean": float(np.mean(np.abs(pre))), "pre_abs_p99": float(np.percentile(np.abs(pre), 99)),
        "pre_abs_p99_9": float(np.percentile(np.abs(pre), 99.9)), "pre_abs_p99_99": float(np.percentile(np.abs(pre), 99.99)),
        "post_abs_mean": float(np.mean(np.abs(post))), "post_abs_p99": float(np.percentile(np.abs(post), 99)),
        "post_abs_p99_9": float(np.percentile(np.abs(post), 99.9)), "post_abs_p99_99": float(np.percentile(np.abs(post), 99.99)),
        "upstream_relative_l1": rel(upstream, ref), "upstream_cosine": cos(ref, pre),
        "local_quant_relative_l1": rel(local, pre), "local_quant_cosine": cos(pre, post),
        "effective_vs_fp32_relative_l1": rel(total, ref), "effective_vs_fp32_cosine": cos(ref, post),
        "local_mae": float(np.mean(np.abs(local))), "local_rmse": float(np.sqrt(np.mean(local * local))),
        "scale_min": float(np.min(scale)), "scale_median": float(np.median(scale)), "scale_max": float(np.max(scale)), "scale_count": int(scale.size),
        "zero_point_min": int(np.min(zp)), "zero_point_max": int(np.max(zp)),
        "saturation_ratio": float(np.mean(np.abs(q.astype(np.int16)) >= 127)),
    }


def projection(name):
    for key, label in (("/attention/attention/query/MatMul", "Q"), ("/attention/attention/key/MatMul", "K"), ("/attention/attention/value/MatMul", "V"), ("/attention/output/dense/MatMul", "AttnOut"), ("/mlp/fc1/MatMul", "FFN-fc1"), ("/mlp/fc2/MatMul", "FFN-fc2")):
        if key in name: return label
    return "unknown"


def main():
    fp = onnx.load(str(FP), load_external_data=True); fpnodes = target_nodes(fp)
    specs = {}
    for name, node in fpnodes.items():
        specs[name] = {"source": node.input[0], "layer": int(name.split("/layer.")[1].split("/")[0]), "projection": projection(name)}
    source_names = [x["source"] for x in specs.values()]
    source_out = run_onnxrt(fp, source_names, INPUTS, "fp32_activation_inputs")
    rows = []
    model_cache = {}
    for algorithm, path in MODELS.items():
        model = onnx.load(str(path), load_external_data=True); model_cache[algorithm] = model; nodes = target_nodes(model)
        info = {name: input_quant_info(model, nodes[name]) for name in specs}
        names = list(dict.fromkeys([x[0] for x in info.values()] + [x[1] for x in info.values()]))
        got = run_onnxrt(model, names, INPUTS, algorithm + "_activation_inputs")
        for name, sp in specs.items():
            pre_name, post_name, scale, zp, axis = info[name]
            r = metrics(source_out[sp["source"]], got[pre_name], got[post_name], scale, zp, algorithm, name, sp["layer"], sp["projection"])
            r["pre_tensor"] = pre_name; r["post_tensor"] = post_name; r["scale_axis"] = axis
            # SmoothQuant's pre-quant tensor includes its channel smoothing;
            # this is the intended reparameterization, not a calibration error.
            r["has_prequant_reparameterization"] = algorithm == "SmoothQuant"
            rows.append(r)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / "activation_input_layerwise_distributions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (HERE / "activation_input_distribution_comparison.json").write_text(json.dumps({"scope": "encoder MatMul input activations", "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    if plt is not None:
        colors = {"MinMax": "#4575b4", "Percentile": "#d73027", "SmoothQuant": "#1a9850"}
        fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for algorithm in MODELS:
            rs = [r for r in rows if r["algorithm"] == algorithm]
            ax[0].plot(range(len(rs)), [r["upstream_relative_l1"] for r in rs], '.-', label=algorithm, color=colors[algorithm])
            ax[1].plot(range(len(rs)), [r["local_quant_relative_l1"] for r in rs], '.-', label=algorithm, color=colors[algorithm])
            ax[2].plot(range(len(rs)), [r["effective_vs_fp32_relative_l1"] for r in rs], '.-', label=algorithm, color=colors[algorithm])
        ax[0].set_title("Upstream accumulated input error"); ax[1].set_title("Current activation Q/DQ error"); ax[2].set_title("Effective input vs FP32")
        for a in ax: a.set_xlabel("encoder projection order"); a.set_ylabel("relative L1"); a.set_yscale("symlog", linthresh=.01); a.grid(alpha=.2); a.legend()
        fig.savefig(HERE / "activation_input_error_breakdown.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for algorithm in MODELS:
            rs = [r for r in rows if r["algorithm"] == algorithm]
            ax.hist(np.log10([r["scale_median"] for r in rs]), bins=30, alpha=.5, label=algorithm, color=colors[algorithm])
        ax.set_xlabel("log10 activation scale"); ax.set_ylabel("number of projection inputs"); ax.set_title("Activation input scale distribution"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(HERE / "activation_input_scale_distribution.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for algorithm in MODELS:
            rs = [r for r in rows if r["algorithm"] == algorithm]
            ax.plot(range(len(rs)), [r["saturation_ratio"] * 100 for r in rs], '.-', label=algorithm, color=colors[algorithm])
        ax.set_xlabel("encoder projection order"); ax.set_ylabel("activation saturation |q|=127 (%)"); ax.set_title("Activation INT8 saturation"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(HERE / "activation_input_saturation.png", dpi=160); plt.close(fig)
        # Representative distributions: FP32 input, pre-quant input and DQ output.
        reps = [r for r in rows if r["algorithm"] == "MinMax" and r["layer"] in (0, 8, 10) and r["projection"] in ("Q", "AttnOut", "FFN-fc2")]
        fig, axes = plt.subplots(len(reps), 3, figsize=(15, max(8, 2.5 * len(reps))), squeeze=False)
        for i, base in enumerate(reps):
            name = base["node_name"]; sp = specs[name]; ref = source_out[sp["source"]].astype(np.float32).ravel()
            for j, algorithm in enumerate(MODELS):
                rr = next(x for x in rows if x["algorithm"] == algorithm and x["node_name"] == name)
                model = model_cache[algorithm]; node = target_nodes(model)[name]; pre_name, post_name, _, _, _ = input_quant_info(model, node)
                pre = run_onnxrt(model, [pre_name], INPUTS, algorithm + "_hist_pre")[pre_name].ravel(); post = run_onnxrt(model, [post_name], INPUTS, algorithm + "_hist_post")[post_name].ravel()
                lo, hi = np.percentile(np.concatenate([ref, pre, post]), [0.1, 99.9]); bins = np.linspace(lo, hi, 90)
                ax = axes[i, j]; ax.hist(ref, bins=bins, density=True, histtype="step", color="black", label="FP32")
                ax.hist(pre, bins=bins, density=True, histtype="step", color="#fdae61", label="pre-Q")
                ax.hist(post, bins=bins, density=True, histtype="step", color=colors[algorithm], label="post-DQ")
                ax.set_title(f"block {base['layer']} {base['projection']}\n{algorithm}"); ax.grid(alpha=.15)
                if i == 0: ax.legend(fontsize=8)
        fig.suptitle("MatMul input activation: FP32 vs pre-quant vs dequantized", y=1.0); fig.tight_layout(); fig.savefig(HERE / "activation_input_histograms_representative.png", dpi=160); plt.close(fig)
    print(f"wrote {HERE / 'activation_input_distribution_comparison.json'} rows={len(rows)}")


if __name__ == "__main__":
    main()
