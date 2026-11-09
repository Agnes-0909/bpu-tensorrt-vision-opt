#!/usr/bin/env python3
"""Replace encoder QK^T/Softmax/AV triplets with a TensorRT plugin node.

The plugin consumes the already scaled Q/K tensors and the value tensor in
the original [B,H,S,D] layout.  Keeping the scale nodes in the graph makes
the rewrite numerically equivalent to the exported graph and keeps this
script independent of the earlier ONNX Attention experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


PREFIX = "/backbone/backbone.0/encoder/encoder/encoder/layer."


def rewrite(model: onnx.ModelProto, expected_layers: int = 12, half: bool = False, torch_flash: bool = False) -> int:
    nodes = list(model.graph.node)
    by_name = {n.name: n for n in nodes}
    removed: set[str] = set()
    replacements: dict[str, list[onnx.NodeProto]] = {}
    for layer in range(expected_layers):
        prefix = f"{PREFIX}{layer}/attention/attention/"
        qk = by_name[prefix + "MatMul"]
        softmax = by_name[prefix + "Softmax"]
        av = by_name[prefix + "MatMul_1"]
        if list(softmax.input) != list(qk.output):
            raise RuntimeError(f"unexpected Softmax input for layer {layer}")
        if list(av.input) != [softmax.output[0], prefix + "Transpose_output_0"]:
            raise RuntimeError(f"unexpected AV inputs for layer {layer}: {list(av.input)}")
        if half:
            q_half, k_half, v_half = (prefix + n for n in ("Plugin_Q_half", "Plugin_K_half", "Plugin_V_half"))
            plugin_out = prefix + "Plugin_output_half"
            k_source = qk.input[1]
            k_nodes: list[onnx.NodeProto] = []
            if torch_flash:
                k_transposed = prefix + "Plugin_K_transposed"
                k_nodes.append(helper.make_node(
                    "Transpose", [qk.input[1]], [k_transposed], prefix + "Plugin_K_Transpose", perm=[0, 1, 3, 2]
                ))
                k_source = k_transposed
            replacements[av.name] = [
                helper.make_node("Cast", [qk.input[0]], [q_half], prefix + "Plugin_Q_Cast", to=10),
                *k_nodes,
                helper.make_node("Cast", [k_source], [k_half], prefix + "Plugin_K_Cast", to=10),
                helper.make_node("Cast", [av.input[1]], [v_half], prefix + "Plugin_V_Cast", to=10),
                helper.make_node(
                    "RFDetrTorchFlashAttention" if torch_flash else "RFDetrFusedAttention",
                    [q_half, k_half, v_half], [plugin_out],
                    prefix + "RFDetrFusedAttention"
                ),
                helper.make_node("Cast", [plugin_out], [av.output[0]], prefix + "Plugin_Output_Cast", to=1),
            ]
        else:
            replacements[av.name] = [
                helper.make_node(
                    "w" if torch_flash else "RFDetrFusedAttention",
                    inputs=[qk.input[0], qk.input[1], av.input[1]],
                    outputs=[av.output[0]],
                    name=prefix + "RFDetrFusedAttention",
                )
            ]
        removed.update((qk.name, softmax.name, av.name))

    final = []
    for node in nodes:
        if node.name in removed:
            if node.name in replacements:
                final.extend(replacements[node.name])
            continue
        final.append(node)
    del model.graph.node[:]
    model.graph.node.extend(final)
    return expected_layers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--half", action="store_true", help="cast Q/K/V to FP16 around the plugin")
    p.add_argument("--torch-flash", action="store_true", help="use the libtorch SDPA plugin name")
    args = p.parse_args()
    model = onnx.load(str(args.input), load_external_data=True)
    count = rewrite(model, half=args.half, torch_flash=args.torch_flash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)
    print(f"replaced encoder attention blocks: {count} (half={args.half}, torch_flash={args.torch_flash})")
    print(f"wrote: {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
