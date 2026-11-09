#!/usr/bin/env python3
"""Replace RF-DETR encoder self-attention subgraphs with ONNX Attention.

The current RF-DETR export contains twelve identical encoder attention blocks:
Q/K/V projections, reshape/transpose, explicit scale, MatMul, Softmax and
MatMul.  This script replaces only the QK^T/Softmax/AV portion.  Projection,
output projection, residual and SmoothQuant Q/DQ nodes are preserved.

The script is deliberately conservative and writes a new ONNX file.  It does
not modify the source model in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


PREFIX = "/backbone/backbone.0/encoder/encoder/encoder/layer."


def node_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {node.name: node for node in model.graph.node}


def rewrite(model: onnx.ModelProto, expected_layers: int = 12, layout: str = "bhsd") -> int:
    if layout not in {"bhsd", "bshd"}:
        raise ValueError("layout must be bhsd or bshd")
    nodes = list(model.graph.node)
    by_name = node_map(model)
    new_nodes: list[onnx.NodeProto] = []
    replaced = 0
    removed_names: set[str] = set()

    for layer in range(expected_layers):
        prefix = f"{PREFIX}{layer}/attention/attention/"
        qk_name = prefix + "MatMul"
        softmax_name = prefix + "Softmax"
        av_name = prefix + "MatMul_1"
        for name in (qk_name, softmax_name, av_name):
            if name not in by_name:
                raise RuntimeError(f"missing expected attention node: {name}")
        qk, softmax, av = by_name[qk_name], by_name[softmax_name], by_name[av_name]
        if list(qk.input) != [prefix + "Mul_output_0", prefix + "Mul_1_output_0"]:
            raise RuntimeError(f"unexpected QK inputs in layer {layer}: {list(qk.input)}")
        if list(softmax.input) != [qk.output[0]]:
            raise RuntimeError(f"unexpected Softmax input in layer {layer}: {list(softmax.input)}")
        if list(av.input) != [softmax.output[0], prefix + "Transpose_output_0"]:
            raise RuntimeError(f"unexpected AV inputs in layer {layer}: {list(av.input)}")

        # The explicit exporter scale is sqrt(1/sqrt(head_dim)) on Q and K.
        # Removing both factors means Attention receives unscaled Q/K and its
        # scale attribute must be 1/sqrt(64) = 0.125.
        if layout == "bhsd":
            q_input = prefix + "Transpose_1_output_0"
            k_input = prefix + "FusedAttention_K_output_0"
            v_input = prefix + "Transpose_output_0"
            attention_output = av.output[0]
        else:
            # ONNX Attention's canonical four-dimensional layout is
            # [batch, sequence, heads, head_dim]. Keep the original AV
            # output contract by transposing the result back to [B,H,S,D].
            q_input = prefix + "Reshape_2_output_0"
            k_input = prefix + "Reshape_output_0"
            v_input = prefix + "Reshape_1_output_0"
            attention_output = prefix + "FusedAttention_BSHD_output_0"

        attention = helper.make_node(
            "Attention",
            inputs=[
                q_input,
                k_input,
                v_input,
            ],
            outputs=[attention_output],
            name=prefix + "FusedAttention",
            scale=0.125,
            q_num_heads=12,
            kv_num_heads=12,
            is_causal=0,
        )
        new_nodes.append(attention)
        if layout == "bhsd":
            new_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[prefix + "Reshape_output_0"],
                    outputs=[prefix + "FusedAttention_K_output_0"],
                    name=prefix + "FusedAttention_K_Transpose",
                    perm=[0, 2, 1, 3],
                )
            )
        else:
            new_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[attention_output],
                    outputs=[av.output[0]],
                    name=prefix + "FusedAttention_BSHD_to_BHSD",
                    perm=[0, 2, 1, 3],
                )
            )
        replaced += 1

        # Remove the explicit scale calculation and its three attention ops.
        # cleanup below removes only now-unused constants and arithmetic nodes.
        for node in nodes:
            if node.name in {qk_name, softmax_name, av_name}:
                removed_names.add(node.name)
            elif node.name in {
                prefix + suffix
                for suffix in {
                    "Shape", "Constant_3", "Constant_4", "Slice", "Cast", "Sqrt",
                    "Constant_5", "Div", "Cast_1", "Sqrt_1", "Mul", "Sqrt_2", "Mul_1",
                    "Transpose_2",
                }
            }:
                removed_names.add(node.name)

    inserted = False
    for node in nodes:
        if node.name in removed_names:
            continue
        # Insert the replacement immediately before the first node in each
        # removed chain, while preserving the rest of the original ordering.
        if node.name == f"{PREFIX}0/attention/attention/Transpose_3":
            new_nodes = new_nodes[:0] + new_nodes  # keep mypy/type checkers quiet
        new_nodes.append(node)

    # The loop above appends all replacements at the beginning. Topological
    # order is still valid because Attention only consumes tensors defined by
    # earlier nodes, but place each replacement at the original AV position for
    # readable TensorRT metadata and deterministic graph order.
    replacement_by_name = {
        n.name: n for n in new_nodes if "FusedAttention" in n.name
    }
    final_nodes: list[onnx.NodeProto] = []
    emitted = 0
    for node in nodes:
        if node.name in removed_names:
            if node.name.endswith("/MatMul_1"):
                prefix = f"{PREFIX}{emitted}/attention/attention/"
                if layout == "bhsd":
                    final_nodes.append(replacement_by_name[prefix + "FusedAttention_K_Transpose"])
                final_nodes.append(replacement_by_name[prefix + "FusedAttention"])
                if layout == "bshd":
                    final_nodes.append(replacement_by_name[prefix + "FusedAttention_BSHD_to_BHSD"])
                emitted += 1
            continue
        final_nodes.append(node)
    if emitted != expected_layers:
        raise RuntimeError(f"inserted {emitted} Attention nodes, expected {expected_layers}")

    del model.graph.node[:]
    model.graph.node.extend(final_nodes)
    for imp in model.opset_import:
        if imp.domain == "":
            imp.version = max(imp.version, 23)
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("bhsd", "bshd"), default="bhsd")
    args = parser.parse_args()
    model = onnx.load(str(args.input), load_external_data=True)
    count = rewrite(model, layout=args.layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)
    print(f"replaced encoder attention blocks: {count} (layout={args.layout})")
    print(f"wrote: {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print("note: the source graph uses opset-16 attributes while ONNX Attention requires opset-23; TensorRT parser is the compatibility check")


if __name__ == "__main__":
    main()
