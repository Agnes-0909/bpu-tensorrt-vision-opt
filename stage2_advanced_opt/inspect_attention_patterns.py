#!/usr/bin/env python3
"""Inspect ONNX attention subgraphs before attempting a graph rewrite.

This is intentionally read-only.  It reports the nearby operators for each
encoder attention block so that a rewrite to TensorRT IAttention or a custom
plugin can be reviewed against the original graph first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.onnx), load_external_data=False)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        text = " ".join([node.name, node.op_type, *node.input, *node.output])
        if "/attention/" not in text.lower():
            continue
        block = "unknown"
        for token in text.split("/"):
            if token.startswith("layer."):
                block = token
                break
        groups[block].append(f"{node.op_type:16s} {node.name}")
    print(f"graph nodes: {len(model.graph.node)}")
    print(f"attention blocks: {len(groups)}")
    for block in sorted(groups):
        print(f"\n[{block}] nodes={len(groups[block])}")
        for line in groups[block]:
            print(line)


if __name__ == "__main__":
    main()
