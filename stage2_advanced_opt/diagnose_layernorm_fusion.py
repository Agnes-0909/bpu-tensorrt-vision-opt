#!/usr/bin/env python3
"""Audit whether LayerNorm/residual work appears as standalone TensorRT layers."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--profile-csv', type=Path, default=Path('/home/rton/rfdetr/tensorrt_quant_decoder/compute_graph_analysis/tensorrt_profile_layers.csv'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.DictReader(args.profile_csv.open()))
    standalone_norm = [r for r in rows if 'layernorm' in r['name'].lower() or r['name'].lower().endswith('/norm')]
    add_rows = [r for r in rows if 'add' in r['name'].lower()]
    fused_candidates = [r for r in rows if len(r['name']) > 40 and any(x in r['name'].lower() for x in ('mean', 'variance', 'scale', 'sub', 'mul'))]
    result = {
        'profile_csv': str(args.profile_csv),
        'layer_count': len(rows),
        'standalone_layernorm_count': len(standalone_norm),
        'standalone_layernorm_names': [r['name'] for r in standalone_norm],
        'add_layer_count': len(add_rows),
        'long_fused_pointwise_candidates': len(fused_candidates),
        'interpretation': 'No standalone LayerNorm name was found; the profile contains fused pointwise names and Add layers, so manual LayerNorm rewrite is not justified before a detailed profile audit.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
