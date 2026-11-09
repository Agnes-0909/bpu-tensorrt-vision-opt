#!/usr/bin/env python3
"""Compare two Polygraphy --save-outputs JSON files with common metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from polygraphy.json import load_json


def outputs(path: Path):
    runs = load_json(str(path))
    runner = next(iter(runs.keys()))
    return runs[runner][0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    ref, cand = outputs(args.reference), outputs(args.candidate)
    for name in ref:
        if name not in cand:
            continue
        a, b = np.asarray(ref[name]).astype(np.float64), np.asarray(cand[name]).astype(np.float64)
        diff = a - b
        cosine = np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
        print(
            f"{name}: shape={a.shape} max_abs={np.max(np.abs(diff)):.6g} "
            f"rmse={np.sqrt(np.mean(diff * diff)):.6g} "
            f"relative_l1={np.sum(np.abs(diff))/(np.sum(np.abs(a))+1e-12):.6g} cosine={cosine:.9f}"
        )


if __name__ == "__main__":
    main()
