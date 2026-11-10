#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
torch_so="${TORCH_PLUGIN_SO:-${script_dir}/build/libtorch_sdpa_attention.so}"
cutlass_so="${CUTLASS_PLUGIN_SO:-${script_dir}/build/libcutlass_flash_attention.so}"

for so in "$torch_so" "$cutlass_so"; do
    if [[ ! -e "$so" ]]; then
        echo "missing plugin: $so" >&2
        exit 1
    fi
done

echo "[libtorch plugin] $torch_so"
ldd "$torch_so" | rg -i 'torch|c10|nvinfer|cuda|not found' || true
if ! ldd "$torch_so" | rg -q 'libtorch|libc10'; then
    echo "error: the reference PyTorch plugin no longer exposes its expected torch dependency" >&2
    exit 1
fi

echo "[standalone CUTLASS plugin] $cutlass_so"
ldd "$cutlass_so" | rg -i 'torch|c10|nvinfer|cuda|not found' || true
if ldd "$cutlass_so" | rg -q 'libtorch|libc10'; then
    echo "error: standalone CUTLASS plugin unexpectedly depends on libtorch" >&2
    exit 1
fi

echo "dependency check passed"
