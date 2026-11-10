#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trt_root="${TENSORRT_ROOT:-/home/rton/TensorRT/TensorRT-11.4.0.106}"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
cutlass_root="${CUTLASS_ROOT:-/home/rton/cutlass}"
nvcc_bin="${NVCC:-${cuda_root}/bin/nvcc}"
output="${CUTLASS_PLUGIN_OUTPUT:-${script_dir}/build/libcutlass_flash_attention.so}"
arch="${CUDA_ARCH:-120}"

for path in "$nvcc_bin" "$trt_root/include/NvInfer.h" \
    "$cutlass_root/include/cutlass/cutlass.h" \
    "$cutlass_root/examples/41_fused_multi_head_attention/kernel_forward.h"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required file: $path" >&2
        echo "Clone CUTLASS outside the repository and set CUTLASS_ROOT." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$output")"
"$nvcc_bin" -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr \
    -gencode "arch=compute_${arch},code=sm_${arch}" \
    -Xcompiler -fPIC -shared "$script_dir/cutlass_flash_attention_plugin.cu" \
    -o "$output" \
    -I"$trt_root/include" \
    -I"$cutlass_root/include" \
    -I"$cutlass_root/examples/41_fused_multi_head_attention" \
    -I"$cutlass_root/tools/util/include" \
    -I"$script_dir" \
    -L"$trt_root/lib" -lnvinfer -lcudart

echo "built: $output"
