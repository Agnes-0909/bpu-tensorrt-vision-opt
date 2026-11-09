#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trt_root="${TENSORRT_ROOT:-/home/rton/TensorRT/TensorRT-11.4.0.106}"
torch_root="${TORCH_ROOT:-/home/rton/miniconda3/envs/torch}"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
torch_site="${TORCH_SITE_PACKAGES:-${torch_root}/lib/python3.12/site-packages}"
torch_inc="${TORCH_INCLUDE:-${torch_site}/torch/include}"
torch_lib="${TORCH_LIB:-${torch_site}/torch/lib}"
nvcc_bin="${NVCC:-${cuda_root}/bin/nvcc}"
output="${TORCH_PLUGIN_OUTPUT:-${script_dir}/build/libtorch_sdpa_attention.so}"
abi="${TORCH_CXX11_ABI:-1}"
arch="${CUDA_ARCH:-120}"

for path in "$nvcc_bin" "$trt_root/include/NvInfer.h" "$torch_inc/ATen/ATen.h" "$torch_lib/libtorch.so"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$output")"
echo "nvcc: $nvcc_bin"
echo "TensorRT: $trt_root"
echo "PyTorch: $torch_site"
echo "output: $output"

"$nvcc_bin" -std=c++17 -O3 --use_fast_math \
    -gencode "arch=compute_${arch},code=sm_${arch}" \
    -DTORCH_FLASH -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -Xcompiler -fPIC -shared "$script_dir/fused_attention_plugin.cu" \
    -o "$output" \
    -I"$trt_root/include" \
    -I"$torch_inc" \
    -I"$torch_inc/torch/csrc/api/include" \
    -L"$trt_root/lib" \
    -L"$torch_lib" \
    -lnvinfer -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 -lcudart \
    -Xlinker -rpath -Xlinker "$torch_lib"

echo "built: $output"
