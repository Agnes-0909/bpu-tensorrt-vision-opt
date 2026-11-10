#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
opencv_dir="${OpenCV_DIR:-/home/rton/opencv-4.9.0-install/lib/cmake/opencv4}"
cuda_architectures="${CUDA_ARCHITECTURES:-120}"
cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="${opencv_dir}" -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}"
cmake --build "${build_dir}" -j"$(nproc)"
echo "Built ${build_dir}/rfdetr_trt, ${build_dir}/rfdetr_benchmark, ${build_dir}/rfdetr_preprocess_benchmark, ${build_dir}/rfdetr_coco_cuda_preprocess, and ${build_dir}/rfdetr_nvjpeg_preprocess_benchmark"
