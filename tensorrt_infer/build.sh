#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
opencv_dir="${OpenCV_DIR:-/home/rton/opencv-4.9.0-install/lib/cmake/opencv4}"
cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="${opencv_dir}"
cmake --build "${build_dir}" -j"$(nproc)"
echo "Built ${build_dir}/rfdetr_trt and ${build_dir}/rfdetr_benchmark"
