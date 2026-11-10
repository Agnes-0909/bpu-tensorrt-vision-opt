#include "cuda_preprocess.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace rfdetr {
namespace {

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ std::uint8_t load_rgb(const std::uint8_t* source, int width, int x, int y, int channel) {
    return source[(static_cast<std::size_t>(y) * width + x) * 3 + channel];
}

__global__ void resize_normalize_nchw_kernel(const std::uint8_t* source, int source_width, int source_height,
                                              float* output, int target_width, int target_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= target_width || y >= target_height) return;

    const float source_y = (static_cast<float>(y) + 0.5f) * source_height / target_height - 0.5f;
    const float source_x = (static_cast<float>(x) + 0.5f) * source_width / target_width - 0.5f;
    const float floor_y = floorf(source_y);
    const float floor_x = floorf(source_x);
    const int y0 = max(0, min(static_cast<int>(floor_y), source_height - 1));
    const int x0 = max(0, min(static_cast<int>(floor_x), source_width - 1));
    const int y1 = min(y0 + 1, source_height - 1);
    const int x1 = min(x0 + 1, source_width - 1);
    const float wy = fminf(fmaxf(source_y - floor_y, 0.0f), 1.0f);
    const float wx = fminf(fmaxf(source_x - floor_x, 0.0f), 1.0f);

    for (int channel = 0; channel < 3; ++channel) {
        const float p00 = static_cast<float>(load_rgb(source, source_width, x0, y0, channel));
        const float p01 = static_cast<float>(load_rgb(source, source_width, x1, y0, channel));
        const float p10 = static_cast<float>(load_rgb(source, source_width, x0, y1, channel));
        const float p11 = static_cast<float>(load_rgb(source, source_width, x1, y1, channel));
        const float top = p00 * (1.0f - wx) + p01 * wx;
        const float bottom = p10 * (1.0f - wx) + p11 * wx;
        const float value = (top * (1.0f - wy) + bottom * wy) / 255.0f;
        const float mean = channel == 0 ? 0.485f : (channel == 1 ? 0.456f : 0.406f);
        const float standard_deviation = channel == 0 ? 0.229f : (channel == 1 ? 0.224f : 0.225f);
        output[(static_cast<std::size_t>(channel) * target_height + y) * target_width + x] =
            (value - mean) / standard_deviation;
    }
}

}  // namespace

CudaPreprocessWorkspace::~CudaPreprocessWorkspace() {
    if (source_device_) cudaFree(source_device_);
    if (pinned_host_) cudaFreeHost(pinned_host_);
}

void CudaPreprocessWorkspace::ensure_source_capacity(std::size_t bytes) {
    if (bytes <= source_capacity_) return;
    if (source_device_) {
        check_cuda(cudaFree(source_device_), "cudaFree(source RGB)");
        source_device_ = nullptr;
        source_capacity_ = 0;
    }
    check_cuda(cudaMalloc(&source_device_, bytes), "cudaMalloc(source RGB)");
    source_capacity_ = bytes;
}

void CudaPreprocessWorkspace::ensure_pinned_host_capacity(std::size_t bytes) {
    if (bytes <= pinned_host_capacity_) return;
    if (pinned_host_) {
        check_cuda(cudaFreeHost(pinned_host_), "cudaFreeHost(pinned RGB)");
        pinned_host_ = nullptr;
        pinned_host_capacity_ = 0;
    }
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&pinned_host_), bytes, cudaHostAllocPortable),
               "cudaHostAlloc(pinned RGB)");
    pinned_host_capacity_ = bytes;
}

void CudaPreprocessWorkspace::copy_to_pinned_host(const std::uint8_t* host_rgb, std::size_t bytes) {
    if (!host_rgb || bytes == 0) throw std::invalid_argument("invalid pinned RGB copy arguments");
    ensure_pinned_host_capacity(bytes);
    std::memcpy(pinned_host_, host_rgb, bytes);
}

void preprocess_rgb_cuda(const std::uint8_t* host_rgb, int source_width, int source_height,
                         float* destination_nchw, int target_width, int target_height,
                         cudaStream_t stream, CudaPreprocessWorkspace& workspace) {
    if (!host_rgb || !destination_nchw || source_width <= 0 || source_height <= 0 || target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("invalid CUDA preprocess arguments");
    }
    const std::size_t source_bytes = static_cast<std::size_t>(source_width) * source_height * 3;
    workspace.ensure_source_capacity(source_bytes);
    check_cuda(cudaMemcpyAsync(workspace.source_device(), host_rgb, source_bytes, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(source RGB)");

    preprocess_rgb_cuda_device(static_cast<const std::uint8_t*>(workspace.source_device()), source_width,
                               source_height, destination_nchw, target_width, target_height, stream);
}

void preprocess_rgb_cuda_device(const std::uint8_t* device_rgb, int source_width, int source_height,
                               float* destination_nchw, int target_width, int target_height,
                               cudaStream_t stream) {
    if (!device_rgb || !destination_nchw || source_width <= 0 || source_height <= 0 || target_width <= 0 ||
        target_height <= 0) {
        throw std::invalid_argument("invalid CUDA device preprocess arguments");
    }

    constexpr dim3 block(16, 16);
    const dim3 grid((target_width + block.x - 1) / block.x, (target_height + block.y - 1) / block.y);
    resize_normalize_nchw_kernel<<<grid, block, 0, stream>>>(
        device_rgb, source_width, source_height, destination_nchw, target_width, target_height);
    check_cuda(cudaGetLastError(), "resize_normalize_nchw_kernel");
}

}  // namespace rfdetr
