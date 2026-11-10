#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace rfdetr {

// Owns the temporary device buffer containing the decoded interleaved RGB image
// and an optional page-locked host staging buffer. The output buffer is supplied
// by the TensorRT runtime so the CUDA kernel can write directly into the engine's
// persistent input allocation.
class CudaPreprocessWorkspace {
public:
    CudaPreprocessWorkspace() = default;
    ~CudaPreprocessWorkspace();

    CudaPreprocessWorkspace(const CudaPreprocessWorkspace&) = delete;
    CudaPreprocessWorkspace& operator=(const CudaPreprocessWorkspace&) = delete;

    void ensure_source_capacity(std::size_t bytes);
    void* source_device() const { return source_device_; }

    void ensure_pinned_host_capacity(std::size_t bytes);
    std::uint8_t* pinned_host() const { return pinned_host_; }
    void copy_to_pinned_host(const std::uint8_t* host_rgb, std::size_t bytes);

private:
    void* source_device_ = nullptr;
    std::size_t source_capacity_ = 0;
    std::uint8_t* pinned_host_ = nullptr;
    std::size_t pinned_host_capacity_ = 0;
};

// Uploads decoded interleaved RGB uint8 pixels and fuses torchvision-compatible
// bilinear resize, ImageNet normalization, and HWC -> NCHW FP32 conversion.
// The kernel is asynchronous with respect to the host and runs on `stream`.
void preprocess_rgb_cuda(const std::uint8_t* host_rgb, int source_width, int source_height,
                        float* destination_nchw, int target_width, int target_height,
                        cudaStream_t stream, CudaPreprocessWorkspace& workspace);

// Applies the same fused kernel when the decoded interleaved RGB buffer is
// already on the device (for example, an nvJPEG or camera surface output).
void preprocess_rgb_cuda_device(const std::uint8_t* device_rgb, int source_width, int source_height,
                               float* destination_nchw, int target_width, int target_height,
                               cudaStream_t stream);

}  // namespace rfdetr
