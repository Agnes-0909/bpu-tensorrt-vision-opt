#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace rfdetr {

struct TensorOutput {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

class TensorRTEngine {
public:
    explicit TensorRTEngine(const std::string& engine_path);
    ~TensorRTEngine();

    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    std::vector<TensorOutput> infer(const std::vector<float>& input);
    // Reuses the CUDA stream and device buffers across calls. This is the
    // stage-2 runtime experiment; the original infer() path remains the
    // per-call allocation reference.
    std::vector<TensorOutput> infer_persistent(const std::vector<float>& input);
    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }

private:
    class Logger;
    Logger* logger_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::string input_name_;
    std::vector<std::string> output_names_;
    int input_width_ = 0;
    int input_height_ = 0;

    cudaStream_t persistent_stream_ = nullptr;
    void* persistent_input_device_ = nullptr;
    std::vector<void*> persistent_output_devices_;
    bool persistent_initialized_ = false;

    void release_persistent() noexcept;
    void initialize_persistent();
};

}  // namespace rfdetr
