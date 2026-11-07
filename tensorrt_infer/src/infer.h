#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

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
};

}  // namespace rfdetr
