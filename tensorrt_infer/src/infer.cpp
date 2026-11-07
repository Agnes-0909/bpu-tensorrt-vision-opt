#include "infer.h"

#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rfdetr {
namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(status));
    }
}

void check_dims(const nvinfer1::Dims& dims, const char* name) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(std::string("dynamic or invalid dimension in ") + name);
        }
    }
}

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0 || result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dims.d[i])) {
            throw std::runtime_error("invalid tensor dimensions");
        }
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

}  // namespace

class TensorRTEngine::Logger : public TrtLogger {};

TensorRTEngine::TensorRTEngine(const std::string& engine_path) {
    std::ifstream stream(engine_path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open TensorRT engine: " + engine_path);
    const std::streamsize size = stream.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + engine_path);
    stream.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    if (!stream.read(serialized.data(), size)) throw std::runtime_error("failed to read TensorRT engine");

    logger_ = new Logger();
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    initLibNvInferPlugins(logger_, "");
    engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (!input_name_.empty()) throw std::runtime_error("engine has more than one input tensor");
            input_name_ = name;
            const nvinfer1::Dims dims = engine_->getTensorShape(name);
            check_dims(dims, name);
            if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3) {
                throw std::runtime_error("expected input shape [1,3,H,W]");
            }
            input_height_ = dims.d[2];
            input_width_ = dims.d[3];
        } else {
            output_names_.emplace_back(name);
        }
    }
    if (input_name_.empty() || output_names_.empty()) throw std::runtime_error("engine has invalid I/O tensors");
    if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("this example expects a FLOAT input tensor");
    }
}

TensorRTEngine::~TensorRTEngine() {
    delete context_;
    delete engine_;
    delete runtime_;
    delete logger_;
}

std::vector<TensorOutput> TensorRTEngine::infer(const std::vector<float>& input) {
    const nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (volume(input_dims) != input.size()) throw std::invalid_argument("input element count does not match engine");
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) throw std::runtime_error("setInputShape failed");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    void* input_device = nullptr;
    std::vector<void*> output_devices(output_names_.size(), nullptr);
    std::vector<TensorOutput> outputs;
    try {
        check_cuda(cudaMalloc(&input_device, input.size() * sizeof(float)), "cudaMalloc(input)");
        check_cuda(cudaMemcpyAsync(input_device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(input)");
        if (!context_->setTensorAddress(input_name_.c_str(), input_device)) throw std::runtime_error("setTensorAddress(input) failed");

        outputs.reserve(output_names_.size());
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            const std::string& name = output_names_[i];
            if (engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("output '" + name + "' is not FLOAT");
            }
            const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            check_dims(dims, name.c_str());
            TensorOutput output;
            output.name = name;
            output.shape.reserve(dims.nbDims);
            for (int j = 0; j < dims.nbDims; ++j) output.shape.push_back(dims.d[j]);
            output.values.resize(volume(dims));
            check_cuda(cudaMalloc(&output_devices[i], output.values.size() * sizeof(float)), "cudaMalloc(output)");
            if (!context_->setTensorAddress(name.c_str(), output_devices[i])) throw std::runtime_error("setTensorAddress(output) failed");
            outputs.push_back(std::move(output));
        }

        if (!context_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            check_cuda(cudaMemcpyAsync(outputs[i].values.data(), output_devices[i],
                                       outputs[i].values.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync(output)");
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        cudaFree(input_device);
        for (void* pointer : output_devices) cudaFree(pointer);
        cudaStreamDestroy(stream);
        return outputs;
    } catch (...) {
        if (input_device) cudaFree(input_device);
        for (void* pointer : output_devices) if (pointer) cudaFree(pointer);
        cudaStreamDestroy(stream);
        throw;
    }
}

}  // namespace rfdetr
