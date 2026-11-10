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
    release_persistent();
    delete context_;
    delete engine_;
    delete runtime_;
    delete logger_;
}

void TensorRTEngine::release_persistent() noexcept {
    if (persistent_graph_exec_) {
        cudaGraphExecDestroy(persistent_graph_exec_);
        persistent_graph_exec_ = nullptr;
    }
    if (persistent_graph_) {
        cudaGraphDestroy(persistent_graph_);
        persistent_graph_ = nullptr;
    }
    persistent_graph_initialized_ = false;
    if (persistent_input_device_) {
        cudaFree(persistent_input_device_);
        persistent_input_device_ = nullptr;
    }
    for (void* pointer : persistent_output_devices_) {
        if (pointer) cudaFree(pointer);
    }
    persistent_output_devices_.clear();
    if (persistent_stream_) {
        cudaStreamDestroy(persistent_stream_);
        persistent_stream_ = nullptr;
    }
    persistent_initialized_ = false;
}

void TensorRTEngine::initialize_persistent() {
    if (persistent_initialized_) return;
    const nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    check_dims(input_dims, input_name_.c_str());
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        throw std::runtime_error("setInputShape failed");
    }

    try {
        check_cuda(cudaStreamCreate(&persistent_stream_), "cudaStreamCreate(persistent)");
        check_cuda(cudaMalloc(&persistent_input_device_, volume(input_dims) * sizeof(float)),
                   "cudaMalloc(persistent input)");
        persistent_output_devices_.resize(output_names_.size(), nullptr);
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            const std::string& name = output_names_[i];
            if (engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("output '" + name + "' is not FLOAT");
            }
            const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            check_dims(dims, name.c_str());
            check_cuda(cudaMalloc(&persistent_output_devices_[i], volume(dims) * sizeof(float)),
                       "cudaMalloc(persistent output)");
        }
        if (!context_->setTensorAddress(input_name_.c_str(), persistent_input_device_)) {
            throw std::runtime_error("setTensorAddress(persistent input) failed");
        }
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            if (!context_->setTensorAddress(output_names_[i].c_str(), persistent_output_devices_[i])) {
                throw std::runtime_error("setTensorAddress(persistent output) failed");
            }
        }
        persistent_initialized_ = true;
    } catch (...) {
        release_persistent();
        throw;
    }
}

void TensorRTEngine::initialize_cuda_graph() {
    initialize_persistent();
    if (persistent_graph_initialized_) return;

    check_cuda(cudaStreamSynchronize(persistent_stream_), "cudaStreamSynchronize(graph capture precondition)");
    if (cudaStreamBeginCapture(persistent_stream_, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        throw std::runtime_error("cudaStreamBeginCapture failed");
    }
    bool capture_ended = false;
    try {
        if (!context_->enqueueV3(persistent_stream_)) {
            cudaGraph_t abandoned_graph = nullptr;
            cudaStreamEndCapture(persistent_stream_, &abandoned_graph);
            if (abandoned_graph) cudaGraphDestroy(abandoned_graph);
            capture_ended = true;
            throw std::runtime_error("enqueueV3 during CUDA Graph capture failed");
        }
        check_cuda(cudaStreamEndCapture(persistent_stream_, &persistent_graph_), "cudaStreamEndCapture");
        capture_ended = true;
        check_cuda(cudaGraphInstantiate(&persistent_graph_exec_, persistent_graph_, 0),
                   "cudaGraphInstantiate");
        persistent_graph_initialized_ = true;
    } catch (...) {
        if (!capture_ended) {
            cudaGraph_t abandoned_graph = nullptr;
            cudaStreamEndCapture(persistent_stream_, &abandoned_graph);
            if (abandoned_graph) cudaGraphDestroy(abandoned_graph);
        }
        if (persistent_graph_exec_) {
            cudaGraphExecDestroy(persistent_graph_exec_);
            persistent_graph_exec_ = nullptr;
        }
        if (persistent_graph_) {
            cudaGraphDestroy(persistent_graph_);
            persistent_graph_ = nullptr;
        }
        throw;
    }
}

void* TensorRTEngine::persistent_input_device() {
    initialize_persistent();
    return persistent_input_device_;
}

cudaStream_t TensorRTEngine::persistent_stream() const {
    if (!persistent_initialized_ || !persistent_stream_) {
        throw std::runtime_error("persistent runtime is not initialized");
    }
    return persistent_stream_;
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

std::vector<TensorOutput> TensorRTEngine::infer_persistent(const std::vector<float>& input) {
    const nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (volume(input_dims) != input.size()) throw std::invalid_argument("input element count does not match engine");
    initialize_persistent();

    std::vector<TensorOutput> outputs;
    outputs.reserve(output_names_.size());
    for (const std::string& name : output_names_) {
        const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        check_dims(dims, name.c_str());
        TensorOutput output;
        output.name = name;
        output.shape.reserve(dims.nbDims);
        for (int j = 0; j < dims.nbDims; ++j) output.shape.push_back(dims.d[j]);
        output.values.resize(volume(dims));
        outputs.push_back(std::move(output));
    }

    check_cuda(cudaMemcpyAsync(persistent_input_device_, input.data(), input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, persistent_stream_),
               "cudaMemcpyAsync(persistent input)");
    if (!context_->enqueueV3(persistent_stream_)) throw std::runtime_error("enqueueV3 failed");
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        check_cuda(cudaMemcpyAsync(outputs[i].values.data(), persistent_output_devices_[i],
                                   outputs[i].values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   persistent_stream_),
                   "cudaMemcpyAsync(persistent output)");
    }
    check_cuda(cudaStreamSynchronize(persistent_stream_), "cudaStreamSynchronize(persistent)");
    return outputs;
}

std::vector<TensorOutput> TensorRTEngine::infer_persistent_device_input() {
    initialize_persistent();
    const nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        throw std::runtime_error("setInputShape(device input) failed");
    }

    // Rebind addresses on every device-input call. Some TensorRT versions may
    // update execution-context bindings during an enqueue; explicit rebinding
    // keeps this path independent of the preceding host-input API.
    if (!context_->setTensorAddress(input_name_.c_str(), persistent_input_device_)) {
        throw std::runtime_error("setTensorAddress(device input) failed");
    }
    for (std::size_t i = 0; i < output_names_.size(); ++i) {
        if (!context_->setTensorAddress(output_names_[i].c_str(), persistent_output_devices_[i])) {
            throw std::runtime_error("setTensorAddress(device output) failed");
        }
    }

    std::vector<TensorOutput> outputs;
    outputs.reserve(output_names_.size());
    for (const std::string& name : output_names_) {
        const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        check_dims(dims, name.c_str());
        TensorOutput output;
        output.name = name;
        output.shape.reserve(dims.nbDims);
        for (int j = 0; j < dims.nbDims; ++j) output.shape.push_back(dims.d[j]);
        output.values.resize(volume(dims));
        outputs.push_back(std::move(output));
    }

    if (!context_->enqueueV3(persistent_stream_)) throw std::runtime_error("enqueueV3(device input) failed");
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        check_cuda(cudaMemcpyAsync(outputs[i].values.data(), persistent_output_devices_[i],
                                   outputs[i].values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   persistent_stream_),
                   "cudaMemcpyAsync(device output)");
    }
    check_cuda(cudaStreamSynchronize(persistent_stream_), "cudaStreamSynchronize(device input)");
    return outputs;
}

std::vector<TensorOutput> TensorRTEngine::infer_persistent_device_input_cuda_graph() {
    initialize_cuda_graph();
    std::vector<TensorOutput> outputs;
    outputs.reserve(output_names_.size());
    for (const std::string& name : output_names_) {
        const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        check_dims(dims, name.c_str());
        TensorOutput output;
        output.name = name;
        output.shape.reserve(dims.nbDims);
        for (int j = 0; j < dims.nbDims; ++j) output.shape.push_back(dims.d[j]);
        output.values.resize(volume(dims));
        outputs.push_back(std::move(output));
    }

    check_cuda(cudaGraphLaunch(persistent_graph_exec_, persistent_stream_), "cudaGraphLaunch");
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        check_cuda(cudaMemcpyAsync(outputs[i].values.data(), persistent_output_devices_[i],
                                   outputs[i].values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   persistent_stream_),
                   "cudaMemcpyAsync(graph output)");
    }
    check_cuda(cudaStreamSynchronize(persistent_stream_), "cudaStreamSynchronize(graph inference)");
    return outputs;
}

}  // namespace rfdetr
