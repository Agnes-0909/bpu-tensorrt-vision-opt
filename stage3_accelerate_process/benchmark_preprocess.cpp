#include "cuda_preprocess.h"
#include "preprocess.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(status));
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const double position = p * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = std::min(values.size() - 1, lower + 1);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

void print_timing(const char* name, const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::cout << name << " mean=" << mean << " ms P95=" << percentile(values, 0.95)
              << " ms P99=" << percentile(values, 0.99) << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string image_path = argc > 1 ? argv[1] : "/home/rton/rfdetr/img/000000085772.jpg";
        const int target_width = argc > 2 ? std::stoi(argv[2]) : 700;
        const int target_height = argc > 3 ? std::stoi(argv[3]) : 700;
        const int iterations = argc > 4 ? std::stoi(argv[4]) : 100;
        const int warmup = argc > 5 ? std::stoi(argv[5]) : 30;
        if (iterations <= 0 || warmup < 0) throw std::invalid_argument("iterations must be positive and warmup non-negative");

        const rfdetr::RgbImage image = rfdetr::load_image(image_path);
        const std::vector<float> cpu = rfdetr::preprocess(image, target_width, target_height);
        const std::size_t output_bytes = cpu.size() * sizeof(float);
        float* device_output = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes), "cudaMalloc(output)");
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        rfdetr::CudaPreprocessWorkspace workspace;
        const std::size_t source_bytes = static_cast<std::size_t>(image.width) * image.height * 3;
        workspace.copy_to_pinned_host(image.pixels.data(), source_bytes);

        for (int i = 0; i < warmup; ++i) {
            rfdetr::preprocess_rgb_cuda(image.pixels.data(), image.width, image.height, device_output,
                                        target_width, target_height, stream, workspace);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
            rfdetr::preprocess_rgb_cuda(workspace.pinned_host(), image.width, image.height, device_output,
                                        target_width, target_height, stream, workspace);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pinned warmup)");
        }

        std::vector<float> gpu(cpu.size());
        rfdetr::preprocess_rgb_cuda(image.pixels.data(), image.width, image.height, device_output,
                                    target_width, target_height, stream, workspace);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(reference)");
        check_cuda(cudaMemcpy(gpu.data(), device_output, output_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(reference output)");
        rfdetr::preprocess_rgb_cuda(workspace.pinned_host(), image.width, image.height, device_output,
                                    target_width, target_height, stream, workspace);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pinned reference)");
        std::vector<float> pinned_gpu(cpu.size());
        check_cuda(cudaMemcpy(pinned_gpu.data(), device_output, output_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(pinned reference output)");

        double max_abs = 0.0;
        double sum_abs = 0.0;
        double sum_sq = 0.0;
        double dot = 0.0;
        double cpu_norm = 0.0;
        double gpu_norm = 0.0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            const double difference = static_cast<double>(cpu[i]) - gpu[i];
            max_abs = std::max(max_abs, std::abs(difference));
            sum_abs += std::abs(difference);
            sum_sq += difference * difference;
            dot += static_cast<double>(cpu[i]) * gpu[i];
            cpu_norm += static_cast<double>(cpu[i]) * cpu[i];
            gpu_norm += static_cast<double>(gpu[i]) * gpu[i];
        }
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "image=" << image_path << " source=" << image.width << "x" << image.height
                  << " target=" << target_width << "x" << target_height << "\n";
        std::cout << "max_abs=" << max_abs << " rmse=" << std::sqrt(sum_sq / cpu.size())
                  << " relative_l1=" << sum_abs / (std::accumulate(cpu.begin(), cpu.end(), 0.0,
                                                                     [](double total, float value) {
                                                                         return total + std::abs(static_cast<double>(value));
                                                                     }) + 1e-12)
                  << " cosine=" << dot / (std::sqrt(cpu_norm) * std::sqrt(gpu_norm) + 1e-12) << "\n";
        double pinned_max_abs = 0.0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            pinned_max_abs = std::max(pinned_max_abs, std::abs(static_cast<double>(cpu[i]) - pinned_gpu[i]));
        }
        std::cout << "pinned_max_abs=" << pinned_max_abs << "\n";

        std::vector<double> cpu_times;
        std::vector<double> pageable_gpu_times;
        std::vector<double> pinned_gpu_times;
        cpu_times.reserve(iterations);
        pageable_gpu_times.reserve(iterations);
        pinned_gpu_times.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            auto begin = Clock::now();
            static_cast<void>(rfdetr::preprocess(image, target_width, target_height));
            cpu_times.push_back(elapsed_ms(begin, Clock::now()));

            begin = Clock::now();
            rfdetr::preprocess_rgb_cuda(image.pixels.data(), image.width, image.height, device_output,
                                        target_width, target_height, stream, workspace);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(measurement)");
            pageable_gpu_times.push_back(elapsed_ms(begin, Clock::now()));

            begin = Clock::now();
            rfdetr::preprocess_rgb_cuda(workspace.pinned_host(), image.width, image.height, device_output,
                                        target_width, target_height, stream, workspace);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pinned measurement)");
            pinned_gpu_times.push_back(elapsed_ms(begin, Clock::now()));
        }
        print_timing("cpu_preprocess", cpu_times);
        print_timing("cuda_pageable", pageable_gpu_times);
        print_timing("cuda_pinned", pinned_gpu_times);
        std::cout << "note=timing includes RGB uint8 H2D plus CUDA resize/normalize/layout; image file decode and the one-time pageable-to-pinned copy are excluded\n";

        check_cuda(cudaFree(device_output), "cudaFree(output)");
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
