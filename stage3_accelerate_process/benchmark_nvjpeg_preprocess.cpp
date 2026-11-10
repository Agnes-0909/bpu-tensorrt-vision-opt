#include "cuda_preprocess.h"
#include "preprocess.h"

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
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
void check_nvjpeg(nvjpegStatus_t status, const char* expression) {
    if (status != NVJPEG_STATUS_SUCCESS) throw std::runtime_error(std::string(expression) + " failed: status=" + std::to_string(status));
}
double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
void print_stats(const char* name, const std::vector<double>& values) {
    std::vector<double> ordered = values;
    std::sort(ordered.begin(), ordered.end());
    const auto percentile = [&ordered](double p) {
        const double pos = p * static_cast<double>(ordered.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = std::min(ordered.size() - 1, lo + 1);
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo);
    };
    std::cout << name << " mean=" << std::accumulate(values.begin(), values.end(), 0.0) / values.size()
              << " ms P95=" << percentile(0.95) << " ms P99=" << percentile(0.99) << " ms\n";
}
std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open JPEG: " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) throw std::runtime_error("cannot read JPEG: " + path);
    return data;
}
}

int main(int argc, char** argv) {
    try {
        const std::string image_path = argc > 1 ? argv[1] : "/home/rton/val2017/val2017/000000082085.jpg";
        const int target_width = argc > 2 ? std::stoi(argv[2]) : 700;
        const int target_height = argc > 3 ? std::stoi(argv[3]) : 700;
        const int iterations = argc > 4 ? std::stoi(argv[4]) : 60;
        const int warmup = argc > 5 ? std::stoi(argv[5]) : 20;
        if (iterations <= 0 || warmup < 0) throw std::invalid_argument("iterations must be positive and warmup non-negative");

        const auto jpeg = read_file(image_path);
        const rfdetr::RgbImage cpu_image = rfdetr::load_image(image_path);
        const std::vector<float> cpu = rfdetr::preprocess(cpu_image, target_width, target_height);
        float* device_output = nullptr;
        unsigned char* device_rgb = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), cpu.size() * sizeof(float)), "cudaMalloc(output)");
        nvjpegHandle_t handle = nullptr;
        nvjpegJpegState_t state = nullptr;
        check_nvjpeg(nvjpegCreateSimple(&handle), "nvjpegCreateSimple");
        check_nvjpeg(nvjpegJpegStateCreate(handle, &state), "nvjpegJpegStateCreate");
        int channels = 0;
        int source_width = 0;
        int source_height = 0;
        nvjpegChromaSubsampling_t subsampling{};
        int widths[NVJPEG_MAX_COMPONENT] = {};
        int heights[NVJPEG_MAX_COMPONENT] = {};
        check_nvjpeg(nvjpegGetImageInfo(handle, jpeg.data(), jpeg.size(), &channels, &subsampling, widths, heights),
                     "nvjpegGetImageInfo");
        source_width = widths[0];
        source_height = heights[0];
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_rgb), static_cast<std::size_t>(source_width) * source_height * 3),
                   "cudaMalloc(RGB)");
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        rfdetr::CudaPreprocessWorkspace workspace;
        nvjpegImage_t decoded{};
        decoded.channel[0] = device_rgb;
        decoded.pitch[0] = source_width * 3;

        auto run_once = [&]() {
            check_nvjpeg(nvjpegDecode(handle, state, jpeg.data(), jpeg.size(), NVJPEG_OUTPUT_RGBI, &decoded, stream),
                         "nvjpegDecode");
            rfdetr::preprocess_rgb_cuda_device(device_rgb, source_width, source_height, device_output,
                                               target_width, target_height, stream);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(nvjpeg)");
        };
        for (int i = 0; i < warmup; ++i) run_once();
        std::vector<double> nvjpeg_times;
        nvjpeg_times.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            const auto begin = Clock::now();
            run_once();
            nvjpeg_times.push_back(elapsed_ms(begin, Clock::now()));
        }
        std::vector<float> gpu(cpu.size());
        check_cuda(cudaMemcpy(gpu.data(), device_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(output)");
        double max_abs = 0.0;
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            const double difference = static_cast<double>(cpu[i]) - gpu[i];
            max_abs = std::max(max_abs, std::abs(difference));
            sum_sq += difference * difference;
        }
        std::cout << std::fixed << std::setprecision(9)
                  << "image=" << image_path << " source=" << source_width << "x" << source_height
                  << " target=" << target_width << "x" << target_height << " channels=" << channels << "\n"
                  << "max_abs=" << max_abs << " rmse=" << std::sqrt(sum_sq / cpu.size()) << "\n";
        print_stats("nvjpeg_decode_plus_preprocess", nvjpeg_times);
        std::cout << "note=includes JPEG decode, GPU RGB output, and fused resize/normalize/layout; file read is excluded\n";
        cudaFree(device_rgb);
        cudaFree(device_output);
        cudaStreamDestroy(stream);
        nvjpegJpegStateDestroy(state);
        nvjpegDestroy(handle);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
