#include "cuda_preprocess.h"
#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments {
    std::string engine = "model.engine";
    std::string image = "img/000000085772.jpg";
    int iterations = 5000;
    int warmup = 100;
    float threshold = 0.5f;
    int background_class = -1;
    std::string runtime_mode = "per_call";
};

struct Stats {
    double mean = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

Stats summarize(std::vector<double> samples) {
    if (samples.empty()) throw std::invalid_argument("no timing samples");
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](double p) {
        const double position = p * static_cast<double>(samples.size() - 1);
        const auto lower = static_cast<std::size_t>(std::floor(position));
        const auto upper = std::min(samples.size() - 1, lower + 1);
        const double fraction = position - static_cast<double>(lower);
        return samples[lower] + (samples[upper] - samples[lower]) * fraction;
    };
    Stats stats;
    stats.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);
    stats.max = samples.back();
    return stats;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [engine] [image] [iterations] [warmup] [threshold] [background_class]"
                 " [per_call|persistent|cuda_preprocess|cuda_preprocess_pinned|cuda_preprocess_async|cuda_graph]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    if (argc > 1) args.engine = argv[1];
    if (argc > 2) args.image = argv[2];
    if (argc > 3) args.iterations = std::stoi(argv[3]);
    if (argc > 4) args.warmup = std::stoi(argv[4]);
    if (argc > 5) args.threshold = std::stof(argv[5]);
    if (argc > 6) args.background_class = std::stoi(argv[6]);
    if (argc > 7) args.runtime_mode = argv[7];
    if (argc > 8) {
        print_usage(argv[0]);
        throw std::invalid_argument("too many arguments");
    }
    if (argc <= 1) {
        for (const std::string& candidate : {std::string("model.engine"), std::string("../model.engine"),
                                              std::string("../../model.engine")}) {
            std::ifstream file(candidate);
            if (file.good()) {
                args.engine = candidate;
                break;
            }
        }
        for (const std::string& candidate : {std::string("img/000000085772.jpg"),
                                              std::string("../img/000000085772.jpg"),
                                              std::string("../../img/000000085772.jpg")}) {
            std::ifstream file(candidate);
            if (file.good()) {
                args.image = candidate;
                break;
            }
        }
    }
    if (args.iterations <= 0 || args.warmup < 0 || args.threshold < 0.0f || args.threshold > 1.0f ||
        (args.runtime_mode != "per_call" && args.runtime_mode != "persistent" &&
         args.runtime_mode != "cuda_preprocess" && args.runtime_mode != "cuda_preprocess_pinned" &&
         args.runtime_mode != "cuda_preprocess_async" && args.runtime_mode != "cuda_graph")) {
        throw std::invalid_argument(
            "iterations must be positive, warmup non-negative, threshold in [0,1], runtime mode must be "
            "per_call, persistent, cuda_preprocess, cuda_preprocess_pinned, cuda_preprocess_async, or cuda_graph");
    }
    return args;
}

const rfdetr::TensorOutput* find_output(const std::vector<rfdetr::TensorOutput>& outputs,
                                        const std::string& name_part) {
    for (const auto& output : outputs) {
        if (output.name.find(name_part) != std::string::npos) return &output;
    }
    return nullptr;
}

std::size_t decode(const std::vector<rfdetr::TensorOutput>& outputs, const rfdetr::RgbImage& image,
                   float threshold, int background_class) {
    const auto* boxes = find_output(outputs, "dets");
    const auto* labels = find_output(outputs, "labels");
    if (!boxes || !labels || boxes->shape.size() != 3 || labels->shape.size() != 3 ||
        boxes->shape[0] != 1 || boxes->shape[2] != 4 || labels->shape[0] != 1 ||
        labels->shape[1] != boxes->shape[1]) {
        throw std::runtime_error("engine outputs must be dets [1,N,4] and labels [1,N,C]");
    }
    return rfdetr::decode_detections(
               boxes->values.data(), static_cast<std::size_t>(boxes->shape[1]), labels->values.data(),
               static_cast<std::size_t>(labels->shape[2]), image.width, image.height, threshold,
               background_class)
        .size();
}

void print_stats(const char* name, const Stats& stats) {
    std::cout << std::left << std::setw(14) << name << "mean=" << std::setw(10) << stats.mean
              << " ms P95=" << std::setw(10) << stats.p95 << " ms P99=" << std::setw(10)
              << stats.p99 << " ms Max=" << stats.max << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        rfdetr::TensorRTEngine engine(args.engine);
        const rfdetr::RgbImage image = rfdetr::load_image(args.image);
        const bool cuda_preprocess = args.runtime_mode == "cuda_preprocess" ||
                                     args.runtime_mode == "cuda_preprocess_pinned" ||
                                     args.runtime_mode == "cuda_preprocess_async" ||
                                     args.runtime_mode == "cuda_graph";
        const bool cuda_preprocess_pinned = args.runtime_mode == "cuda_preprocess_pinned";
        const bool cuda_graph = args.runtime_mode == "cuda_graph";
        const bool cuda_preprocess_async = args.runtime_mode == "cuda_preprocess_async";
        std::unique_ptr<rfdetr::CudaPreprocessWorkspace> cuda_workspace;
        if (cuda_preprocess) {
            engine.initialize_persistent();
            cuda_workspace = std::make_unique<rfdetr::CudaPreprocessWorkspace>();
            if (cuda_graph) engine.initialize_cuda_graph();
            if (cuda_preprocess_pinned) {
                const std::size_t source_bytes = static_cast<std::size_t>(image.width) * image.height * 3;
                cuda_workspace->copy_to_pinned_host(image.pixels.data(), source_bytes);
            }
        }

        const std::uint8_t* cuda_host_rgb = cuda_preprocess_pinned
            ? cuda_workspace->pinned_host()
            : image.pixels.data();

        auto run_warmup = [&]() {
            if (cuda_preprocess) {
                rfdetr::preprocess_rgb_cuda(
                    cuda_host_rgb, image.width, image.height,
                    static_cast<float*>(engine.persistent_input_device()), engine.input_width(),
                    engine.input_height(), engine.persistent_stream(), *cuda_workspace);
                if (!cuda_preprocess_async && cudaStreamSynchronize(engine.persistent_stream()) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize(preprocess warmup) failed");
                }
                static_cast<void>(cuda_graph ? engine.infer_persistent_device_input_cuda_graph()
                                             : engine.infer_persistent_device_input());
            } else {
                const auto input = rfdetr::preprocess(image, engine.input_width(), engine.input_height());
                if (args.runtime_mode == "persistent") {
                    static_cast<void>(engine.infer_persistent(input));
                } else {
                    static_cast<void>(engine.infer(input));
                }
            }
        };
        for (int i = 0; i < args.warmup; ++i) run_warmup();

        std::vector<double> preprocess_times;
        std::vector<double> inference_times;
        std::vector<double> postprocess_times;
        std::vector<double> end_to_end_times;
        preprocess_times.reserve(args.iterations);
        inference_times.reserve(args.iterations);
        postprocess_times.reserve(args.iterations);
        end_to_end_times.reserve(args.iterations);
        std::size_t detection_count = 0;

        for (int i = 0; i < args.iterations; ++i) {
            std::vector<rfdetr::TensorOutput> outputs;
            const auto preprocess_begin = Clock::now();
            Clock::time_point preprocess_end;
            Clock::time_point inference_begin;
            Clock::time_point inference_end;
            if (cuda_preprocess) {
                rfdetr::preprocess_rgb_cuda(
                    cuda_host_rgb, image.width, image.height,
                    static_cast<float*>(engine.persistent_input_device()), engine.input_width(),
                    engine.input_height(), engine.persistent_stream(), *cuda_workspace);
                if (!cuda_preprocess_async && cudaStreamSynchronize(engine.persistent_stream()) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize(preprocess) failed");
                }
                preprocess_end = Clock::now();
                inference_begin = Clock::now();
                outputs = cuda_graph ? engine.infer_persistent_device_input_cuda_graph()
                                     : engine.infer_persistent_device_input();
                inference_end = Clock::now();
            } else {
                const auto input = rfdetr::preprocess(image, engine.input_width(), engine.input_height());
                preprocess_end = Clock::now();
                inference_begin = Clock::now();
                outputs = args.runtime_mode == "persistent" ? engine.infer_persistent(input)
                                                              : engine.infer(input);
                inference_end = Clock::now();
            }

            const auto postprocess_begin = Clock::now();
            detection_count += decode(outputs, image, args.threshold, args.background_class);
            const auto postprocess_end = Clock::now();
            preprocess_times.push_back(elapsed_ms(preprocess_begin, preprocess_end));
            inference_times.push_back(elapsed_ms(inference_begin, inference_end));
            if (cuda_preprocess_async) {
                end_to_end_times.push_back(elapsed_ms(preprocess_begin, inference_end));
            }
            postprocess_times.push_back(elapsed_ms(postprocess_begin, postprocess_end));
        }

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "engine=" << args.engine << " image=" << args.image << '\n';
        std::cout << "iterations=" << args.iterations << " warmup=" << args.warmup
                  << " average_detections=" << static_cast<double>(detection_count) / args.iterations << '\n';
        std::cout << "runtime_mode=" << args.runtime_mode << '\n';
        std::cout << "timing unit: milliseconds; cuda_preprocess includes RGB H2D plus fused resize/normalize/layout; "
                     "inference includes TensorRT and D2H; cuda_preprocess_async inference also waits for queued "
                     "preprocess; per_call includes CUDA allocations, persistent reuses "
                     "stream and device buffers\n";
        print_stats("preprocess", summarize(std::move(preprocess_times)));
        print_stats("inference", summarize(std::move(inference_times)));
        if (cuda_preprocess_async) print_stats("end_to_end", summarize(std::move(end_to_end_times)));
        print_stats("postprocess", summarize(std::move(postprocess_times)));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
