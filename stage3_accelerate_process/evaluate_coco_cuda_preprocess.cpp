#include "cuda_preprocess.h"
#include "infer.h"
#include "preprocess.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Arguments {
    std::string engine;
    fs::path image_dir;
    fs::path output;
    int limit = 0;
    float threshold = 0.0f;
    std::string preprocess_mode = "cuda";
};

struct TimingSummary {
    double total_ms = 0.0;
    std::size_t count = 0;
    double mean_ms() const { return count == 0 ? 0.0 : total_ms / static_cast<double>(count); }
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <engine> <image_dir> <output.json> [limit] [score_threshold] [cpu|cuda|cuda_graph]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    if (argc < 4 || argc > 7) {
        print_usage(argv[0]);
        throw std::invalid_argument("invalid argument count");
    }
    Arguments args;
    args.engine = argv[1];
    args.image_dir = argv[2];
    args.output = argv[3];
    if (argc > 4) args.limit = std::stoi(argv[4]);
    if (argc > 5) args.threshold = std::stof(argv[5]);
    if (argc > 6) args.preprocess_mode = argv[6];
    if (args.limit < 0 || args.threshold < 0.0f || args.threshold > 1.0f ||
        (args.preprocess_mode != "cpu" && args.preprocess_mode != "cuda" &&
         args.preprocess_mode != "cuda_graph")) {
        throw std::invalid_argument("limit must be non-negative, threshold in [0,1], mode must be cpu, cuda, or cuda_graph");
    }
    if (!fs::is_directory(args.image_dir)) {
        throw std::invalid_argument("image directory does not exist: " + args.image_dir.string());
    }
    return args;
}

std::int64_t image_id_from_path(const fs::path& path) {
    const std::string stem = path.stem().string();
    if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return -1;
    }
    try {
        return std::stoll(stem);
    } catch (const std::exception&) {
        return -1;
    }
}

std::vector<fs::path> collect_images(const fs::path& image_dir, int limit) {
    std::vector<fs::path> paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        const auto extension = entry.path().extension().string();
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".JPG" && extension != ".JPEG") continue;
        if (image_id_from_path(entry.path()) >= 0) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return image_id_from_path(lhs) < image_id_from_path(rhs);
    });
    if (limit > 0 && paths.size() > static_cast<std::size_t>(limit)) paths.resize(static_cast<std::size_t>(limit));
    if (paths.empty()) throw std::runtime_error("no numeric JPEG images found in " + image_dir.string());
    return paths;
}

const rfdetr::TensorOutput* find_output(const std::vector<rfdetr::TensorOutput>& outputs,
                                        const std::string& name_part) {
    for (const auto& output : outputs) {
        if (output.name.find(name_part) != std::string::npos) return &output;
    }
    return nullptr;
}

// COCO category IDs are used as output slots by the existing Python evaluator.
// Keeping this list here makes the CUDA and CPU JSON paths use exactly the same
// class selection rule as evaluate_coco_tensorrt.py.
constexpr int kCocoCategoryIds[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90};

void write_image_predictions(std::ofstream& output, bool& first_prediction,
                             const std::vector<rfdetr::TensorOutput>& tensors, std::int64_t image_id,
                             int image_width, int image_height, float threshold) {
    const auto* boxes = find_output(tensors, "dets");
    const auto* labels = find_output(tensors, "labels");
    if (!boxes || !labels || boxes->shape.size() != 3 || labels->shape.size() != 3 ||
        boxes->shape[0] != 1 || boxes->shape[2] != 4 || labels->shape[0] != 1 ||
        labels->shape[1] != boxes->shape[1]) {
        throw std::runtime_error("engine outputs must be dets [1,N,4] and labels [1,N,C]");
    }

    struct Prediction {
        int category_id;
        float score;
        float x1;
        float y1;
        float x2;
        float y2;
    };
    std::vector<Prediction> predictions;
    const std::size_t num_queries = static_cast<std::size_t>(boxes->shape[1]);
    const std::size_t num_classes = static_cast<std::size_t>(labels->shape[2]);
    for (std::size_t query = 0; query < num_queries; ++query) {
        float best_score = -1.0f;
        int best_category = -1;
        for (const int category_id : kCocoCategoryIds) {
            const std::size_t slot = static_cast<std::size_t>(category_id);
            if (slot >= num_classes) continue;
            const float logit = std::clamp(labels->values[query * num_classes + slot], -88.0f, 88.0f);
            const float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > best_score) {
                best_score = score;
                best_category = category_id;
            }
        }
        if (best_category < 0 || best_score <= threshold) continue;
        const float cx = boxes->values[query * 4 + 0];
        const float cy = boxes->values[query * 4 + 1];
        const float width = boxes->values[query * 4 + 2];
        const float height = boxes->values[query * 4 + 3];
        predictions.push_back({
            best_category,
            best_score,
            std::clamp((cx - width * 0.5f) * image_width, 0.0f, static_cast<float>(image_width)),
            std::clamp((cy - height * 0.5f) * image_height, 0.0f, static_cast<float>(image_height)),
            std::clamp((cx + width * 0.5f) * image_width, 0.0f, static_cast<float>(image_width)),
            std::clamp((cy + height * 0.5f) * image_height, 0.0f, static_cast<float>(image_height)),
        });
    }
    std::sort(predictions.begin(), predictions.end(), [](const Prediction& lhs, const Prediction& rhs) {
        return lhs.score > rhs.score;
    });

    output << std::setprecision(9);
    for (const Prediction& prediction : predictions) {
        if (!first_prediction) output << ',';
        first_prediction = false;
        output << "\n  {\"image_id\":" << image_id << ",\"category_id\":" << prediction.category_id
               << ",\"bbox\":[" << prediction.x1 << ',' << prediction.y1 << ','
               << std::max(0.0f, prediction.x2 - prediction.x1) << ','
               << std::max(0.0f, prediction.y2 - prediction.y1) << "],\"score\":" << prediction.score << '}';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        const std::vector<fs::path> images = collect_images(args.image_dir, args.limit);
        rfdetr::TensorRTEngine engine(args.engine);
        std::unique_ptr<rfdetr::CudaPreprocessWorkspace> workspace;
        const bool cuda_mode = args.preprocess_mode != "cpu";
        const bool cuda_graph = args.preprocess_mode == "cuda_graph";
        if (cuda_mode) {
            engine.initialize_persistent();
            workspace = std::make_unique<rfdetr::CudaPreprocessWorkspace>();
            if (cuda_graph) engine.initialize_cuda_graph();
        }

        fs::create_directories(args.output.parent_path());
        std::ofstream output(args.output);
        if (!output) throw std::runtime_error("cannot open output: " + args.output.string());
        output << '[';
        bool first_prediction = true;
        TimingSummary preprocess_timing;
        TimingSummary inference_timing;

        for (std::size_t index = 0; index < images.size(); ++index) {
            const rfdetr::RgbImage image = rfdetr::load_image(images[index].string());
            const auto preprocess_begin = Clock::now();
            std::vector<rfdetr::TensorOutput> tensors;
            if (cuda_mode) {
                rfdetr::preprocess_rgb_cuda(
                    image.pixels.data(), image.width, image.height,
                    static_cast<float*>(engine.persistent_input_device()), engine.input_width(),
                    engine.input_height(), engine.persistent_stream(), *workspace);
                if (cudaStreamSynchronize(engine.persistent_stream()) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize(preprocess) failed");
                }
            } else {
                const auto input = rfdetr::preprocess(image, engine.input_width(), engine.input_height());
                const auto preprocess_end = Clock::now();
                preprocess_timing.total_ms += std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin).count();
                ++preprocess_timing.count;
                const auto inference_begin = Clock::now();
                tensors = engine.infer_persistent(input);
                const auto inference_end = Clock::now();
                inference_timing.total_ms += std::chrono::duration<double, std::milli>(inference_end - inference_begin).count();
                ++inference_timing.count;
                write_image_predictions(output, first_prediction, tensors, image_id_from_path(images[index]),
                                        image.width, image.height, args.threshold);
                if (index == 0 || (index + 1) % 250 == 0 || index + 1 == images.size()) {
                    std::cout << "processed " << (index + 1) << '/' << images.size() << " images\n";
                }
                continue;
            }

            const auto preprocess_end = Clock::now();
            preprocess_timing.total_ms += std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin).count();
            ++preprocess_timing.count;
            const auto inference_begin = Clock::now();
                tensors = cuda_graph ? engine.infer_persistent_device_input_cuda_graph()
                                     : engine.infer_persistent_device_input();
            const auto inference_end = Clock::now();
            inference_timing.total_ms += std::chrono::duration<double, std::milli>(inference_end - inference_begin).count();
            ++inference_timing.count;
            write_image_predictions(output, first_prediction, tensors, image_id_from_path(images[index]),
                                    image.width, image.height, args.threshold);
            if (index == 0 || (index + 1) % 250 == 0 || index + 1 == images.size()) {
                std::cout << "processed " << (index + 1) << '/' << images.size() << " images\n";
            }
        }
        output << (first_prediction ? "\n]" : "\n]\n");
        output.close();
        std::cout << std::fixed << std::setprecision(4)
                  << "preprocess_mode=" << args.preprocess_mode
                  << " images=" << images.size()
                  << " preprocess_mean_ms=" << preprocess_timing.mean_ms()
                  << " inference_mean_ms=" << inference_timing.mean_ms()
                  << " output=" << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
