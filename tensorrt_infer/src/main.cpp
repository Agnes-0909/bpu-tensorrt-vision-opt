#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::string engine = "model.engine";
    std::string image = "img/000000085772.jpg";
    float threshold = 0.5f;
    int background_class = -1;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [engine] [image] [threshold] [background_class]\n"
              << "Defaults are model.engine, img/000000085772.jpg, 0.5, -1.\n"
              << "Use -1 to remove the last exported class slot; use 0 to remove slot 0.\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    if (argc > 1) args.engine = argv[1];
    if (argc > 2) args.image = argv[2];
    if (argc > 3) args.threshold = std::stof(argv[3]);
    if (argc > 4) args.background_class = std::stoi(argv[4]);
    if (argc > 5) {
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
    if (!(args.threshold >= 0.0f && args.threshold <= 1.0f)) {
        throw std::invalid_argument("threshold must be in [0, 1]");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        rfdetr::TensorRTEngine engine(args.engine);
        const rfdetr::RgbImage image = rfdetr::load_image(args.image);
        const std::vector<float> input = rfdetr::preprocess(image, engine.input_width(), engine.input_height());
        const std::vector<rfdetr::TensorOutput> outputs = engine.infer(input);

        const rfdetr::TensorOutput* boxes = nullptr;
        const rfdetr::TensorOutput* labels = nullptr;
        for (const auto& output : outputs) {
            if (output.name.find("dets") != std::string::npos) boxes = &output;
            if (output.name.find("labels") != std::string::npos) labels = &output;
        }
        if (!boxes || !labels || boxes->shape.size() != 3 || labels->shape.size() != 3 ||
            boxes->shape[0] != 1 || boxes->shape[2] != 4 || labels->shape[0] != 1 ||
            labels->shape[1] != boxes->shape[1]) {
            throw std::runtime_error("engine outputs must be dets [1,N,4] and labels [1,N,C]");
        }

        const auto detections = rfdetr::decode_detections(
            boxes->values.data(), static_cast<std::size_t>(boxes->shape[1]), labels->values.data(),
            static_cast<std::size_t>(labels->shape[2]), image.width, image.height, args.threshold,
            args.background_class);
        std::cout << "detections: " << detections.size() << '\n';
        std::cout << std::fixed << std::setprecision(4);
        for (std::size_t i = 0; i < detections.size(); ++i) {
            const auto& detection = detections[i];
            std::cout << i << ": class=" << detection.class_id << " score=" << detection.score
                      << " box=[" << detection.x1 << ", " << detection.y1 << ", "
                      << detection.x2 << ", " << detection.y2 << "]\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
