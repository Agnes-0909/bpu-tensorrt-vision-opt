#include "preprocess.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rfdetr {

RgbImage load_image(const std::string& path) {
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("OpenCV could not read image: " + path);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    RgbImage image;
    image.width = rgb.cols;
    image.height = rgb.rows;
    image.pixels.assign(rgb.ptr<std::uint8_t>(), rgb.ptr<std::uint8_t>() + rgb.total() * rgb.elemSize());
    return image;
}

std::vector<float> preprocess(const RgbImage& image, int target_width, int target_height) {
    if (image.width <= 0 || image.height <= 0 || image.pixels.size() !=
        static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::invalid_argument("invalid RGB image");
    }
    if (target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("target image dimensions must be positive");
    }

    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> output(static_cast<std::size_t>(3) * target_height * target_width);

    // This is the align_corners=False bilinear rule used by torchvision's
    // tensor resize with antialias disabled.
    for (int y = 0; y < target_height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * image.height / target_height - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, image.height - 1);
        const int y1 = std::clamp(y0 + 1, 0, image.height - 1);
        const float wy = std::clamp(source_y - std::floor(source_y), 0.0f, 1.0f);
        for (int x = 0; x < target_width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * image.width / target_width - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, image.width - 1);
            const int x1 = std::clamp(x0 + 1, 0, image.width - 1);
            const float wx = std::clamp(source_x - std::floor(source_x), 0.0f, 1.0f);
            const std::size_t p00 = (static_cast<std::size_t>(y0) * image.width + x0) * 3;
            const std::size_t p01 = (static_cast<std::size_t>(y0) * image.width + x1) * 3;
            const std::size_t p10 = (static_cast<std::size_t>(y1) * image.width + x0) * 3;
            const std::size_t p11 = (static_cast<std::size_t>(y1) * image.width + x1) * 3;
            for (int c = 0; c < 3; ++c) {
                const float top = image.pixels[p00 + c] * (1.0f - wx) + image.pixels[p01 + c] * wx;
                const float bottom = image.pixels[p10 + c] * (1.0f - wx) + image.pixels[p11 + c] * wx;
                const float value = (top * (1.0f - wy) + bottom * wy) / 255.0f;
                output[(static_cast<std::size_t>(c) * target_height + y) * target_width + x] =
                    (value - kMean[c]) / kStd[c];
            }
        }
    }
    return output;
}

}  // namespace rfdetr
