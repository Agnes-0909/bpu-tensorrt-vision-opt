#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rfdetr {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;  // RGB, interleaved, row-major.
};

// Loads an image with OpenCV and stores it as RGB, interleaved, uint8 pixels.
RgbImage load_image(const std::string& path);

// Resizes to NCHW and applies torchvision-compatible ImageNet normalization.
std::vector<float> preprocess(const RgbImage& image, int target_width, int target_height);

}  // namespace rfdetr
