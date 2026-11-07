#pragma once

#include <cstddef>
#include <vector>

namespace rfdetr {

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
};

std::vector<Detection> decode_detections(const float* boxes_cxcywh, std::size_t num_queries,
                                          const float* raw_logits, std::size_t num_classes,
                                          int image_width, int image_height, float threshold,
                                          int background_class_id);

}  // namespace rfdetr
