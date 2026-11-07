#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rfdetr {

std::vector<Detection> decode_detections(const float* boxes_cxcywh, std::size_t num_queries,
                                          const float* raw_logits, std::size_t num_classes,
                                          int image_width, int image_height, float threshold,
                                          int background_class_id) {
    if (!boxes_cxcywh || !raw_logits || image_width <= 0 || image_height <= 0 ||
        !(threshold >= 0.0f && threshold <= 1.0f)) {
        throw std::invalid_argument("invalid postprocessing arguments");
    }
    if (background_class_id < -1 || background_class_id >= static_cast<int>(num_classes)) {
        throw std::invalid_argument("background class index is outside labels output");
    }
    // Match Python's negative indexing: -1 selects the final exported slot.
    const int background_slot = background_class_id < 0
        ? static_cast<int>(num_classes) - 1
        : background_class_id;
    std::vector<Detection> detections;
    for (std::size_t query = 0; query < num_queries; ++query) {
        float best_score = -1.0f;
        int best_slot = -1;
        for (std::size_t slot = 0; slot < num_classes; ++slot) {
            if (static_cast<int>(slot) == background_slot) continue;
            const float logit = std::clamp(raw_logits[query * num_classes + slot], -88.0f, 88.0f);
            const float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > best_score) {
                best_score = score;
                best_slot = static_cast<int>(slot);
            }
        }
        if (best_slot < 0 || best_score <= threshold) continue;

        const float cx = boxes_cxcywh[query * 4 + 0];
        const float cy = boxes_cxcywh[query * 4 + 1];
        const float width = boxes_cxcywh[query * 4 + 2];
        const float height = boxes_cxcywh[query * 4 + 3];
        Detection detection;
        detection.x1 = std::clamp((cx - width * 0.5f) * image_width, 0.0f, static_cast<float>(image_width));
        detection.y1 = std::clamp((cy - height * 0.5f) * image_height, 0.0f, static_cast<float>(image_height));
        detection.x2 = std::clamp((cx + width * 0.5f) * image_width, 0.0f, static_cast<float>(image_width));
        detection.y2 = std::clamp((cy + height * 0.5f) * image_height, 0.0f, static_cast<float>(image_height));
        detection.score = best_score;
        detection.class_id = best_slot;
        detections.push_back(detection);
    }
    std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
    });
    return detections;
}

}  // namespace rfdetr
