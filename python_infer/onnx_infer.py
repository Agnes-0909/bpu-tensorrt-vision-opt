import onnxruntime as ort
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("/home/rton/rfdetr/python_infer/output/rfdetr-nano.onnx")

# Prepare input image
input_height, input_width = session.get_inputs()[0].shape[2:4]
image = Image.open("/home/rton/rfdetr/img/000000085772.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

# Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

# Convert to NCHW format
image_array = image_tensor.unsqueeze(0).numpy()

# Run inference
outputs = session.run(None, {"input": image_array})

# Match outputs by name — do NOT assume positional order or infer role from shape.
output_names = [out.name for out in session.get_outputs()]
boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), None)
logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), None)
if boxes_idx is None or logits_idx is None:
    raise ValueError(f"Could not find expected outputs 'dets'/'labels'. Available outputs: {output_names}")

boxes_cwh = outputs[boxes_idx][0]  # (num_queries, 4) normalized cxcywh
raw_logits = outputs[logits_idx][0]

# Select this from the checkpoint layout. Use None for official sparse-ID COCO
# checkpoints, -1 for contiguous-ID/active-first checkpoints, or 0 for legacy
# background-first keypoint checkpoints.
background_class_id = -1
class_slots = np.arange(raw_logits.shape[-1])
if background_class_id is None:
    logits = raw_logits
else:
    num_slots = raw_logits.shape[-1]
    if not -num_slots <= background_class_id < num_slots:
        raise ValueError(f"background_class_id must index one of {num_slots} exported class slots")
    background_class_id %= num_slots
    foreground_mask = class_slots != background_class_id
    logits = raw_logits[:, foreground_mask]
    class_slots = class_slots[foreground_mask]

# RF-DETR uses per-class sigmoid (multi-label), not softmax. This compact example keeps
# one top class per query; the reference decoders instead rank query/class pairs globally,
# so they can retain multiple above-threshold classes for one query.
scores_all = 1.0 / (1.0 + np.exp(-logits.clip(-88, 88)))
scores = scores_all.max(axis=-1)
class_ids = class_slots[scores_all.argmax(axis=-1)]

threshold = 0.5
keep = scores > threshold

# cxcywh (normalized) -> xyxy (pixel space)
cx, cy, bw, bh = boxes_cwh[keep].T
xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
xyxy *= np.array([image.width, image.height, image.width, image.height], dtype=np.float32)

boxes, labels, confidences = xyxy, class_ids[keep], scores[keep]