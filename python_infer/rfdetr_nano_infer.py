import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRNano(pretrain_weights="./rf-detr-nano.pth")

# detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

# labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

# annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# import cv2
# res = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# cv2.imwrite("res.jpg", res)


# export the model

#
model.export(format = "onnx", opset_version=16)

# model.export(format = "tensorrt")