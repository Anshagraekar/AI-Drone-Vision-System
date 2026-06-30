from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,     # keeps track IDs
            imgsz=416,
            conf=0.2,
            iou=0.4
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            track_id = int(box.id[0]) if box.id is not None else -1

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "confidence": conf,
                "track_id": track_id
            })

        return detections