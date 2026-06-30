from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class SAHIDetector:

    def __init__(self, model_path):

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.35,
            device="cpu"
        )

    def detect(self, image):

        result = get_sliced_prediction(
            image,
            self.model,

            slice_height=320,
            slice_width=320,

            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        detections = []

        for pred in result.object_prediction_list:

            bbox = pred.bbox

            detections.append({

                "bbox": [
                    int(bbox.minx),
                    int(bbox.miny),
                    int(bbox.maxx),
                    int(bbox.maxy)
                ],

                "confidence": float(pred.score.value),

                "class_id": int(pred.category.id),

                "track_id": None

            })

        return detections