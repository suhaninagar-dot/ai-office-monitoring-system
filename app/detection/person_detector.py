from typing import List, Dict, Any

import cv2
from ultralytics import YOLO


class PersonDetector:
    """
    YOLOv8 person detector.

    Detects only class_id 0, which is 'person' in the COCO dataset.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.45,
        person_class_id: int = 0,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id

        self.model = YOLO(self.model_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Returns list of detected persons.

        Each detection:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.92,
            "class_id": 0,
            "label": "person"
        }
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id != self.person_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 2),
                    "class_id": class_id,
                    "label": "person",
                }
            )

        return detections

    def draw_detections(self, frame, detections: List[Dict[str, Any]]):
        """
        Draws bounding boxes and confidence scores.
        """

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            label = f"Person {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame