from datetime import datetime
from typing import List, Dict, Any


class OccupancyCounter:
    """
    Converts person detections into structured occupancy events.

    Current MVP logic:
    occupancy_count = number of detected persons in current frame

    Later this can be upgraded with:
    - zone occupancy
    - camera-wise occupancy
    - tracking-based unique counts
    - duplicate filtering
    """

    def __init__(
        self,
        camera_id: str = "CAM_001",
        empty_threshold: int = 0,
        crowd_threshold: int = 5,
    ):
        self.camera_id = camera_id
        self.empty_threshold = empty_threshold
        self.crowd_threshold = crowd_threshold

    def calculate(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        occupancy_count = len(detections)
        status = self._get_status(occupancy_count)

        event = {
            "camera_id": self.camera_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "occupancy_count": occupancy_count,
            "status": status,
            "detections": self._format_detections(detections),
        }

        return event

    def _get_status(self, occupancy_count: int) -> str:
        if occupancy_count <= self.empty_threshold:
            return "EMPTY"

        if occupancy_count >= self.crowd_threshold:
            return "CROWDED"

        return "OCCUPIED"

    def _format_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        formatted = []

        for detection in detections:
            formatted.append(
                {
                    "label": detection.get("label", "person"),
                    "confidence": detection.get("confidence", 0.0),
                    "bbox": detection.get("bbox", []),
                }
            )

        return formatted