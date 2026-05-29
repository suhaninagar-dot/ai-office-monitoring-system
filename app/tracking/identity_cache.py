import time
from typing import List, Dict, Any, Optional


class IdentityCache:
    """
    Maintains temporary employee identity against person bounding boxes.

    This is a lightweight solution before ByteTrack.

    It helps keep employee names visible continuously even when face
    recognition is not running on every frame.
    """

    def __init__(
        self,
        ttl_seconds: float = 10.0,
        iou_threshold: float = 0.25,
    ):
        self.ttl_seconds = ttl_seconds
        self.iou_threshold = iou_threshold
        self.cached_identities: List[Dict[str, Any]] = []

    def update(
        self,
        person_detections: List[Dict[str, Any]],
        recognitions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Returns person detections enriched with identity.

        Output detection example:
        {
            "bbox": [...],
            "confidence": 0.88,
            "identity": {
                "name": "Deepak",
                "employee_id": "EMP002",
                "similarity": 0.63,
                "is_known": True
            }
        }
        """

        current_time = time.time()

        self._remove_expired(current_time)

        # First update cache using fresh recognitions
        for detection in person_detections:
            matched_recognition = self._match_recognition_to_person(
                detection,
                recognitions,
            )

            if matched_recognition is not None:
                self._save_identity(
                    bbox=detection["bbox"],
                    recognition=matched_recognition,
                    timestamp=current_time,
                )

        # Then assign cached identity to each current person detection
        enriched_detections = []

        for detection in person_detections:
            identity = self._find_cached_identity(detection["bbox"])

            enriched_detection = dict(detection)
            enriched_detection["identity"] = identity

            enriched_detections.append(enriched_detection)

        return enriched_detections

    def _save_identity(
        self,
        bbox: List[int],
        recognition: Dict[str, Any],
        timestamp: float,
    ) -> None:
        identity_record = {
            "bbox": bbox,
            "employee_id": recognition.get("employee_id"),
            "name": recognition.get("name", "Unknown"),
            "department": recognition.get("department", ""),
            "role": recognition.get("role", ""),
            "similarity": recognition.get("similarity", 0.0),
            "is_known": recognition.get("is_known", False),
            "timestamp": timestamp,
        }

        # Replace old matching cache if bbox overlaps
        replaced = False

        for index, existing in enumerate(self.cached_identities):
            if self._iou(existing["bbox"], bbox) >= self.iou_threshold:
                self.cached_identities[index] = identity_record
                replaced = True
                break

        if not replaced:
            self.cached_identities.append(identity_record)

    def _find_cached_identity(self, bbox: List[int]) -> Optional[Dict[str, Any]]:
        best_identity = None
        best_iou = 0.0

        for identity in self.cached_identities:
            iou_score = self._iou(identity["bbox"], bbox)

            if iou_score > best_iou:
                best_iou = iou_score
                best_identity = identity

        if best_identity is None:
            return None

        if best_iou < self.iou_threshold:
            return None

        return {
            "employee_id": best_identity["employee_id"],
            "name": best_identity["name"],
            "department": best_identity["department"],
            "role": best_identity["role"],
            "similarity": best_identity["similarity"],
            "is_known": best_identity["is_known"],
        }

    def _remove_expired(self, current_time: float) -> None:
        self.cached_identities = [
            item
            for item in self.cached_identities
            if current_time - item["timestamp"] <= self.ttl_seconds
        ]

    def _match_recognition_to_person(
        self,
        person_detection: Dict[str, Any],
        recognitions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        person_bbox = person_detection["bbox"]

        for recognition in recognitions:
            face_bbox = recognition.get("bbox")

            if not face_bbox:
                continue

            face_center = self._get_box_center(face_bbox)

            if self._is_point_inside_box(face_center, person_bbox):
                return recognition

        return None

    def _get_box_center(self, bbox: List[int]):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y

    def _is_point_inside_box(self, point, bbox: List[int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = bbox

        return x1 <= x <= x2 and y1 <= y <= y2

    def _iou(self, box_a: List[int], box_b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)

        intersection = inter_width * inter_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union