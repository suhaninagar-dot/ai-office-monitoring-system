from typing import List, Dict, Any, Optional


def get_box_center(bbox: List[int]):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def is_point_inside_box(point, bbox: List[int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox

    return x1 <= x <= x2 and y1 <= y <= y2


def match_recognition_to_person(
    person_detection: Dict[str, Any],
    recognitions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Match a recognized face to a YOLO person box.

    Logic:
    - Take face bbox center point.
    - Check which person bbox contains that center.
    - Return the matching recognition.
    """

    person_bbox = person_detection["bbox"]

    for recognition in recognitions:
        face_bbox = recognition.get("bbox")

        if not face_bbox:
            continue

        face_center = get_box_center(face_bbox)

        if is_point_inside_box(face_center, person_bbox):
            return recognition

    return None