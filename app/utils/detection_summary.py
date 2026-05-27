import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class DetectionSummary:
    """
    Collects detection-level statistics for one video/camera run.

    Tracks:
    - total frames processed
    - runtime FPS
    - occupancy
    - confidence scores
    - frames with/without persons
    - pass/fail summary
    """

    def __init__(self, source: str):
        self.source = source

        self.total_frames = 0
        self.frames_with_person = 0
        self.frames_without_person = 0

        self.max_occupancy = 0
        self.total_person_detections = 0

        self.confidences: List[float] = []

        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """
        Update summary for each processed frame.

        Args:
            detections: list of person detections from PersonDetector
        """

        self.total_frames += 1

        occupancy_count = len(detections)

        self.max_occupancy = max(self.max_occupancy, occupancy_count)
        self.total_person_detections += occupancy_count

        if occupancy_count > 0:
            self.frames_with_person += 1
        else:
            self.frames_without_person += 1

        for detection in detections:
            confidence = detection.get("confidence")

            if confidence is not None:
                self.confidences.append(float(confidence))

    def finish(self) -> None:
        self.end_time = datetime.now()

    def get_duration_seconds(self) -> float:
        end_time = self.end_time or datetime.now()
        return max((end_time - self.start_time).total_seconds(), 0.001)

    def get_average_fps(self) -> float:
        return round(self.total_frames / self.get_duration_seconds(), 2)

    def get_min_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return round(min(self.confidences), 2)

    def get_max_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return round(max(self.confidences), 2)

    def get_average_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return round(sum(self.confidences) / len(self.confidences), 2)

    def get_status(self) -> str:
        """
        Simple MVP pass/fail logic.

        PASS if:
        - frames were processed
        - at least one person was detected
        - average FPS is usable for demo
        """

        if self.total_frames == 0:
            return "FAIL - No frames processed"

        if self.total_person_detections == 0:
            return "FAIL - No person detected"

        if self.get_average_fps() < 5:
            return "WARN - Low FPS"

        return "PASS"

    def build_report_text(self) -> str:
        self.finish()

        report = f"""
========== Detection Test Summary ==========
Source: {self.source}
Started At: {self.start_time}
Ended At: {self.end_time}
Duration Seconds: {round(self.get_duration_seconds(), 2)}

Total Frames Processed: {self.total_frames}
Average Runtime FPS: {self.get_average_fps()}

Max Occupancy: {self.max_occupancy}
Total Person Detections: {self.total_person_detections}

Frames With Person: {self.frames_with_person}
Frames Without Person: {self.frames_without_person}

Min Confidence: {self.get_min_confidence()}
Max Confidence: {self.get_max_confidence()}
Average Confidence: {self.get_average_confidence()}

Status: {self.get_status()}
============================================
"""
        return report.strip()

    def print_report(self) -> None:
        print("\n" + self.build_report_text())

    def save_report(self, output_dir: str = "data/reports") -> str:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_summary_{timestamp}.txt"
        file_path = os.path.join(output_dir, filename)

        report_text = self.build_report_text()

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(report_text)

        return file_path