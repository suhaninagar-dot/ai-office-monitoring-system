import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import cv2

from app.camera.video_source import VideoSource
from app.config.settings import settings
from app.detection.person_detector import PersonDetector
from app.utils.detection_summary import DetectionSummary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=settings.VIDEO_SOURCE,
        help="Video source: webcam index, video path, or RTSP URL",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detection summary report to data/reports",
    )
    args = parser.parse_args()

    video = VideoSource(
        source=args.source,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
        resize=True,
    )

    detector = PersonDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence_threshold=settings.YOLO_CONFIDENCE,
        person_class_id=settings.YOLO_PERSON_CLASS_ID,
    )

    summary = DetectionSummary(source=args.source)

    video.open()

    while True:
        ret, frame = video.read()

        if not ret:
            print("[INFO] Video ended or frame not available.")
            break

        detections = detector.detect(frame)

        summary.update(detections)

        frame = detector.draw_detections(frame, detections)

        occupancy_count = len(detections)
        runtime_fps = video.get_runtime_fps()

        cv2.putText(
            frame,
            f"Occupancy: {occupancy_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {runtime_fps}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Phase 2 - Person Detection", frame)

        for detection in detections:
            print(
                f"Person detected | Confidence: {detection['confidence']} | "
                f"BBox: {detection['bbox']}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped manually by user.")
            break

    video.release()
    cv2.destroyAllWindows()

    if args.save_report:
        report_path = summary.save_report()
        print(f"\n[INFO] Detection summary saved: {report_path}")
    else:
        summary.print_report()


if __name__ == "__main__":
    main()