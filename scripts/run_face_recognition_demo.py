import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import cv2
import time

from app.camera.video_source import VideoSource
from app.config.settings import settings
from app.detection.person_detector import PersonDetector
from app.recognition.face_recognizer import FaceRecognizer


def draw_dashboard_info(frame, recognitions, occupancy_count, runtime_fps, frame_index):
    known_count = sum(1 for item in recognitions if item["is_known"])
    unknown_count = sum(1 for item in recognitions if not item["is_known"])

    cv2.putText(frame, f"Occupancy: {occupancy_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Known Faces: {known_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Unknown Faces: {unknown_count}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {runtime_fps}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Frame: {frame_index}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default=settings.VIDEO_SOURCE,
        help="Video source: webcam index, video path, or RTSP URL",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_RECOGNITION_THRESHOLD,
        help="Face recognition similarity threshold",
    )

    parser.add_argument(
        "--yolo-interval",
        type=int,
        default=getattr(settings, "YOLO_FRAME_INTERVAL", 3),
        help="Run YOLO every N frames",
    )

    parser.add_argument(
        "--face-interval",
        type=int,
        default=settings.FACE_RECOGNITION_INTERVAL,
        help="Run face recognition every N frames",
    )

    parser.add_argument(
        "--face-det-size",
        type=int,
        default=settings.FACE_DETECTION_SIZE,
        help="InsightFace detection size",
    )

    parser.add_argument(
        "--print-logs",
        action="store_true",
        help="Print recognition logs",
    )

    args = parser.parse_args()

    video = VideoSource(
        source=args.source,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
        resize=True,
    )

    person_detector = PersonDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence_threshold=settings.YOLO_CONFIDENCE,
        person_class_id=settings.YOLO_PERSON_CLASS_ID,
    )

    face_recognizer = FaceRecognizer(
        embeddings_path=settings.FACE_EMBEDDINGS_PATH,
        recognition_threshold=args.threshold,
        use_gpu=False,
        detection_size=args.face_det_size,
    )

    video.open()

    frame_index = 0
    last_person_detections = []
    last_recognitions = []

    start_time = time.time()

    while True:
        ret, frame = video.read()

        if not ret:
            print("[INFO] Video ended or frame not available.")
            break

        frame_index += 1

        # Run YOLO only every N frames
        if frame_index % args.yolo_interval == 0:
            last_person_detections = person_detector.detect(frame)

        # Run face recognition only every N frames
        if frame_index % args.face_interval == 0:
            last_recognitions = face_recognizer.recognize_faces(frame)

            if args.print_logs:
                for recognition in last_recognitions:
                    if recognition["is_known"]:
                        print(
                            f"Recognized: {recognition['name']} | "
                            f"Similarity: {recognition['similarity']}"
                        )
                    else:
                        print(
                            f"Unknown detected | "
                            f"Best similarity: {recognition['similarity']}"
                        )

        occupancy_count = len(last_person_detections)

        frame = person_detector.draw_detections(frame, last_person_detections)
        frame = face_recognizer.draw_recognitions(frame, last_recognitions)

        elapsed = time.time() - start_time
        runtime_fps = round(frame_index / max(elapsed, 0.001), 2)

        frame = draw_dashboard_info(
            frame=frame,
            recognitions=last_recognitions,
            occupancy_count=occupancy_count,
            runtime_fps=runtime_fps,
            frame_index=frame_index,
        )

        cv2.imshow("Phase 3 - Optimized Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped manually by user.")
            break

    video.release()
    cv2.destroyAllWindows()

    total_time = round(time.time() - start_time, 2)

    print("\n========== Face Recognition Runtime Summary ==========")
    print(f"Frames Processed: {frame_index}")
    print(f"Total Runtime Seconds: {total_time}")
    print(f"Average FPS: {round(frame_index / max(total_time, 0.001), 2)}")
    print(f"YOLO Interval: {args.yolo_interval}")
    print(f"Face Recognition Interval: {args.face_interval}")
    print(f"Face Detection Size: {args.face_det_size}")
    print("======================================================")


if __name__ == "__main__":
    main()