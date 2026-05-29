import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import cv2
import time
from app.tracking.identity_cache import IdentityCache
from app.camera.video_source import VideoSource
from app.config.settings import settings
from app.detection.person_detector import PersonDetector
from app.recognition.face_recognizer import FaceRecognizer
from app.recognition.face_worker import FaceRecognitionWorker


def draw_dashboard_info(
    frame,
    occupancy_count,
    display_fps,
    frame_index,
    face_worker_status,
):
    panel_x = 20
    panel_y = 30
    line_gap = 26

    lines = [
        f"Occupancy: {occupancy_count}",
        f"Display FPS: {display_fps}",
        f"Frame: {frame_index}",
        f"Face Worker: {face_worker_status}",
    ]

    y = panel_y

    for line in lines:
        cv2.putText(
            frame,
            line,
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += line_gap

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
        default=settings.YOLO_FRAME_INTERVAL,
        help="Run YOLO every N frames",
    )

    parser.add_argument(
        "--face-det-size",
        type=int,
        default=settings.FACE_DETECTION_SIZE,
        help="InsightFace detection size",
    )

    parser.add_argument(
        "--face-worker-interval",
        type=float,
        default=2.0,
        help="Run background face recognition every N seconds",
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

    face_worker = FaceRecognitionWorker(
        face_recognizer=face_recognizer,
        process_interval_seconds=args.face_worker_interval,
    )

    identity_cache = IdentityCache(
        ttl_seconds=10.0,
        iou_threshold=0.25,
    )

    video.open()
    face_worker.start()

    frame_index = 0
    last_person_detections = []

    start_time = time.time()

    try:
        while True:
            ret, frame = video.read()

            if not ret:
                print("[INFO] Video ended or frame not available.")
                break

            frame_index += 1

            # YOLO runs occasionally, not every frame
            if frame_index % args.yolo_interval == 0:
                last_person_detections = person_detector.detect(frame)

            # Send current frame to face worker.
            # This does not block the display loop.
            face_worker.submit_frame(frame)

            # Get latest available face results.
            # These may be from a previous frame, which is okay for demo.
            recognitions = face_worker.get_latest_results()

            tracked_detections = identity_cache.update(
                person_detections=last_person_detections,
                recognitions=recognitions,
            )

            occupancy_count = len(tracked_detections)

            frame = person_detector.draw_detections_with_cached_names(
                frame=frame,
                detections=tracked_detections,
       )
            #frame = face_recognizer.draw_recognitions(frame, recognitions)

            elapsed = time.time() - start_time
            display_fps = round(frame_index / max(elapsed, 0.001), 2)

            face_worker_status = "PROCESSING" if face_worker.is_processing else "IDLE"

            frame = draw_dashboard_info(
                frame=frame,
                occupancy_count=occupancy_count,
                display_fps=display_fps,
                frame_index=frame_index,
                face_worker_status=face_worker_status,
            )

            cv2.imshow("Realtime Face Recognition - Non Blocking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped manually by user.")
                break

    finally:
        face_worker.stop()
        video.release()
        cv2.destroyAllWindows()

    total_time = round(time.time() - start_time, 2)

    print("\n========== Realtime Face Demo Summary ==========")
    print(f"Frames Displayed: {frame_index}")
    print(f"Runtime Seconds: {total_time}")
    print(f"Average Display FPS: {round(frame_index / max(total_time, 0.001), 2)}")
    print(f"YOLO Interval: {args.yolo_interval}")
    print(f"Face Worker Interval Seconds: {args.face_worker_interval}")
    print(f"Face Detection Size: {args.face_det_size}")
    print("================================================")


if __name__ == "__main__":
    main()