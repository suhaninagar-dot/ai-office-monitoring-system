import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import time
from datetime import datetime

import cv2

from app.config.settings import settings
from app.detection.person_detector import PersonDetector
from app.recognition.face_recognizer import FaceRecognizer
from app.tracking.identity_cache import IdentityCache
from app.io.event_writer import EventWriter


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_video_metadata(cap: cv2.VideoCapture) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if fps is None or fps <= 0:
        fps = 30.0

    duration_seconds = frame_count / fps if frame_count and frame_count > 0 else 0

    return {
        "fps": round(float(fps), 2),
        "frame_count": int(frame_count) if frame_count else 0,
        "width": int(width) if width else 0,
        "height": int(height) if height else 0,
        "duration_seconds": round(duration_seconds, 2),
        "duration_hhmmss": seconds_to_hhmmss(duration_seconds),
    }


def extract_identity_summary(tracked_detections):
    employee_ids = []
    employee_names = []
    known_count = 0
    unknown_count = 0

    for detection in tracked_detections:
        identity = detection.get("identity")

        if not identity:
            continue

        if identity.get("is_known"):
            known_count += 1

            employee_id = identity.get("employee_id", "")
            name = identity.get("name", "")

            if employee_id and employee_id not in employee_ids:
                employee_ids.append(employee_id)

            if name and name not in employee_names:
                employee_names.append(name)
        else:
            unknown_count += 1

    return {
        "known_count": known_count,
        "unknown_count": unknown_count,
        "employee_ids": "|".join(employee_ids),
        "employee_names": "|".join(employee_names),
    }


def save_screenshot_if_needed(
    frame,
    screenshot_dir: str,
    source_name: str,
    video_time_seconds: float,
    event_type: str,
):
    screenshot_root = Path(screenshot_dir)
    screenshot_root.mkdir(parents=True, exist_ok=True)

    safe_source = Path(source_name).stem.replace(" ", "_")
    hhmmss = seconds_to_hhmmss(video_time_seconds).replace(":", "_")

    filename = f"{safe_source}_{event_type}_{hhmmss}.jpg"
    path = screenshot_root / filename

    cv2.imwrite(str(path), frame)

    return str(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Large CCTV video file path",
    )

    parser.add_argument(
        "--camera-id",
        type=str,
        default=settings.CAMERA_ID,
        help="Camera ID for reporting",
    )

    parser.add_argument(
        "--process-fps",
        type=float,
        default=settings.BATCH_PROCESS_FPS,
        help="How many frames per second to process from video",
    )

    parser.add_argument(
        "--face-interval-seconds",
        type=float,
        default=settings.BATCH_FACE_INTERVAL_SECONDS,
        help="Run face recognition every N video seconds",
    )

    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Optional safety limit for testing first N minutes",
    )

    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        help="Save screenshots for unknown person events",
    )

    args = parser.parse_args()

    source_path = Path(args.source)

    if not source_path.exists():
        raise RuntimeError(f"Source video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    metadata = get_video_metadata(cap)

    source_fps = metadata["fps"]
    sample_interval_frames = max(1, int(round(source_fps / args.process_fps)))

    print("\n========== Batch Video Analysis Started ==========")
    print(f"Source: {source_path}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Video FPS: {source_fps}")
    print(f"Video Duration: {metadata['duration_hhmmss']}")
    print(f"Video Resolution: {metadata['width']}x{metadata['height']}")
    print(f"Batch Process FPS: {args.process_fps}")
    print(f"Sample Every N Frames: {sample_interval_frames}")
    print(f"Face Interval Seconds: {args.face_interval_seconds}")
    print("Display Window: Disabled")
    print("==================================================\n")

    person_detector = PersonDetector(
        model_path=settings.YOLO_MODEL_PATH,
        confidence_threshold=settings.YOLO_CONFIDENCE,
        person_class_id=settings.YOLO_PERSON_CLASS_ID,
    )

    face_recognizer = FaceRecognizer(
        embeddings_path=settings.FACE_EMBEDDINGS_PATH,
        recognition_threshold=settings.FACE_RECOGNITION_THRESHOLD,
        use_gpu=False,
        detection_size=settings.FACE_DETECTION_SIZE,
    )

    identity_cache = IdentityCache(
        ttl_seconds=30.0,
        iou_threshold=0.25,
    )

    writer = EventWriter(
        output_dir=settings.BATCH_OUTPUT_DIR,
        source_name=source_path.name,
    )

    frame_index = 0
    processed_frame_index = 0

    last_face_recognition_video_time = -999999.0
    last_recognitions = []

    max_video_seconds = None
    if args.max_minutes is not None:
        max_video_seconds = args.max_minutes * 60

    total_person_detections = 0
    max_occupancy = 0
    unknown_event_count = 0

    previous_occupancy = None
    processing_start_time = time.time()
    last_progress_print_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[INFO] Video ended or frame unavailable.")
                break

            frame_index += 1

            video_time_seconds = frame_index / source_fps

            if max_video_seconds is not None and video_time_seconds > max_video_seconds:
                print(f"[INFO] Stopped after max-minutes limit: {args.max_minutes}")
                break

            if frame_index % sample_interval_frames != 0:
                continue

            processed_frame_index += 1

            frame = cv2.resize(
                frame,
                (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
            )

            detections = person_detector.detect(frame)

            total_person_detections += len(detections)
            max_occupancy = max(max_occupancy, len(detections))

            should_run_face = (
                video_time_seconds - last_face_recognition_video_time
                >= args.face_interval_seconds
            )

            if should_run_face:
                last_recognitions = face_recognizer.recognize_faces(frame)
                last_face_recognition_video_time = video_time_seconds

            tracked_detections = identity_cache.update(
                person_detections=detections,
                recognitions=last_recognitions,
            )

            identity_summary = extract_identity_summary(tracked_detections)

            occupancy_count = len(tracked_detections)

            event_type = "SAMPLED_STATE"
            screenshot_path = ""

            if previous_occupancy is None:
                event_type = "INITIAL_STATE"
            elif previous_occupancy != occupancy_count:
                event_type = "OCCUPANCY_CHANGED"

            if identity_summary["unknown_count"] > 0:
                event_type = "UNKNOWN_PERSON"
                unknown_event_count += 1

                if args.save_screenshots or settings.BATCH_SAVE_SCREENSHOTS:
                    annotated = person_detector.draw_detections_with_cached_names(
                        frame.copy(),
                        tracked_detections,
                    )

                    screenshot_path = save_screenshot_if_needed(
                        frame=annotated,
                        screenshot_dir=settings.BATCH_SCREENSHOT_DIR,
                        source_name=source_path.name,
                        video_time_seconds=video_time_seconds,
                        event_type="unknown",
                    )

            event = {
                "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_time_seconds": round(video_time_seconds, 2),
                "video_time_hhmmss": seconds_to_hhmmss(video_time_seconds),
                "camera_id": args.camera_id,
                "source_video": str(source_path),
                "frame_index": frame_index,
                "processed_frame_index": processed_frame_index,
                "occupancy_count": occupancy_count,
                "known_count": identity_summary["known_count"],
                "unknown_count": identity_summary["unknown_count"],
                "employee_ids": identity_summary["employee_ids"],
                "employee_names": identity_summary["employee_names"],
                "event_type": event_type,
                "notes": screenshot_path,
            }

            writer.write_event(event)

            previous_occupancy = occupancy_count

            current_time = time.time()

            if current_time - last_progress_print_time >= settings.BATCH_PROGRESS_INTERVAL_SECONDS:
                elapsed = current_time - processing_start_time
                processing_fps = processed_frame_index / max(elapsed, 0.001)

                print(
                    f"[PROGRESS] Video Time: {seconds_to_hhmmss(video_time_seconds)} | "
                    f"Processed Frames: {processed_frame_index} | "
                    f"Occupancy: {occupancy_count} | "
                    f"Known: {identity_summary['known_count']} | "
                    f"Unknown: {identity_summary['unknown_count']} | "
                    f"Processing FPS: {round(processing_fps, 2)}"
                )

                last_progress_print_time = current_time

    finally:
        cap.release()
        writer.close()

    total_runtime_seconds = time.time() - processing_start_time

    summary = {
        "source_video": str(source_path),
        "camera_id": args.camera_id,
        "video_metadata": metadata,
        "batch_settings": {
            "process_fps": args.process_fps,
            "sample_interval_frames": sample_interval_frames,
            "face_interval_seconds": args.face_interval_seconds,
            "frame_width": settings.FRAME_WIDTH,
            "frame_height": settings.FRAME_HEIGHT,
        },
        "processing_summary": {
            "total_frames_read": frame_index,
            "total_frames_processed": processed_frame_index,
            "total_person_detections": total_person_detections,
            "max_occupancy": max_occupancy,
            "unknown_event_count": unknown_event_count,
            "runtime_seconds": round(total_runtime_seconds, 2),
            "processed_frames_per_second": round(
                processed_frame_index / max(total_runtime_seconds, 0.001),
                2,
            ),
        },
    }

    writer.write_summary(summary)

    paths = writer.get_paths()

    print("\n========== Batch Video Analysis Completed ==========")
    print(f"Total Frames Read: {frame_index}")
    print(f"Total Frames Processed: {processed_frame_index}")
    print(f"Max Occupancy: {max_occupancy}")
    print(f"Unknown Events: {unknown_event_count}")
    print(f"Runtime Seconds: {round(total_runtime_seconds, 2)}")
    print(f"Events CSV: {paths['events_csv']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print("====================================================")


if __name__ == "__main__":
    main()