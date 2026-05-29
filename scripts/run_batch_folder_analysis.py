import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import subprocess
from datetime import datetime

from app.config.settings import settings
from app.io.processing_manifest import ProcessingManifest


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}


def find_video_files(input_dir: Path, recursive: bool = False):
    if recursive:
        files = [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
    else:
        files = [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

    return sorted(files)


def build_command(
    video_path: str,
    camera_id: str,
    process_fps: float,
    face_interval_seconds: float,
    max_minutes,
    save_screenshots: bool,
):
    command = [
        sys.executable,
        "scripts/run_batch_video_analysis.py",
        "--source",
        video_path,
        "--camera-id",
        camera_id,
        "--process-fps",
        str(process_fps),
        "--face-interval-seconds",
        str(face_interval_seconds),
    ]

    if max_minutes is not None:
        command.extend(["--max-minutes", str(max_minutes)])

    if save_screenshots:
        command.append("--save-screenshots")

    return command


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing CCTV video files",
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
        help="How many frames per second to process from each video",
    )

    parser.add_argument(
        "--face-interval-seconds",
        type=float,
        default=settings.BATCH_FACE_INTERVAL_SECONDS,
        help="Run face recognition every N video seconds",
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest CSV path",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search videos recursively in subfolders",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only first N pending files",
    )

    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Testing mode: process only first N minutes of each video",
    )

    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        help="Save screenshots for unknown person events",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        manifest_path = (
            Path(settings.BATCH_OUTPUT_DIR)
            / f"processing_manifest_{timestamp}.csv"
        )

    video_files = find_video_files(
        input_dir=input_dir,
        recursive=args.recursive,
    )

    if not video_files:
        print(f"[WARN] No video files found in: {input_dir}")
        return

    manifest = ProcessingManifest(str(manifest_path))
    manifest.initialize_from_videos(video_files)

    print("\n========== Batch Folder Analysis Started ==========")
    print(f"Input Directory: {input_dir}")
    print(f"Recursive: {args.recursive}")
    print(f"Videos Found: {len(video_files)}")
    print(f"Manifest: {manifest_path}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Process FPS: {args.process_fps}")
    print(f"Face Interval Seconds: {args.face_interval_seconds}")
    print("=================================================\n")

    processed_count = 0

    while True:
        if args.max_files is not None and processed_count >= args.max_files:
            print(f"[INFO] Reached max-files limit: {args.max_files}")
            break

        next_item = manifest.get_next_pending()

        if next_item is None:
            print("[INFO] No pending or failed videos left to process.")
            break

        video_path = next_item["video_path"]

        print("\n--------------------------------------------------")
        print(f"[INFO] Processing video: {video_path}")
        print("--------------------------------------------------")

        manifest.mark_processing(video_path)

        command = build_command(
            video_path=video_path,
            camera_id=args.camera_id,
            process_fps=args.process_fps,
            face_interval_seconds=args.face_interval_seconds,
            max_minutes=args.max_minutes,
            save_screenshots=args.save_screenshots,
        )

        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                error_message = result.stderr.strip() or result.stdout.strip()
                print(f"[ERROR] Failed video: {video_path}")
                print(error_message)

                manifest.mark_failed(
                    video_path=video_path,
                    error_message=error_message[:1000],
                )
            else:
                print(result.stdout)

                events_csv = ""
                summary_json = ""

                for line in result.stdout.splitlines():
                    if line.startswith("Events CSV:"):
                        events_csv = line.replace("Events CSV:", "").strip()

                    if line.startswith("Summary JSON:"):
                        summary_json = line.replace("Summary JSON:", "").strip()

                manifest.mark_completed(
                    video_path=video_path,
                    events_csv=events_csv,
                    summary_json=summary_json,
                )

                processed_count += 1

        except Exception as error:
            print(f"[ERROR] Exception while processing {video_path}: {error}")

            manifest.mark_failed(
                video_path=video_path,
                error_message=str(error),
            )

    counts = manifest.get_summary_counts()

    print("\n========== Batch Folder Analysis Completed ==========")
    print(f"Manifest: {manifest_path}")
    print(f"Completed: {counts.get('COMPLETED', 0)}")
    print(f"Failed: {counts.get('FAILED', 0)}")
    print(f"Pending: {counts.get('PENDING', 0)}")
    print(f"Processing: {counts.get('PROCESSING', 0)}")
    print("====================================================")


if __name__ == "__main__":
    main()