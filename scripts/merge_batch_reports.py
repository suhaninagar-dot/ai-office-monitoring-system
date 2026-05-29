import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
import json
from datetime import datetime
from typing import List, Dict, Any


EVENTS_PREFIX = "events_"
SUMMARY_PREFIX = "summary_"


def find_event_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/events_*.csv" if recursive else "events_*.csv"
    return sorted(input_dir.glob(pattern))


def find_summary_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/summary_*.json" if recursive else "summary_*.json"
    return sorted(input_dir.glob(pattern))


def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def read_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_int(value, default: int = 0) -> int:
    try:
        if value in [None, ""]:
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except Exception:
        return default


def get_all_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def write_merged_events(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            file.write("")
        return

    fieldnames = get_all_fieldnames(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def build_merged_summary(
    event_rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    event_files: List[Path],
    summary_files: List[Path],
) -> Dict[str, Any]:
    employee_ids = set()
    employee_names = set()

    max_occupancy = 0
    total_unknown_events = 0
    total_event_rows = len(event_rows)

    event_type_counts = {}

    cameras = set()
    source_videos = set()

    earliest_video_time = None
    latest_video_time = None

    for row in event_rows:
        camera_id = row.get("camera_id", "")
        source_video = row.get("source_video", "")
        event_type = row.get("event_type", "")

        if camera_id:
            cameras.add(camera_id)

        if source_video:
            source_videos.add(source_video)

        occupancy = safe_int(row.get("occupancy_count"))
        max_occupancy = max(max_occupancy, occupancy)

        unknown_count = safe_int(row.get("unknown_count"))
        if unknown_count > 0 or event_type == "UNKNOWN_PERSON":
            total_unknown_events += 1

        if event_type:
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        emp_ids = row.get("employee_ids", "")
        emp_names = row.get("employee_names", "")

        for emp_id in emp_ids.split("|"):
            emp_id = emp_id.strip()
            if emp_id:
                employee_ids.add(emp_id)

        for name in emp_names.split("|"):
            name = name.strip()
            if name:
                employee_names.add(name)

        video_time_seconds = safe_float(row.get("video_time_seconds"), None)

        if video_time_seconds is not None:
            if earliest_video_time is None or video_time_seconds < earliest_video_time:
                earliest_video_time = video_time_seconds

            if latest_video_time is None or video_time_seconds > latest_video_time:
                latest_video_time = video_time_seconds

    total_frames_read = 0
    total_frames_processed = 0
    total_person_detections = 0
    combined_runtime_seconds = 0.0

    video_metadata_list = []
    batch_settings_list = []

    for summary in summaries:
        processing_summary = summary.get("processing_summary", {})
        video_metadata = summary.get("video_metadata", {})
        batch_settings = summary.get("batch_settings", {})

        total_frames_read += safe_int(processing_summary.get("total_frames_read"))
        total_frames_processed += safe_int(processing_summary.get("total_frames_processed"))
        total_person_detections += safe_int(processing_summary.get("total_person_detections"))
        combined_runtime_seconds += safe_float(processing_summary.get("runtime_seconds"))

        if video_metadata:
            video_metadata_list.append(video_metadata)

        if batch_settings:
            batch_settings_list.append(batch_settings)

    avg_processed_fps = 0.0

    if combined_runtime_seconds > 0:
        avg_processed_fps = round(total_frames_processed / combined_runtime_seconds, 2)

    merged_summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_event_files_count": len(event_files),
        "input_summary_files_count": len(summary_files),
        "total_event_rows": total_event_rows,
        "cameras": sorted(list(cameras)),
        "source_videos": sorted(list(source_videos)),
        "known_employee_ids": sorted(list(employee_ids)),
        "known_employee_names": sorted(list(employee_names)),
        "max_occupancy": max_occupancy,
        "total_unknown_event_rows": total_unknown_events,
        "event_type_counts": event_type_counts,
        "video_time_range": {
            "earliest_video_time_seconds": earliest_video_time,
            "latest_video_time_seconds": latest_video_time,
        },
        "combined_processing_summary": {
            "total_frames_read": total_frames_read,
            "total_frames_processed": total_frames_processed,
            "total_person_detections": total_person_detections,
            "combined_runtime_seconds": round(combined_runtime_seconds, 2),
            "average_processed_frames_per_second": avg_processed_fps,
        },
        "input_files": {
            "event_files": [str(path) for path in event_files],
            "summary_files": [str(path) for path in summary_files],
        },
        "video_metadata_list": video_metadata_list,
        "batch_settings_list": batch_settings_list,
    }

    return merged_summary


def write_summary(summary: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing batch event CSV and summary JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged reports",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search reports recursively",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="batch",
        help="Name prefix for merged output files",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "merged"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    merged_events_path = output_dir / f"merged_events_{args.name}_{timestamp}.csv"
    merged_summary_path = output_dir / f"merged_summary_{args.name}_{timestamp}.json"

    event_files = find_event_files(input_dir, recursive=args.recursive)
    summary_files = find_summary_files(input_dir, recursive=args.recursive)

    print("\n========== Batch Report Merge Started ==========")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Recursive: {args.recursive}")
    print(f"Event CSV Files Found: {len(event_files)}")
    print(f"Summary JSON Files Found: {len(summary_files)}")
    print("===============================================\n")

    if not event_files:
        print("[WARN] No event CSV files found. Nothing to merge.")
        return

    all_event_rows = []

    for event_file in event_files:
        rows = read_csv_rows(event_file)

        for row in rows:
            row["source_event_file"] = str(event_file)

        all_event_rows.extend(rows)

    summaries = []

    for summary_file in summary_files:
        try:
            summary = read_json(summary_file)
            summary["source_summary_file"] = str(summary_file)
            summaries.append(summary)
        except Exception as error:
            print(f"[WARN] Could not read summary file: {summary_file} | {error}")

    write_merged_events(
        rows=all_event_rows,
        output_path=merged_events_path,
    )

    merged_summary = build_merged_summary(
        event_rows=all_event_rows,
        summaries=summaries,
        event_files=event_files,
        summary_files=summary_files,
    )

    merged_summary["merged_events_path"] = str(merged_events_path)
    merged_summary["merged_summary_path"] = str(merged_summary_path)

    write_summary(
        summary=merged_summary,
        output_path=merged_summary_path,
    )

    print("\n========== Batch Report Merge Completed ==========")
    print(f"Total Event Rows: {len(all_event_rows)}")
    print(f"Max Occupancy: {merged_summary['max_occupancy']}")
    print(f"Known Employees: {len(merged_summary['known_employee_ids'])}")
    print(f"Unknown Event Rows: {merged_summary['total_unknown_event_rows']}")
    print(f"Merged Events CSV: {merged_events_path}")
    print(f"Merged Summary JSON: {merged_summary_path}")
    print("=================================================")


if __name__ == "__main__":
    main()