import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class EventWriter:
    """
    Writes batch video analysis events to CSV and JSON summary files.

    Designed for large CCTV files:
    - writes rows directly to disk
    - flushes after every row
    - does not keep all events in memory
    """

    def __init__(self, output_dir: str, source_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_source_name = Path(source_name).stem.replace(" ", "_")

        self.events_csv_path = (
            self.output_dir / f"events_{safe_source_name}_{timestamp}.csv"
        )

        self.summary_json_path = (
            self.output_dir / f"summary_{safe_source_name}_{timestamp}.json"
        )

        self.csv_file = open(
            self.events_csv_path,
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.fieldnames = [
            "system_time",
            "video_time_seconds",
            "video_time_hhmmss",
            "camera_id",
            "source_video",
            "frame_index",
            "processed_frame_index",
            "occupancy_count",
            "known_count",
            "unknown_count",
            "employee_ids",
            "employee_names",
            "event_type",
            "notes",
        ]

        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

        self.total_rows = 0

    def write_event(self, event: Dict[str, Any]) -> None:
        row = {field: event.get(field, "") for field in self.fieldnames}
        self.writer.writerow(row)
        self.csv_file.flush()
        self.total_rows += 1

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary["events_csv_path"] = str(self.events_csv_path)
        summary["total_event_rows"] = self.total_rows

        with open(self.summary_json_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

    def close(self) -> None:
        if not self.csv_file.closed:
            self.csv_file.close()

    def get_paths(self) -> Dict[str, str]:
        return {
            "events_csv": str(self.events_csv_path),
            "summary_json": str(self.summary_json_path),
        }