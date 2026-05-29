import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ProcessingManifest:
    """
    Tracks batch processing status for multiple video files.

    Status values:
    - PENDING
    - PROCESSING
    - COMPLETED
    - FAILED

    This allows large CCTV processing to resume safely.
    """

    FIELDNAMES = [
        "video_path",
        "status",
        "started_at",
        "completed_at",
        "events_csv",
        "summary_json",
        "error_message",
    ]

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            self._create_empty_manifest()

    def _create_empty_manifest(self) -> None:
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def load(self) -> List[Dict[str, str]]:
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    def save(self, rows: List[Dict[str, str]]) -> None:
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for row in rows:
                writer.writerow({
                    field: row.get(field, "")
                    for field in self.FIELDNAMES
                })

    def initialize_from_videos(self, video_paths: List[Path]) -> None:
        """
        Adds new videos to manifest as PENDING.
        Keeps existing statuses unchanged.
        """

        rows = self.load()
        existing_paths = {row["video_path"] for row in rows}

        for video_path in video_paths:
            video_path_str = str(video_path)

            if video_path_str in existing_paths:
                continue

            rows.append({
                "video_path": video_path_str,
                "status": "PENDING",
                "started_at": "",
                "completed_at": "",
                "events_csv": "",
                "summary_json": "",
                "error_message": "",
            })

        self.save(rows)

    def get_next_pending(self) -> Optional[Dict[str, str]]:
        rows = self.load()

        for row in rows:
            if row["status"] in ["PENDING", "FAILED"]:
                return row

        return None

    def mark_processing(self, video_path: str) -> None:
        rows = self.load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for row in rows:
            if row["video_path"] == video_path:
                row["status"] = "PROCESSING"
                row["started_at"] = now
                row["completed_at"] = ""
                row["error_message"] = ""
                break

        self.save(rows)

    def mark_completed(
        self,
        video_path: str,
        events_csv: str,
        summary_json: str,
    ) -> None:
        rows = self.load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for row in rows:
            if row["video_path"] == video_path:
                row["status"] = "COMPLETED"
                row["completed_at"] = now
                row["events_csv"] = events_csv
                row["summary_json"] = summary_json
                row["error_message"] = ""
                break

        self.save(rows)

    def mark_failed(self, video_path: str, error_message: str) -> None:
        rows = self.load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for row in rows:
            if row["video_path"] == video_path:
                row["status"] = "FAILED"
                row["completed_at"] = now
                row["error_message"] = error_message
                break

        self.save(rows)

    def get_summary_counts(self) -> Dict[str, int]:
        rows = self.load()

        counts = {
            "PENDING": 0,
            "PROCESSING": 0,
            "COMPLETED": 0,
            "FAILED": 0,
        }

        for row in rows:
            status = row.get("status", "PENDING")
            counts[status] = counts.get(status, 0) + 1

        return counts