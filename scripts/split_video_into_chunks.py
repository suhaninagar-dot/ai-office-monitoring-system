import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import subprocess
from datetime import datetime


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(command)}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def get_video_duration_seconds(source_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source_path),
    ]

    output = run_command(command)
    metadata = json.loads(output)

    duration = float(metadata["format"]["duration"])
    return duration


def split_video_fast_copy(
    source_path: Path,
    output_dir: Path,
    chunk_minutes: int,
    output_prefix: str,
):
    """
    Splits video using stream copy.

    This is fast and does not re-encode video.
    It is best for large CCTV files.

    Note:
    Cuts may align near keyframes, so exact split time can vary slightly.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_seconds = chunk_minutes * 60
    output_pattern = str(output_dir / f"{output_prefix}_part_%03d.mp4")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        output_pattern,
    ]

    run_command(command)


def split_video_reencode(
    source_path: Path,
    output_dir: Path,
    chunk_minutes: int,
    output_prefix: str,
    width: int,
    height: int,
    fps: int,
):
    """
    Splits and compresses video.

    Use this when you want AI-friendly smaller files.
    This is slower than stream copy but reduces storage and processing load.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_seconds = chunk_minutes * 60
    output_pattern = str(output_dir / f"{output_prefix}_part_%03d.mp4")

    video_filter = f"fps={fps},scale={width}:{height}"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-an",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        output_pattern,
    ]

    run_command(command)


def list_output_chunks(output_dir: Path):
    chunks = sorted(output_dir.glob("*.mp4"))
    return chunks


def write_chunk_manifest(
    source_path: Path,
    output_dir: Path,
    chunk_minutes: int,
    mode: str,
    chunks,
):
    manifest_path = output_dir / "chunk_manifest.json"

    manifest = {
        "source_video": str(source_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_minutes": chunk_minutes,
        "mode": mode,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_index": index + 1,
                "chunk_path": str(chunk),
                "chunk_name": chunk.name,
            }
            for index, chunk in enumerate(chunks)
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    return manifest_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source video path",
    )

    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=30,
        help="Chunk duration in minutes",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "reencode"],
        default="copy",
        help="copy = fast split, reencode = compress and resize",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width for reencode mode",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Height for reencode mode",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="FPS for reencode mode",
    )

    args = parser.parse_args()

    source_path = Path(args.source)

    if not source_path.exists():
        raise RuntimeError(f"Source video not found: {source_path}")

    output_prefix = source_path.stem.replace(" ", "_")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT_DIR / "data" / "video_chunks" / output_prefix

    print("\n========== Video Chunking Started ==========")
    print(f"Source: {source_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Chunk Minutes: {args.chunk_minutes}")
    print(f"Mode: {args.mode}")

    duration_seconds = get_video_duration_seconds(source_path)
    print(f"Source Duration: {seconds_to_hhmmss(duration_seconds)}")

    if args.mode == "copy":
        print("[INFO] Using fast copy mode. No re-encoding.")
        split_video_fast_copy(
            source_path=source_path,
            output_dir=output_dir,
            chunk_minutes=args.chunk_minutes,
            output_prefix=output_prefix,
        )

    else:
        print("[INFO] Using reencode mode. This may take longer.")
        print(f"Output Size: {args.width}x{args.height}")
        print(f"Output FPS: {args.fps}")

        split_video_reencode(
            source_path=source_path,
            output_dir=output_dir,
            chunk_minutes=args.chunk_minutes,
            output_prefix=output_prefix,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    chunks = list_output_chunks(output_dir)

    manifest_path = write_chunk_manifest(
        source_path=source_path,
        output_dir=output_dir,
        chunk_minutes=args.chunk_minutes,
        mode=args.mode,
        chunks=chunks,
    )

    print("\n========== Video Chunking Completed ==========")
    print(f"Chunks Created: {len(chunks)}")
    print(f"Manifest: {manifest_path}")

    for chunk in chunks[:5]:
        print(f"- {chunk}")

    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks")

    print("==============================================")


if __name__ == "__main__":
    main()