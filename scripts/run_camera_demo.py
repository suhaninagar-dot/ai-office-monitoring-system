import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2

from app.camera.video_source import VideoSource
from app.config.settings import settings


def main():
    video = VideoSource(
        source=settings.VIDEO_SOURCE,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
        resize=True,
    )

    video.open()

    while True:
        ret, frame = video.read()

        if not ret:
            print("[INFO] Video ended or frame not available.")
            break

        runtime_fps = video.get_runtime_fps()

        cv2.putText(
            frame,
            f"FPS: {runtime_fps}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(settings.DISPLAY_WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()