import time
import cv2
from typing import Optional, Union


class VideoSource:
    """
    Generic video source for:
    - saved video files
    - webcam
    - RTSP CCTV stream
    """

    def __init__(
        self,
        source: Union[int, str],
        width: int = 960,
        height: int = 540,
        resize: bool = True,
        reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay_seconds: int = 2,
    ):
        self.source = self._normalize_source(source)
        self.width = width
        self.height = height
        self.resize = resize

        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.start_time = time.time()

    def _normalize_source(self, source: Union[int, str]) -> Union[int, str]:
        if isinstance(source, int):
            return source

        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())

        return source

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print(f"[INFO] Video source opened: {self.source}")
        print(f"[INFO] Source FPS: {self.get_source_fps()}")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Video source is not opened. Call open() first.")

        ret, frame = self.cap.read()

        if not ret:
            if self._is_rtsp_source() and self.reconnect:
                return self._attempt_reconnect()

            return False, None

        self.frame_count += 1

        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))

        return True, frame

    def _attempt_reconnect(self):
        print("[WARN] Frame read failed. Attempting reconnect...")

        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(f"[INFO] Reconnect attempt {attempt}/{self.max_reconnect_attempts}")

            self.release()
            time.sleep(self.reconnect_delay_seconds)

            self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret:
                    if self.resize:
                        frame = cv2.resize(frame, (self.width, self.height))

                    print("[INFO] Reconnected successfully.")
                    return True, frame

        print("[ERROR] Reconnect failed.")
        return False, None

    def _is_rtsp_source(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def get_source_fps(self) -> float:
        if self.cap is None:
            return 0.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            return 30.0

        return round(fps, 2)

    def get_runtime_fps(self) -> float:
        elapsed = time.time() - self.start_time

        if elapsed <= 0:
            return 0.0

        return round(self.frame_count / elapsed, 2)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None