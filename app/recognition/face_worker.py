import threading
import time
from typing import List, Dict, Any, Optional


class FaceRecognitionWorker:
    """
    Background worker for non-blocking face recognition.

    Main video loop remains smooth.
    Face recognition runs separately in another thread.
    """

    def __init__(
        self,
        face_recognizer,
        process_interval_seconds: float = 2.0,
    ):
        self.face_recognizer = face_recognizer
        self.process_interval_seconds = process_interval_seconds

        self.latest_frame = None
        self.latest_results: List[Dict[str, Any]] = []

        self.lock = threading.Lock()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.last_processed_time = 0.0
        self.is_processing = False

        self.latest_results_time = 0.0
        self.result_ttl_seconds = 3.0

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.worker_thread.start()

    def submit_frame(self, frame):
        """
        Called from main video loop.
        Stores only the latest frame.
        Older frames are automatically ignored.
        """
        with self.lock:
            self.latest_frame = frame.copy()

    def get_latest_results(self) -> List[Dict[str, Any]]:
        with self.lock:
            if time.time() - self.latest_results_time > self.result_ttl_seconds:
                return []

            return list(self.latest_results)

    def _run(self):
        while self.running:
            current_time = time.time()

            if current_time - self.last_processed_time < self.process_interval_seconds:
                time.sleep(0.01)
                continue

            with self.lock:
                if self.latest_frame is None:
                    frame_to_process = None
                else:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            self.is_processing = True

            try:
                results = self.face_recognizer.recognize_faces(frame_to_process)

                with self.lock:
                    self.latest_results = results
                    self.latest_results_time = time.time()

                self.last_processed_time = time.time()

            except Exception as error:
                print(f"[ERROR] Face recognition worker failed: {error}")

            finally:
                self.is_processing = False

    def stop(self):
        self.running = False

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2)