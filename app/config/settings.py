import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "540"))

    DISPLAY_WINDOW_NAME = os.getenv(
        "DISPLAY_WINDOW_NAME",
        "AI Office Monitoring System"
    )

    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt")
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.45"))
    YOLO_PERSON_CLASS_ID = int(os.getenv("YOLO_PERSON_CLASS_ID", "0"))


settings = Settings()