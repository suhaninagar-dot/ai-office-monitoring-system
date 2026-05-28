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

    YOLO_FRAME_INTERVAL = int(os.getenv("YOLO_FRAME_INTERVAL", "3"))
   
    CAMERA_ID = os.getenv("CAMERA_ID", "CAM_001")
    OCCUPANCY_EMPTY_THRESHOLD = int(os.getenv("OCCUPANCY_EMPTY_THRESHOLD", "0"))
    OCCUPANCY_CROWD_THRESHOLD = int(os.getenv("OCCUPANCY_CROWD_THRESHOLD", "5"))

    FACE_IMAGE_SIZE = int(os.getenv("FACE_IMAGE_SIZE", "640"))
    FACE_CROP_PADDING = float(os.getenv("FACE_CROP_PADDING", "0.60"))

    EMPLOYEE_IMAGE_DIR = os.getenv("EMPLOYEE_IMAGE_DIR", "data/employees")
    PROCESSED_EMPLOYEE_IMAGE_DIR = os.getenv(
        "PROCESSED_EMPLOYEE_IMAGE_DIR",
        "data/employees_processed"
    )

    FACE_DB_DIR = os.getenv("FACE_DB_DIR", "data/face_db")
    FACE_EMBEDDINGS_PATH = os.getenv(
        "FACE_EMBEDDINGS_PATH",
        "data/face_db/face_embeddings.pkl"
    )
    FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45"))
    
    FACE_RECOGNITION_INTERVAL = int(os.getenv("FACE_RECOGNITION_INTERVAL", "20"))
    FACE_DETECTION_SIZE = int(os.getenv("FACE_DETECTION_SIZE", "320"))
    FACE_RECHECK_INTERVAL = int(os.getenv("FACE_RECHECK_INTERVAL", "150"))
    
settings = Settings()