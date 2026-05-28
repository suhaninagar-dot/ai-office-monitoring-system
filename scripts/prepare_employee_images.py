import sys
from pathlib import Path
import numpy as np
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import shutil

from app.config.settings import settings


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def create_square_face_crop(image, face_box, padding_ratio=0.60):
    """
    Creates a square crop around the detected face with padding.

    face_box format:
    x, y, w, h
    """

    image_height, image_width = image.shape[:2]
    x, y, w, h = face_box

    face_center_x = x + w // 2
    face_center_y = y + h // 2

    crop_size = int(max(w, h) * (1 + padding_ratio))

    x1 = max(face_center_x - crop_size // 2, 0)
    y1 = max(face_center_y - crop_size // 2, 0)
    x2 = min(face_center_x + crop_size // 2, image_width)
    y2 = min(face_center_y + crop_size // 2, image_height)

    crop = image[y1:y2, x1:x2]

    return crop


def resize_to_square(image, size):
    """
    Resize image into a square canvas without distortion.
    Keeps aspect ratio and pads remaining area.
    """

    image_height, image_width = image.shape[:2]

    scale = size / max(image_width, image_height)

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    square = 255 * np.ones((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    square[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return square


def detect_largest_face(image):
    """
    Uses OpenCV Haar Cascade to detect face.
    Works best for front and slight-angle images.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    # Pick largest detected face
    largest_face = max(faces, key=lambda box: box[2] * box[3])

    return largest_face


def process_image(input_path, output_path):
    image = cv2.imread(str(input_path))

    if image is None:
        print(f"[WARN] Could not read image: {input_path}")
        return False

    filename = input_path.name.lower()

    # Side-angle images are unreliable with Haar Cascade.
    # For these, do not auto-crop. Resize the full image safely.
    side_angle_keywords = [
        "left",
        "right",
        "side",
        "angle",
        "left_15",
        "right_15",
        "left_30",
        "right_30",
    ]

    is_side_angle = any(keyword in filename for keyword in side_angle_keywords)

    if is_side_angle:
        processed = resize_to_square(image, settings.FACE_IMAGE_SIZE)
        print(f"[INFO] Side-angle image resized without crop: {input_path}")

    else:
        face_box = detect_largest_face(image)

        if face_box is not None:
            crop = create_square_face_crop(
                image=image,
                face_box=face_box,
                padding_ratio=settings.FACE_CROP_PADDING,
            )

            processed = resize_to_square(crop, settings.FACE_IMAGE_SIZE)

            print(f"[OK] Face detected and processed: {input_path}")
        else:
            processed = resize_to_square(image, settings.FACE_IMAGE_SIZE)

            print(f"[WARN] Face not detected, resized full image: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), processed)

    return True


def main():
    input_root = ROOT_DIR / settings.EMPLOYEE_IMAGE_DIR
    output_root = ROOT_DIR / settings.PROCESSED_EMPLOYEE_IMAGE_DIR

    if not input_root.exists():
        raise RuntimeError(f"Employee image directory not found: {input_root}")

    print(f"[INFO] Input directory: {input_root}")
    print(f"[INFO] Output directory: {output_root}")

    if output_root.exists():
        print("[INFO] Clearing old processed images...")
        shutil.rmtree(output_root)

    total_images = 0
    processed_images = 0

    for employee_dir in input_root.iterdir():
        if not employee_dir.is_dir():
            continue

        employee_id = employee_dir.name

        for image_path in employee_dir.iterdir():
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            total_images += 1

            output_path = output_root / employee_id / image_path.name

            success = process_image(image_path, output_path)

            if success:
                processed_images += 1

    print("\n========== Employee Image Processing Summary ==========")
    print(f"Total Images Found: {total_images}")
    print(f"Processed Images: {processed_images}")
    print(f"Output Directory: {output_root}")
    print("=======================================================")


if __name__ == "__main__":
    main()