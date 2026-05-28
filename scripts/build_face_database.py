import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import os
import csv
import pickle
from typing import Dict, List, Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config.settings import settings


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_employee_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Reads data/employees/employees.csv.

    Expected columns:
    employee_id,name,department,role,shift_start,shift_end,status
    """

    employees = {}

    if not csv_path.exists():
        raise RuntimeError(f"Employee CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            employee_id = row.get("employee_id", "").strip()

            if not employee_id:
                continue

            employees[employee_id] = {
                "employee_id": employee_id,
                "name": row.get("name", "").strip(),
                "department": row.get("department", "").strip(),
                "role": row.get("role", "").strip(),
                "shift_start": row.get("shift_start", "").strip(),
                "shift_end": row.get("shift_end", "").strip(),
                "status": row.get("status", "").strip(),
            }

    return employees


def initialize_face_model() -> FaceAnalysis:
    """
    Initializes InsightFace model on CPU.

    ctx_id=-1 means CPU.
    Later, if GPU is available, ctx_id=0 can be used.
    """

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    return app


def get_image_paths(employee_dir: Path) -> List[Path]:
    image_paths = []

    for path in employee_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    return image_paths


def extract_face_embedding(face_model: FaceAnalysis, image_path: Path):
    """
    Detects face and returns normalized embedding.

    Returns:
        embedding: np.ndarray or None
        face_count: number of detected faces
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, 0

    faces = face_model.get(image)

    if len(faces) == 0:
        print(f"[WARN] No face detected: {image_path}")
        return None, 0

    if len(faces) > 1:
        print(f"[WARN] Multiple faces detected, using largest face: {image_path}")

    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    )

    embedding = largest_face.embedding

    if embedding is None:
        print(f"[WARN] No embedding generated: {image_path}")
        return None, len(faces)

    embedding = embedding.astype(np.float32)

    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding, len(faces)


def build_face_database() -> Dict[str, Any]:
    processed_root = ROOT_DIR / settings.PROCESSED_EMPLOYEE_IMAGE_DIR
    employee_csv_path = ROOT_DIR / settings.EMPLOYEE_IMAGE_DIR / "employees.csv"

    if not processed_root.exists():
        raise RuntimeError(f"Processed employee image directory not found: {processed_root}")

    employees_metadata = load_employee_metadata(employee_csv_path)
    face_model = initialize_face_model()

    face_database = {
        "model": "insightface_buffalo_l",
        "embedding_size": 512,
        "employees": {},
    }

    total_images = 0
    successful_embeddings = 0
    failed_images = 0

    for employee_dir in processed_root.iterdir():
        if not employee_dir.is_dir():
            continue

        employee_id = employee_dir.name

        if employee_id not in employees_metadata:
            print(f"[WARN] Employee folder exists but CSV entry missing: {employee_id}")
            continue

        image_paths = get_image_paths(employee_dir)

        if not image_paths:
            print(f"[WARN] No images found for employee: {employee_id}")
            continue

        employee_embeddings = []

        print(f"\n[INFO] Processing employee: {employee_id}")

        for image_path in image_paths:
            total_images += 1

            embedding, face_count = extract_face_embedding(face_model, image_path)

            if embedding is None:
                failed_images += 1
                continue

            employee_embeddings.append(
                {
                    "image_name": image_path.name,
                    "embedding": embedding,
                    "face_count": face_count,
                }
            )

            successful_embeddings += 1
            print(f"[OK] Embedding created: {employee_id}/{image_path.name}")

        if employee_embeddings:
            face_database["employees"][employee_id] = {
                "metadata": employees_metadata[employee_id],
                "embeddings": employee_embeddings,
            }

    summary = {
        "total_employees": len(face_database["employees"]),
        "total_images": total_images,
        "successful_embeddings": successful_embeddings,
        "failed_images": failed_images,
    }

    face_database["summary"] = summary

    return face_database


def save_face_database(face_database: Dict[str, Any]) -> str:
    output_path = ROOT_DIR / settings.FACE_EMBEDDINGS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(face_database, file)

    return str(output_path)


def print_summary(face_database: Dict[str, Any]) -> None:
    summary = face_database["summary"]

    print("\n========== Face Database Build Summary ==========")
    print(f"Total Employees Encoded: {summary['total_employees']}")
    print(f"Total Images Found: {summary['total_images']}")
    print(f"Successful Embeddings: {summary['successful_embeddings']}")
    print(f"Failed Images: {summary['failed_images']}")

    print("\nEmployee-wise Embeddings:")

    for employee_id, employee_data in face_database["employees"].items():
        metadata = employee_data["metadata"]
        embeddings = employee_data["embeddings"]

        print(
            f"- {employee_id} | {metadata['name']} | "
            f"Embeddings: {len(embeddings)}"
        )

    print("=================================================")


def main():
    print("[INFO] Building face database...")

    face_database = build_face_database()
    output_path = save_face_database(face_database)

    print_summary(face_database)

    print(f"\n[INFO] Face database saved at: {output_path}")


if __name__ == "__main__":
    main()