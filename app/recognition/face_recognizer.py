import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """
    Face recognition engine using InsightFace embeddings.

    Flow:
    1. Load saved employee embeddings
    2. Detect faces in video frame
    3. Generate embedding for each detected face
    4. Compare with employee database
    5. Return employee name or Unknown
    """

    def __init__( 
        self,
        embeddings_path: str,
        recognition_threshold: float = 0.45,
        use_gpu: bool = False,
        detection_size: int = 320,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.recognition_threshold = recognition_threshold
        self.use_gpu = use_gpu
        self.detection_size = detection_size

        self.face_database = self._load_face_database()
        self.face_model = self._initialize_face_model()

    def _load_face_database(self) -> Dict[str, Any]:
        if not self.embeddings_path.exists():
            raise RuntimeError(f"Face embeddings file not found: {self.embeddings_path}")

        with open(self.embeddings_path, "rb") as file:
            face_database = pickle.load(file)

        print(f"[INFO] Face database loaded: {self.embeddings_path}")
        print(f"[INFO] Employees loaded: {len(face_database.get('employees', {}))}")

        return face_database

    def _initialize_face_model(self) -> FaceAnalysis:
        providers = ["CPUExecutionProvider"]

        ctx_id = -1

        if self.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )

        app.prepare(
            ctx_id=ctx_id,
            det_size=(self.detection_size, self.detection_size),
        )

        return app

    def recognize_faces(self, frame) -> List[Dict[str, Any]]:
        """
        Detect and recognize all faces in a frame.
        """

        faces = self.face_model.get(frame)
        results = []

        for face in faces:
            embedding = face.embedding

            if embedding is None:
                continue

            embedding = embedding.astype(np.float32)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            match = self._find_best_match(embedding)

            x1, y1, x2, y2 = face.bbox.astype(int).tolist()

            result = {
                "employee_id": match["employee_id"],
                "name": match["name"],
                "department": match["department"],
                "role": match["role"],
                "similarity": match["similarity"],
                "is_known": match["is_known"],
                "bbox": [x1, y1, x2, y2],
            }

            results.append(result)

        return results

    def _find_best_match(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        best_match: Optional[Dict[str, Any]] = None
        best_similarity = -1.0

        employees = self.face_database.get("employees", {})

        for employee_id, employee_data in employees.items():
            metadata = employee_data["metadata"]
            embeddings = employee_data["embeddings"]

            for item in embeddings:
                stored_embedding = item["embedding"]

                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "employee_id": employee_id,
                        "name": metadata.get("name", ""),
                        "department": metadata.get("department", ""),
                        "role": metadata.get("role", ""),
                        "similarity": round(float(similarity), 3),
                    }

        if best_match is None:
            return {
                "employee_id": None,
                "name": "Unknown",
                "department": "",
                "role": "",
                "similarity": 0.0,
                "is_known": False,
            }

        is_known = best_similarity >= self.recognition_threshold

        if not is_known:
            return {
                "employee_id": None,
                "name": "Unknown",
                "department": "",
                "role": "",
                "similarity": round(float(best_similarity), 3),
                "is_known": False,
            }

        best_match["is_known"] = True
        return best_match

    def _cosine_similarity(self, embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
        return float(np.dot(embedding_1, embedding_2))

    def draw_recognitions(self, frame, recognitions: List[Dict[str, Any]]):
        for recognition in recognitions:
            x1, y1, x2, y2 = recognition["bbox"]
            name = recognition["name"]
            similarity = recognition["similarity"]
            is_known = recognition["is_known"]

            if is_known:
                color = (0, 255, 0)
                label = f"{name} {similarity:.2f}"
            else:
                color = (0, 0, 255)
                label = f"Unknown {similarity:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame