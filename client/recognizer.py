import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU


def get_face_embedding(image: str | np.ndarray):
    """Extract face embedding from an image"""

    if isinstance(image, str):
        # check if path exists
        if not os.path.exists(image):
            raise FileNotFoundError(f'{image} does not exist')

        image = cv2.imread(image)

    elif not isinstance(image, np.ndarray):
        raise TypeError('image must be str or np.ndarray')

    if image is None:
        raise ValueError(f"Could not read image: {image}")

    faces = app.get(image)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    return faces[0].embedding


def compare_faces(emb1, emb2, threshold=0.65):  # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold


class Recognizer:
    def __init__(self, dataset_path: str):
        self.dataset: dict = {}

        self.threshold = 80

        self.dataset_path = dataset_path

        self.setup()

    def recognize(self, picture: str | np.ndarray):
        if isinstance(picture, str):
            if not os.path.exists(picture):
                raise FileNotFoundError(picture)
            picture = cv2.imread(picture)
        elif not isinstance(picture, np.ndarray):
            raise TypeError('image must be str or np.ndarray')

        if picture is None:
            raise ValueError(f"Could not read image: {picture}")

        try:
            embedding = get_face_embedding(picture)
        except ValueError as e:
            print('error', e)
            return None

        for person, images in self.dataset.items():
            for image in images:
                similarity, is_same = compare_faces(embedding, image, self.threshold)
                if is_same:
                    return person, similarity

        return None, None

    def setup(self):
        persons = os.listdir(self.dataset_path)
        for person in persons:
            self.dataset[person] = []

            images = os.listdir(os.path.join(self.dataset_path, person))
            for image in images:
                embedding = get_face_embedding(os.path.join(self.dataset_path, person, image))
                self.dataset[person].append(embedding)
                print(f"loaded {person} image")
