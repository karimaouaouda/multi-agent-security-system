import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU



def load_or_save_pickle(data_dict, filepath):
    """
    If the pickle file exists, load and return the dict.
    Otherwise, save the provided dict and return it.
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            print(f"Loading data from {filepath}")
            return pickle.load(f)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
            print(f"Saving new data to {filepath}")
            return data_dict

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
            return False, False, False

        best_sim = -1
        best_sim_per = None
        for person, images in self.dataset.items():
            for image in images:
                similarity, is_same = compare_faces(embedding, image, self.threshold)
                if is_same:
                    return person, similarity, True
                elif similarity > best_sim:
                    best_sim = similarity
                    best_sim_per = person

        return best_sim_per, best_sim, False

    def setup(self):
        if os.path.exists('data.pkl'):
            self.load_faces('data.pkl')
            return
         
        persons = os.listdir(self.dataset_path)
        for person in persons:
            self.dataset[person] = []

            images = os.listdir(os.path.join(self.dataset_path, person))
            for image in images:
                embedding = get_face_embedding(os.path.join(self.dataset_path, person, image))
                self.dataset[person].append(embedding)
                print(f"loaded {person} image")

        self.save_faces('data.pkl')
    
    def save_faces(self, filename):
         with open(filename, 'wb') as f:
            pickle.dump(self.dataset, f)
            print(f"Saving new data to {filename}")

    def load_faces(self, filename):
        with open(filename, 'rb') as f:
            print(f"Loading data from {filename}")
            self.dataset = pickle.load(f)


class FaceRecognitionResult:
    def __init__(self, recognizer:Recognizer):
        self.recognizer = recognizer
        self.recognitions = []
