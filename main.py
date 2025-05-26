# from client.app import Application
# import asyncio
#
# if __name__ == '__main__':
#     client = Application()
#     asyncio.run(client.run())
#

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU


def get_face_embedding(image_path, is_path:bool=True):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path) if is_path else image_path
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    return faces[0].embedding


def compare_faces(emb1, emb2, threshold=0.65):  # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold





# camera record
cap = cv2.VideoCapture(0)

base_image = "./storage/images/sulaimen1.jpg"
emb1 = get_face_embedding(base_image)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        # Get embeddings

        emb2 = get_face_embedding(frame, False)

        # Compare faces
        similarity_score, is_same_person = compare_faces(emb1, emb2)

        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Same person? {'YES' if is_same_person else 'NO'}")

    except Exception as e:
        print(f"Error: {str(e)}")



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# extract the picture / frame and compare the face image with anotehr

# if there is a face draw a rectangle on thhe face and say yes if he
# is the person, no otherwise