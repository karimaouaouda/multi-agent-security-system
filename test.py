from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import os
from PIL import Image
import numpy as np
import pickle

# ==================== Setup ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Face detection
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)

# Face embedding model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==================== Load known faces ====================
dataset_path = 'storage/files'  # folder: dataset/<person_name>/*.jpg

known_embeddings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = model(face.unsqueeze(0).to(device))
                known_embeddings.append(embedding.squeeze(0))
                known_names.append(person_name)

# Stack for fast comparison
known_embeddings = torch.stack(known_embeddings)

# ==================== Real-Time Recognition ====================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect face
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn(img)
        if faces is not None:
            for i, face in enumerate(faces):
                if face is None:
                    continue
                with torch.no_grad():
                    embedding = model(face.unsqueeze(0).to(device))
                # Compare with known embeddings (cosine similarity)
                diff = known_embeddings - embedding
                distances = torch.norm(diff, dim=1)
                min_dist, idx = torch.min(distances, dim=0)

                # Threshold (you can tune this)
                threshold = 0.9
                name = known_names[idx] if min_dist < threshold else 'Unknown'

                # Draw box and name
                box = boxes[i].astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} ({min_dist:.2f})', (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('FaceNet Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
