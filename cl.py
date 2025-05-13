import os
import cv2
import numpy as np
import torch
from deepface import DeepFace
from ultralytics import YOLO
import pickle




# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Load YOLOv8 model for person detection
yolo_model = YOLO("yolov8n.pt").to(device)

# Build face representations for known people
print("[INFO] Building known face embeddings...")
known_faces = {}
base_dir = "storage/files"  # Folder structure: known_faces/person_name/image.jpg
stored_embeddings_path = 'storage/pkl/faces.pkl'

if os.path.exists(stored_embeddings_path):
    with open(stored_embeddings_path, "rb") as f:
        known_faces = pickle.load(f)
else:
    print("[INFO] Encoding known faces...")
    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_file)
                try:
                    rep = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        detector_backend="retinaface",
                        enforce_detection=True
                    )[0]["embedding"]
                    embeddings.append(rep)
                    print(f"[+] Encoded {person}/{img_file}")
                except Exception as e:
                    print(f"[!] Skipped {img_file}: {e}")
        if embeddings:
            known_faces[person] = embeddings
    # Save embeddings to file
    with open(stored_embeddings_path, "wb") as f:
        pickle.dump(known_faces, f)


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Person detection with YOLO
    frame.flags.writeable = False
    results = yolo_model(frame, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id != 0:  # 0 = person
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        person_crop = frame[y1:y2, x1:x2]

        try:
            # Face embedding from cropped region
            embedding = DeepFace.represent(
                img_path=person_crop,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="retinaface"
            )[0]["embedding"]
            frame.flags.writeable = True

            # Compare with known faces
            identity = "Unknown"
            min_dist = float("inf")
            for known in known_faces:
                dist = np.linalg.norm(np.array(known["rep"]) - np.array(embedding))
                if dist < 10 and dist < min_dist:  # Facenet threshold ~10
                    min_dist = dist
                    identity = known["name"]

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        except Exception:
            pass  # No face found

    cv2.imshow("DeepFace + YOLOv8 (GPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
