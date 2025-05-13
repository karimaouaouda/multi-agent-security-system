from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
from deepface import DeepFace
import logging
logging.getLogger("deepface").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load YOLOv8 model
print('loading YOLOv8 model...')
yolo_model = YOLO("yolov11n-face.pt").to(device)  # Use yolov8n-face if available

# Build face database
def build_face_db(db_path="known_faces"):
    face_db = []
    for person in os.listdir(db_path):
        person_dir = os.path.join(db_path, person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                face_db.append({"name": person, "embedding": embedding})
                print(f"[INFO] Added {person}/{img_name} to DB")
            except Exception as e:
                print(f"[ERROR] Could not process {img_path}: {e}")
    return face_db

# Compare face with known embeddings
def recognize_face(face_img_array, face_db, threshold=6.7):  # TEMP: higher threshold to test
    try:
        # Get embedding for detected face
        embedding = DeepFace.represent(img_path=face_img_array, model_name="Facenet", enforce_detection=True)[0]["embedding"]

        closest_match = "Unknown"
        min_distance = float("inf")

        for entry in face_db:
            dist = np.linalg.norm(np.array(embedding) - np.array(entry["embedding"]))
            print(f"[DEBUG] Compared with {entry['name']}: distance = {dist:.2f}")
            if dist < min_distance:
                min_distance = dist
                closest_match = entry["name"]

        print(f"[DEBUG] Closest match: {closest_match}, distance: {min_distance:.2f}")
        if min_distance < threshold:
            return closest_match
    except Exception as e:
        print(f"[ERROR] DeepFace issue: {e}")
    return "Unknown"

# Real-time face recognition
def recognize_from_video(face_db):
    cap = cv2.VideoCapture(0)  # Webcam

    if not cap.isOpened():
        print("[ERROR] Webcam not detected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame.flags.writeable = False
        results = yolo_model(frame)

        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                # Skip small/empty crops
                if face_crop.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue


                # Recognizee
                name = recognize_face(frame, face_db)
                frame.flags.writeable = True
                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def collect(name:str):
    cap = cv2.VideoCapture(0)  # Webcam

    count = 0
    save_path = f"yolo_dataset/{name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Collecting images... Press 'q' to quit.")
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)  # Flip the frame horizontally
        frame_index += 1
        if frame_index % 4 != 0:  # Process every 5th frame
            continue
        results = yolo_model(frame)
        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                # Skip small/empty crops
                if face_crop.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                count += 1
                cv2.imwrite(f"{save_path}/img_{count}.jpg", face_crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Collecting {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collecting Images", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Image collection done!")

def recognize():
    face_db = build_face_db('yolo_dataset')
    print("[INFO] Starting real-time face recognition...")
    recognize_from_video(face_db)

# Main
if __name__ == "__main__":
    recognize()