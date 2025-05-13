import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
import traceback
from torchreid import data
from PIL import Image
from torchvision import transforms


# Load YOLOv8 (people detection)
model_yolo = YOLO("yolov8n.pt")  # Can be yolov8s.pt too

# Load OSNet model for ReID
model_reid = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model_reid.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_reid = model_reid.to(device)

""" # Define transform for OSNet (resize + normalize)
transform = data.transforms.build_transforms(
    is_train=False,
    height=256,
    width=128
) """

transform = transforms.Compose([
    transforms.Resize(size=(256, 128), interpolation=Image.BILINEAR, max_size=None, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open webcam
cap = cv2.VideoCapture(0)

# For identity assignment
known_features = []
next_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    frame.flags.writeable = False
    results = model_yolo(frame)
    
    detections = results[0].boxes

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls)
            if cls_id != 0:
                continue  # Skip non-person classes

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            try:
                img_tensor = transform(Image.fromarray(person_crop)).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = self.model_reid(img_tensor).cpu().numpy()

                # Match with known features
                matched_id = None
                for idx, known_feat in enumerate(known_features):
                    dist = np.linalg.norm(feat - known_feat)
                    if dist < 25:
                        matched_id = idx
                        break

                if matched_id is None:
                    known_features.append(feat)
                    matched_id = next_id
                    next_id += 1

                frame.flags.writeable = True

                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {matched_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print("Error processing person crop:", e)
                traceback.print_exc()

    cv2.imshow("YOLOv8 + OSNet ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
