from ultralytics import YOLO
import torch
import cv2

print(f"using : {"cuda" if torch.cuda.is_available() else "cpu"}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8n.pt")  # Load an official Pose model
model.classes = [0]
model = model.to(device)

special_area = (0, 0, 250, 400)  # You can adjust this

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    

    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

    px1, py1, px2, py2 = special_area
    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
    cv2.putText(frame, "Special Area", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    frame.flags.writeable = False
    results = model.track(source=frame, show=False, classes=[0])
    frame.flags.writeable = True
    for box in results[0].boxes:
        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Class
        class_id = int(box.cls[0])
        # Confidence
        conf = float(box.conf[0])
        # ID (if available)
        track_id = int(box.id[0]) if box.id is not None else -1

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{track_id} {conf:.2f}" if track_id != -1 else f"{conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if px1 < cx < px2 and py1 < cy < py2:
            cv2.putText(frame, "IN AREA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "OUTSIDE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()