from ultralytics import YOLO
import torch
import cv2

model = YOLO("yolov11n-face.pt")

model.cuda()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    img = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

    # Inference (GPU automatically used if model is on CUDA)
    results = model(img)

    # Parse results
    annotated_frame = results[0].plot()

    # Draw rectangles
    
        

    # Show result
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

