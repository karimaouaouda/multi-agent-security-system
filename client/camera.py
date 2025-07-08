import cv2
import torch
from .recognizer import Recognizer
from .utils import FrameHandler
from concurrent.futures import ThreadPoolExecutor
import asyncio
from ultralytics import YOLO
from threading import Thread
import random

colors = {}

def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return colors[track_id]

def cut_image(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]

def is_face_in_person(person_box, face_box):
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box

    return (px1 <= fx1 <= px2 and py1 <= fy1 <= py2) or (px1 <= fx2 <= px2 and py1 <= fy2 <= py2)

class Camera:
    def __init__(self, shared_data=None):
        self.checked_before = 0
        self.server = None
        self.handler = FrameHandler()
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.get_event_loop()

        self.karim = "from main"


        self.shared_data = shared_data
        self.recognizer = Recognizer(dataset_path="./client/data/known_faces")

        self.places = {
            'director_office': {
                'box': {
                    'x1': 0,
                    'y1': 0,
                    'x2': 250,
                    'y2': 550,
                }
            },
            'agent': {
                'box': {
                    'x1': 400,
                    'y1': 100,
                    'x2': 600,
                    'y2': 500,
                }
            }
        }

        self.matches = {
            '0': 'abdou',
            '1': 'karim',
            '2': 'billel'
        }

        self.known_features = []

    def run(self, shared_data, server=None):

        cap = cv2.VideoCapture(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        faces_tracker = YOLO('yolov11n-face.pt')
        person_tracker = YOLO('yolo11n.pt')

        faces_tracker = faces_tracker.to(device)
        person_tracker = person_tracker.to(device)

        captured_frames = 0


        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            captured_frames = captured_frames + 1

            faces_tracking_results = faces_tracker.track(frame)
            person_tracking_results = person_tracker.track(frame, classes=[0])

            for result in person_tracking_results:
                if result.boxes.id is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    color = get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            for result in faces_tracking_results:
                if result.boxes.id is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    color = get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            

            frame = self.handler.process_frame(frame)

            if captured_frames % 10 == 0:
                thread = Thread(target=self.process_frame_blocking, args=(frame.copy(),))
                thread.start()

            

            frame_to_show = frame
            cv2.imshow('frame', frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def process_frame_blocking(self, frame):
        # Do heavy detection & recognition work here
        frame = self.handler.process_frame(frame)  # make sure this is blocking
        person, sim, is_same = self.recognizer.recognize(frame)
        # Can annotate frame here if needed
        return person, sim, is_same, frame
    
    async def handle_frame_async(self, frame):
        processed_result = await self.loop.run_in_executor(
            self.executor, self.process_frame_blocking, frame
        )

        if processed_result:
            person, similarity, is_same, processed_frame = processed_result
            if is_same:
                print(f"[Match] {person} with similarity {similarity}")

            print(f"person {person}, sim : {similarity}, is_same : {is_same}")

    def is_inside(self, person_box, special_place_box):
        px1, py1, px2, py2 = person_box
        sx1, sy1, sx2, sy2 = special_place_box

        return px1 >= sx1 and py1 >= sy1 and px2 <= sx2 and py2 <= sy2

    async def handle_frame(self, frame):
        frame.flags.writeable = False
        # results = self.model.track(frame, classes=[0])
        person, similarity = self.recognizer.recognize(frame)

        if similarity is None:
            return frame

        print(f"person: {person}, similarity: {similarity}")
        frame.flags.writeable = True
        suffix = ""

        return frame
