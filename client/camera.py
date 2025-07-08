import cv2
import torch
from .recognizer import Recognizer
from .utils import FrameHandler
from concurrent.futures import ThreadPoolExecutor
import asyncio
from ultralytics import YOLO
from threading import Thread
import random
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np

from client.helpers import *
class Camera:
    def __init__(self, shared_data=None):
        self.checked_before = 0
        self.handler = FrameHandler()

        self.id = random.randint(1, 500)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.get_event_loop()
        self.person_features_extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='~/.torchreid/osnet_x1_0_msmt17.pth',
            device='cuda'  # or 'cpu'
        )
        self.persons_embddings = {}

        self.karim = "from main"

        self.faces = {}
        self.persons = {}

        self.footage_writer = create_writer(self.id, 30, (640, 480), output_dir="footage")[0]


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
        cap = cv2.VideoCapture(1)  # Change to 0 for the default camera
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        person_tracker = YOLO('yolo11n.pt')
        person_tracker = person_tracker.to(device)
        captured_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # self.footage_writer.write(frame)

            captured_frames = captured_frames + 1

            person_tracking_results = person_tracker.track(frame, classes=[0])

            frame = self.process_frame_blocking(frame) # THIS WILL CHANGE THE SELF.FACES MEMBER

            # fps = int(cap.get(cv2.CAP_PROP_FPS))

            for result in person_tracking_results:
                if result.boxes.id is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
            
                for box, track_id in zip(boxes, ids):

                    self.persons[f"{track_id}"] = {
                        "name": "unknown",
                        "box": box,
                        "by": "none",
                        "similarity": -1,
                        "is_same": False
                    }

                    for face in self.faces:
                        if is_face_in_person(box, face["box"]):
                            self.persons[f"{track_id}"]["name"] = face["name"]
                            self.persons[f"{track_id}"]["similarity"] = face["similarity"]
                            self.persons[f"{track_id}"]["is_same"] = face["is_same"]
                            self.persons[f"{track_id}"]["by"] = "face"

                            if face['similarity'] > 0.7:
                                self.puch_person_embedding(frame, box, face['name'])
                            break

                    if self.persons[f"{track_id}"]["by"] == "none" or self.persons[f"{track_id}"]["similarity"] < 0.7:
                        self.recognize_person(frame, box, track_id) # try recognize the person with his body

                    x1, y1, x2, y2 = map(int, box)
                    color = get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame,f"{self.persons[f"{track_id}"]["name"]} ({self.persons[f"{track_id}"]['by']})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

            if captured_frames % 10 == 0:
                thread = Thread(target=self.process_frame_blocking, args=(frame.copy(),))
                thread.start()

            frame_to_show = frame
            cv2.imshow('frame', frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def puch_person_embedding(self, frame, box, name):
        image = cut_image(frame, box)
        embedding = self.person_features_extractor(image)
        if name in self.persons_embddings:
            if len(self.persons_embddings[name]) > 4:
                self.persons_embddings[name][4] = embedding
            else:
                self.persons_embddings[name].append(embedding)
        else:
            self.persons_embddings[name] = [embedding] 

    def recognize_person(self, frame, box, id):
        if self.persons[f"{id}"]["similarity"] > 0.7:
            return
        image = cut_image(frame, box)
        threshhold = 0.8
        features = self.person_features_extractor(image)
        if features is None:
            return
        features = features.cpu().numpy()[0]
        best_sim = -1
        best_sim_person = "unknown"
        for name, embeddings in self.persons_embddings.items():
            for embedding in embeddings:
                sim = np.dot(embedding.cpu().numpy()[0], features) / (np.linalg.norm(embedding.cpu().numpy()[0]) * np.linalg.norm(features))
                if sim > threshhold and sim > best_sim:
                    best_sim = sim
                    best_sim_person = name

        if best_sim > threshhold:
            self.persons[f"{id}"]["name"] = best_sim_person
            self.persons[f"{id}"]['similarity'] = best_sim
            self.persons[f"{id}"]['by'] = 'person'
            self.persons[f"{id}"]['is_same'] = True


    def process_frame_blocking(self, frame):
        # Do heavy detection & recognition work here
        frame = self.handler.process_frame(frame)  # make sure this is blocking

        faces_results = self.recognizer.recognize(frame)

        if faces_results is not None:
            self.faces = faces_results
            for result in faces_results:
                print(f"person: {result["name"]}, similarity: {result["similarity"]}")
                x1, y1, x2, y2 = map(int, result["box"])
                draw_box(frame, (x1, y1, x2, y2), label=f"{result["name"]} {result["similarity"]}%", color=(0, 255, 0))

        return frame
    
    async def handle_frame_async(self, frame):
        processed_result = await self.loop.run_in_executor(
            self.executor, self.process_frame_blocking, frame
        )

        if processed_result:
            person, similarity, is_same, processed_frame = processed_result
            if is_same:
                print(f"[Match] {person} with similarity {similarity}")

            print(f"person {person}, sim : {similarity}, is_same : {is_same}")

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
