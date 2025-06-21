import json
import struct
import cv2
import torch
from ultralytics import YOLO
import base64
from .recognizer import Recognizer
from .utils import FrameHandler


class Camera:
    def __init__(self, shared_data=None):
        self.checked_before = 0
        self.server = None
        self.handler = FrameHandler()

        self.shared_data = shared_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolo11n.pt').to(self.device)
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

    async def run(self, shared_data, server=None):
        cap = cv2.VideoCapture(2)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            self.checked_before = self.checked_before - 1

            frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

            frame = await self.handler.process_frame(frame)

            person, sim, is_same = self.recognizer.recognize(frame)

            if is_same:
                pass # Todo complete this

            frame_to_show = frame
            cv2.imshow('frame', frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


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
