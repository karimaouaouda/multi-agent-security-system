import json
import struct

import cv2
import torch
from ultralytics import YOLO
import base64

from client import Server


class Camera:
    def __init__(self, shared_data=None):
        self.checked_before = 0
        self.server = None

        self.shared_data = shared_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolo11n.pt').to(self.device)

        self.places = {
            'director_office' : {
                'box': {
                    'x1': 0,
                    'y1': 0,
                    'x2': 250,
                    'y2': 550,
                }
            },
            'agent':{
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
        # self.transform = transforms.Compose([
        #     transforms.Resize(size=(256, 128), interpolation=Image.BILINEAR, max_size=None, antialias=True),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        self.known_features = []

    async def run(self, shared_data, server:Server):
        self.shared_data = shared_data
        cap = cv2.VideoCapture(0)
        self.server = server
        persons = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            self.checked_before = self.checked_before - 1
            frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            frame = await self.handle_frame(frame)

            frame_to_show = frame
            cv2.imshow('frame', frame_to_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def brighten_zone(self, frame, box, brightness_increase=40):
        # Extract coordinates
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        return cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def need_to_recheck(self):
        return self.checked_before <= 0



    def get_person_label(self, person):
        pass

    def is_inside(self, person_box, special_place_box):
        px1, py1, px2, py2 = person_box
        sx1, sy1, sx2, sy2 = special_place_box

        return px1 >= sx1 and py1 >= sy1 and px2 <= sx2 and py2 <= sy2
    async def handle_frame(self, frame):
        frame.flags.writeable = False
        results = self.model.track(frame, classes=[0])
        frame.flags.writeable = True



        suffix = ""
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            fid = box.id

            for name, info in self.places.items():
                frame = self.brighten_zone(frame, info['box'])
                if self.is_inside(info['box'].values(), map(int, box.xyxy[0].tolist())):
                    suffix = f" {fid} in {name}"
                    frame = cv2.putText(frame, suffix, (x1, y1 - 10), 2, 1, (0, 255, 0), 2 )

            if self.checked_before <= 0:
                self.checked_before = 30
                # await self.match(frame, [x1, y1, x2, y2], fid)

        frame = results[0].plot()

        return frame

    async def match(self, frame, xis:list[int], fid):
        x1, y1, x2, y2 = xis
        person_crop = frame[y1:y2, x1:x2]
        retval, buffer = cv2.imencode('.jpg', person_crop)
        jpg_as_text = base64.b64encode(buffer)
        msg = json.dumps({
            'image': f"{jpg_as_text.decode()}\n"
        })
        await self.server.ping(struct.pack('!I', len(msg.encode('utf-8'))))
        await self.server.ping(msg.encode('utf-8'))
        raw_len = await self.server.ws_reader.readexactly(4)
        msg_len = struct.unpack('!I', raw_len)[0]
        data = await self.server.ws_reader.readexactly(msg_len)
        try:
            message = json.loads(data.decode())
            print("Received:", data.decode().strip())
        except json.decoder.JSONDecodeError:
            print("Received invalid JSON")



        # for box in results[0].boxes:
        #     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        #     person_crop = frame[y1:y2, x1:x2]
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # img_tensor = self.transform(Image.fromarray(person_crop)).unsqueeze(0).to(self.device)

            # self.get_person_label()

            # with torch.no_grad():
            #     feat = self.model_reid(img_tensor).cpu().numpy()
            #
            # # Match with known features
            # matched_id = None
            # for idx, known_feat in enumerate(self.known_features):
            #     dist = np.linalg.norm(feat - known_feat)
            #     if dist < 25:
            #         matched_id = idx
            #         break
            #
            # if matched_id is None:
            #     self.known_features.append(feat)
            #     matched_id = self.next_id
            #     self.next_id += 1
            #
            # label = self.matches[f"{matched_id}"] if f"{matched_id}" in self.matches else matched_id
            #
            # cv2.putText(frame,
            #             f"person : {label}",
            #             (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6,
            #             (0, 255, 0),
            #             2)

