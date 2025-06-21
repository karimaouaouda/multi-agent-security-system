import numpy as np

class Person:
    def __init__(self, name: str | None, ):
        self.name = name
        self.box = {
            'x1': -1,
            'y1': -1,
            'x2': -1,
            'y2': -1
        }

    def set_name(self, name: str | None):
        self.name = name


class FrameHandler:
    def __init__(self):
        self.frame:np.ndarray|None = None

    def update_frame(self, frame:np.ndarray):
        self.frame = frame


    async def process_frame(self, frame) -> np.ndarray:
        return frame

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