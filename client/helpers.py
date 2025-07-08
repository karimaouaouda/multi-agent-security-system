import os
import datetime
import cv2
import random

colors = {}

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_writer(camera_id, fps, frame_size, output_dir="footage"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/cam_{camera_id}_{get_timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    return writer, filename

def get_color(track_id):
    if track_id not in colors:
        colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return colors[track_id]

def is_person_in_zone(person_bbox, zone_bbox):
    """
    Determines if a person's bounding box is in a zone.

    Args:
        person_bbox (tuple): (x1, y1, x2, y2) coordinates of the person's bounding box.
        zone_bbox (tuple): (x1, y1, x2, y2) coordinates of the zone's bounding box.

    Returns:
        bool: True if the person's bounding box touches or is inside the zone's bounding box.
    """
    px1, py1, px2, py2 = person_bbox
    zx1, zy1, zx2, zy2 = zone_bbox

    # Check for intersection or touch
    horizontal_overlap = px1 <= zx2 and px2 >= zx1
    vertical_overlap = py1 <= zy2 and py2 >= zy1

    return horizontal_overlap and vertical_overlap

def draw_box(frame, box, label=None, color=(0, 255, 0)):
    """
    Draws a bounding box on the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        box (tuple): (x1, y1, x2, y2) coordinates of the bounding box.
        label (str, optional): Label to display above the box. Defaults to None.
        color (tuple, optional): Color of the box in BGR format. Defaults to green.
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def cut_image(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def is_face_in_person(person_box, face_box):
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box

    return (px1 <= fx1 <= px2 and py1 <= fy1 <= py2) or (px1 <= fx2 <= px2 and py1 <= fy2 <= py2)
