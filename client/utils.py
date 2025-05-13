import cv2
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import torch
import os
from torchvision import transforms
from PIL import Image

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


class FaceRecognizer:
    def __init__(self, faces_path, mrcnn=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.faces_path = faces_path
        self.detector = mrcnn
        self.faces = {}
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.load_faces2()

    def recognize(self, image):
        pass

    def load_faces2(self):
        print('load data...')
        persons_images = os.listdir(self.faces_path)
        for image_name in persons_images:
            image = cv2.imread(os.path.join(self.faces_path, image_name), cv2.IMREAD_COLOR_RGB)
            faces = self.detector(image)

            if faces is None:
                return None

            embeddings = self.model(faces.to(self.device))  # batch of embeddings

            self.faces[image_name] = embeddings
    def load_faces(self):
        # Detect faces
        persons = os.listdir(self.faces_path)

        for person in persons:
            person_images = os.listdir(os.path.join(self.faces_path, person))
            for person_image in person_images:
                image = cv2.imread(os.path.join(self.faces_path, person, person_image), cv2.IMREAD_COLOR_RGB)
                image = Image.fromarray(image)
                faces = self.detector(image)

                if faces is None:
                    return None

                embeddings = self.model(faces.to(self.device))  # batch of embeddings

                self.faces[person] = embeddings

        print(f"finish with : {len(self.faces)} faces")

