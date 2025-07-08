import cv2
import numpy as np

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def denoise_frame(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)



def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return result



cap = cv2.VideoCapture(0)
ret, frame = cap.read()

cv2.imshow("Enhanced Frame 1", frame)

frame = denoise_frame(sharpen_image(frame))

cv2.imshow("Enhanced Frame", frame)
cv2.waitKey(10000)

cap.release()


""" import torchreid
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from scipy.spatial.distance import cosine
import cv2
import os
import numpy as np
import random

KARIM_DIR = "./storage/test/persons/karim"
NOUNOU_DIR = "./storage/test/persons/nounou"

# Load a pre-trained Re-ID model
torchreid.models.show_avai_models()
model = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='~/.torchreid/osnet_x1_0_msmt17.pth',
    device='cuda'  # or 'cpu'
)

# load two random pics from karim folder
karim_images = os.listdir(KARIM_DIR)
nounou_images = os.listdir(NOUNOU_DIR)
karim_images_len = len(karim_images)
nounou_images_len = len(nounou_images)


for i in range(50):
    fpic_path = os.path.join(KARIM_DIR, karim_images[random.randint(0, karim_images_len - 1)])
    spic_path = os.path.join(KARIM_DIR, karim_images[random.randint(0, karim_images_len - 1)])
    fpic = cv2.imread(
        fpic_path
    )

    spic = cv2.imread(
        spic_path
    )

    fpic_features = model(fpic)
    spic_features = model(spic)

    fpic_features = fpic_features.cpu().numpy()[0]
    spic_features = spic_features.cpu().numpy()[0]

    

    similarity =  np.dot(fpic_features, spic_features) / (np.linalg.norm(fpic_features) * np.linalg.norm(spic_features))

    print(f"Similarity : {i} ( {fpic_path} / {spic_path}):", similarity) """