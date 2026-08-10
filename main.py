import os
import urllib.request
from ultralytics import YOLO


MODEL_PATH = "yolov8n-face.pt"


def download_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading YOLO face model...")

        url = "LINK_DIRECT_MODEL"

        urllib.request.urlretrieve(
            url,
            MODEL_PATH
        )

        print("Model downloaded successfully")


download_model()


model = YOLO(MODEL_PATH)
