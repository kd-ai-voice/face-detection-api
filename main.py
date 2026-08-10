from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

from PIL import Image
from io import BytesIO

import numpy as np
import os
import urllib.request


app = FastAPI(
    title="Kazem Delsooz AI Computer Vision API",
    version="2.0.0"
)


# =========================================================
# MODEL PATHS
# =========================================================

FACE_MODEL_PATH = "yolov8n-face.pt"


FACE_MODEL_URL = (
    "https://huggingface.co/Autsadin/yolov8-face/"
    "resolve/main/yolov8n-face.pt"
)


# =========================================================
# DOWNLOAD FACE MODEL
# =========================================================

def download_face_model():

    if not os.path.exists(FACE_MODEL_PATH):

        print("Downloading YOLOv8 Face model...")

        urllib.request.urlretrieve(
            FACE_MODEL_URL,
            FACE_MODEL_PATH
        )

        print("Face model downloaded successfully")

    else:

        print("Face model already exists")


download_face_model()


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading Face Detection model...")

face_model = YOLO(
    FACE_MODEL_PATH
)

print("Face Detection model loaded")


print("Loading Object Detection model...")

# Ultralytics automatically downloads yolov8n.pt
object_model = YOLO(
    "yolov8n.pt"
)

print("Object Detection model loaded")


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status": "AI Computer Vision API Running",
        "developer": "Kazem Delsooz",
        "models": [
            "YOLOv8 Face Detection",
            "YOLOv8 Object Detection"
        ]
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# FACE DETECTION
# =========================================================

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    img = np.array(image)


    results = face_model(
        img,
        conf=0.5
    )


    annotated_image = results[0].plot()


    output = Image.fromarray(
        annotated_image
    )


    buffer = BytesIO()


    output.save(
        buffer,
        format="JPEG",
        quality=95
    )


    buffer.seek(0)


    face_count = len(
        results[0].boxes
    )


    return StreamingResponse(

        buffer,

        media_type="image/jpeg",

        headers={
            "faces-detected": str(face_count)
        }

    )


# =========================================================
# OBJECT DETECTION
# =========================================================

@app.post("/object")
async def detect_objects(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    img = np.array(image)


    # YOLO Object Detection

    results = object_model(
        img,
        conf=0.35
    )


    annotated_image = results[0].plot()


    output = Image.fromarray(
        annotated_image
    )


    buffer = BytesIO()


    output.save(
        buffer,
        format="JPEG",
        quality=95
    )


    buffer.seek(0)


    object_count = len(
        results[0].boxes
    )


    return StreamingResponse(

        buffer,

        media_type="image/jpeg",

        headers={
            "objects-detected": str(object_count)
        }

    )
