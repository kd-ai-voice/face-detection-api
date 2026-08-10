from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

from PIL import Image
from io import BytesIO

import numpy as np
import os
import urllib.request


app = FastAPI(
    title="YOLOv8 Face Detection API",
    version="1.0.0"
)


# ============================
# Download YOLO Face Model
# ============================

MODEL_PATH = "yolov8n-face.pt"


MODEL_URL = (
    "https://huggingface.co/Autsadin/yolov8-face/"
    "resolve/main/yolov8n-face.pt"
)


def download_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading YOLOv8 Face model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("Model downloaded successfully")



download_model()



# Load YOLO Model

model = YOLO(MODEL_PATH)



# ============================
# Health Check
# ============================

@app.get("/")
def home():

    return {
        "status": "YOLOv8 Face Detection Running",
        "model": "yolov8n-face",
        "developer": "Kazem Delsooz"
    }



@app.get("/health")
def health():

    return {
        "status": "healthy"
    }



# ============================
# Face Detection
# ============================

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...)
):


    image_bytes = await file.read()


    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")


    img = np.array(image)



    # YOLO inference

    results = model(
        img,
        conf=0.5
    )



    # Draw bounding boxes

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
