from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from ultralytics import YOLO

from PIL import Image
from io import BytesIO

import numpy as np
import os
import urllib.request


app = FastAPI(
    title="Face Detection API",
    version="1.0"
)


# ==============================
# Face Model
# ==============================

MODEL_PATH = "yolov8n-face.pt"


MODEL_URL = (
    "https://huggingface.co/Autsadin/yolov8-face/"
    "resolve/main/yolov8n-face.pt"
)



def download_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading face model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("Model downloaded")

    else:

        print("Model already exists")



download_model()



# ==============================
# Load Model
# ==============================

print("Loading YOLO Face Model...")


model = YOLO(
    MODEL_PATH
)


print("Face Model Ready")



# ==============================
# Home
# ==============================

@app.get("/")
def home():

    return {

        "status":
        "Face Detection API Running",

        "model":
        "YOLOv8 Face Detection"

    }



# ==============================
# Health
# ==============================

@app.get("/health")
def health():

    return {

        "status":
        "healthy"

    }




# ==============================
# Detection
# ==============================

@app.post("/detect")
async def detect(

    file: UploadFile = File(...)

):


    image_bytes = await file.read()



    image = Image.open(

        BytesIO(image_bytes)

    ).convert("RGB")



    img = np.array(image)



    results = model(

        img,

        conf=0.5

    )



    result_image = results[0].plot()



    output = Image.fromarray(

        result_image

    )



    buffer = BytesIO()



    output.save(

        buffer,

        format="JPEG",

        quality=95

    )



    buffer.seek(0)



    faces = len(

        results[0].boxes

    )



    return StreamingResponse(

        buffer,

        media_type="image/jpeg",

        headers={

            "faces-detected":
            str(faces)

        }

    )
