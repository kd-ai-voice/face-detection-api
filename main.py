from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from ultralytics import YOLO

from PIL import Image
from io import BytesIO

import numpy as np
import os
import urllib.request

import torch


# کاهش مصرف RAM و CPU
torch.set_num_threads(1)


app = FastAPI(
    title="Kazem Delsooz AI Vision API",
    version="2.0"
)



# =====================================================
# Model Settings
# =====================================================

FACE_MODEL_PATH = "yolov8n-face.pt"


FACE_MODEL_URL = (
    "https://huggingface.co/Autsadin/yolov8-face/"
    "resolve/main/yolov8n-face.pt"
)



# =====================================================
# Download Face Model
# =====================================================

def download_face_model():

    if not os.path.exists(FACE_MODEL_PATH):

        print("Downloading Face Model...")

        urllib.request.urlretrieve(
            FACE_MODEL_URL,
            FACE_MODEL_PATH
        )

        print("Face Model Downloaded")

    else:

        print("Face Model Exists")



download_face_model()



# =====================================================
# Lazy Loading Models
# =====================================================

face_model = None
object_model = None



def get_face_model():

    global face_model


    if face_model is None:

        print("Loading YOLO Face Model...")

        face_model = YOLO(
            FACE_MODEL_PATH
        )


        print("Face Model Loaded")


    return face_model




def get_object_model():

    global object_model


    if object_model is None:

        print("Loading YOLO Object Model...")

        object_model = YOLO(
            "yolov8n.pt"
        )


        print("Object Model Loaded")


    return object_model




# =====================================================
# Home
# =====================================================

@app.get("/")
def home():

    return {

        "status":
        "AI Vision API Running",

        "developer":
        "Kazem Delsooz",

        "models":[

            "YOLOv8 Face Detection",

            "YOLOv8 Object Detection"

        ]

    }



# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health():

    return {

        "status":"healthy"

    }




# =====================================================
# Face Detection
# =====================================================

@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...)
):


    image_bytes = await file.read()



    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")



    img = np.array(image)



    model = get_face_model()



    results = model(

        img,

        conf=0.5

    )



    output_image = results[0].plot()



    output = Image.fromarray(
        output_image
    )



    buffer = BytesIO()



    output.save(

        buffer,

        format="JPEG",

        quality=95

    )



    buffer.seek(0)



    count = len(
        results[0].boxes
    )



    return StreamingResponse(

        buffer,

        media_type="image/jpeg",

        headers={

            "faces-detected":
            str(count)

        }

    )





# =====================================================
# Object Detection
# =====================================================

@app.post("/object")
async def detect_object(

    file: UploadFile = File(...)

):


    image_bytes = await file.read()



    image = Image.open(

        BytesIO(image_bytes)

    ).convert("RGB")



    img = np.array(image)




    model = get_object_model()




    results = model(

        img,

        conf=0.35

    )




    output_image = results[0].plot()



    output = Image.fromarray(

        output_image

    )



    buffer = BytesIO()



    output.save(

        buffer,

        format="JPEG",

        quality=95

    )



    buffer.seek(0)



    count = len(

        results[0].boxes

    )



    return StreamingResponse(

        buffer,

        media_type="image/jpeg",

        headers={

            "objects-detected":
            str(count)

        }

    )
