from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


app = FastAPI(
    title="AI Face Detection API"
)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


@app.get("/")
def home():
    return {
        "status": "AI Face Detection Running",
        "developer": "Kazem Delsooz"
    }



@app.post("/detect")
async def detect_face(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()


    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")


    img = np.array(image)


    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )


    result = img.copy()


    for x,y,w,h in faces:

        cv2.rectangle(
            result,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            3
        )


    output = Image.fromarray(result)


    buffer = BytesIO()

    output.save(
        buffer,
        format="JPEG"
    )

    buffer.seek(0)


    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "faces-detected":str(len(faces))
        }
    )
