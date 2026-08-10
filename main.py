from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


app = FastAPI(
    title="AI Face Detection API",
    version="1.0.0"
)


# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


@app.get("/")
def home():
    return {
        "status": "AI Face Detection API Running",
        "developer": "Kazem Delsooz",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    if not image_bytes:
        return {
            "error": "Empty image"
        }

    # Convert image bytes to PIL
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # PIL -> NumPy
    img = np.array(image)

    # RGB -> Gray
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Copy image
    result = img.copy()

    # Draw rectangles
    for (x, y, w, h) in faces:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    # Convert result to PIL
    output = Image.fromarray(result)

    # Save to memory
    buffer = BytesIO()

    output.save(
        buffer,
        format="JPEG",
        quality=95
    )

    buffer.seek(0)

    # Return processed image
    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "X-Faces-Detected": str(len(faces))
        }
    )
