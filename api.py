from fastapi import FastAPI, File, UploadFile
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import io


app = FastAPI()


# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="PepsTechRnD/egg-quality-yolo11x",
    filename="best.pt"
)

# Load YOLO11x
model = YOLO(model_path)


@app.get("/")
def home():
    return {"status": "Egg Quality API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_array = np.array(image)

    results = model(image_array)

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[class_id],
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })

    return {
        "detections": detections
    }
