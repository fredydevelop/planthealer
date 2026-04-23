from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path
import numpy as np
import io
import os
import shutil
import uuid

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROFILE_DIR = UPLOAD_DIR / "profile"
SCAN_DIR = UPLOAD_DIR / "scans"

PROFILE_DIR.mkdir(parents=True, exist_ok=True)
SCAN_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "tomato": load_model("models/tomato/tomato_model.keras"),
    "bell_pepper": load_model("models/bell_pepper/retry_bellpepper_model.keras"),
    "potato": load_model("models/potato/new_potato_model.keras"),
}

class_labels = {
    "tomato": [
        "Bacterial_spot",
        "Early_blight",
        "Healthy",
        "Late_blight",
        "Leaf_Mold",
        "Septoria_leaf_spot",
        "Spider_mites",
        "Target_Spot",
        "YellowLeaf_Curl_Virus",
        "mosaic_virus",
    ],
    "bell_pepper": [
        "Bacterial_spot",
        "Healthy",
    ],
    "potato": [
        "Early_blight",
        "Late_blight",
        "Healthy",
    ],
}

@app.get("/")
def root():
    return {"message": "PlantHealer API is running"}

@app.post("/predict")
async def predict(
    plant: str = Form(...),
    file: UploadFile = File(...),
):
    plant = plant.strip().lower()

    if plant not in models:
        raise HTTPException(status_code=400, detail="Invalid plant type")

    model = models[plant]

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_label = class_labels[plant][predicted_class_index]

    return {
        "plant": plant,
        "prediction": predicted_label,
        "prediction_index": predicted_class_index,
        "confidence": confidence,
    }

@app.post("/upload/profile-photo")
async def upload_profile_photo(
    request: Request,
    uid: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{uid}_{uuid.uuid4().hex}{ext}"
    file_path = PROFILE_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = str(request.base_url) + f"uploads/profile/{filename}"

    return {
        "message": "Profile photo uploaded successfully",
        "image_url": image_url,
    }

@app.post("/upload/scan-image")
async def upload_scan_image(
    request: Request,
    uid: str = Form(...),
    plant: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{uid}_{plant}_{uuid.uuid4().hex}{ext}"
    file_path = SCAN_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = str(request.base_url) + f"uploads/scans/{filename}"

    return {
        "message": "Scan image uploaded successfully",
        "image_url": image_url,
    }
