from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import numpy as np
import io
import os
import shutil
import uuid
import traceback


# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Rice Crop Disease Detection API",
    description="Rice leaf disease detection API",
    version="1.0.0",
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
PROFILE_DIR = UPLOAD_DIR / "profile"
SCAN_DIR = UPLOAD_DIR / "scans"

PROFILE_DIR.mkdir(parents=True, exist_ok=True)
SCAN_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)


# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# RICE MODEL
#
# GitHub:
# models/rice_model.keras
# =========================================================
MODEL_PATH = BASE_DIR / "models" / "rice_model.keras"


CLASS_LABELS = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]


DISPLAY_NAMES = {
    "Bacterial Leaf Blight": "Rice Bacterial Leaf Blight",
    "Brown Spot": "Rice Brown Spot",
    "Healthy Rice Leaf": "Healthy Rice Leaf",
    "Leaf Blast": "Rice Leaf Blast",
    "Leaf scald": "Rice Leaf Scald",
    "Sheath Blight": "Rice Sheath Blight",
}


DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "cause": (
            "A bacterial rice disease commonly associated with "
            "Xanthomonas oryzae pv. oryzae. Warm and humid conditions "
            "can encourage its development and spread."
        ),
        "recommendation": (
            "Use clean planting materials, avoid excessive nitrogen "
            "application, maintain good field sanitation, and use "
            "resistant rice varieties where available."
        ),
    },

    "Brown Spot": {
        "cause": (
            "A fungal disease commonly associated with Bipolaris oryzae. "
            "Poor soil fertility, infected seeds, plant stress, and humid "
            "conditions can favour its development."
        ),
        "recommendation": (
            "Improve crop nutrition, use healthy seeds, remove infected "
            "crop residues, reduce plant stress, and apply appropriate "
            "locally approved control measures when necessary."
        ),
    },

    "Healthy Rice Leaf": {
        "cause": (
            "No visible rice leaf disease symptoms were detected by the model."
        ),
        "recommendation": (
            "Continue regular crop monitoring, proper irrigation, "
            "balanced fertilization, and good field sanitation."
        ),
    },

    "Leaf Blast": {
        "cause": (
            "A fungal rice disease commonly caused by Magnaporthe oryzae. "
            "High humidity, prolonged leaf wetness, and excessive nitrogen "
            "can favour the disease."
        ),
        "recommendation": (
            "Manage infected crop residues, avoid excessive nitrogen, "
            "maintain appropriate water management, and use resistant "
            "rice varieties where available."
        ),
    },

    "Leaf scald": {
        "cause": (
            "A fungal rice disease commonly associated with "
            "Microdochium oryzae. Humid conditions and infected crop "
            "residues may encourage disease development."
        ),
        "recommendation": (
            "Use healthy planting materials, manage infected residues, "
            "maintain balanced crop nutrition, and practise good "
            "field sanitation."
        ),
    },

    "Sheath Blight": {
        "cause": (
            "A fungal disease commonly caused by Rhizoctonia solani. "
            "Dense crop stands, high humidity, warm temperatures, and "
            "excessive nitrogen can favour its development."
        ),
        "recommendation": (
            "Avoid excessive nitrogen, maintain suitable plant spacing, "
            "manage infected crop residues, and use appropriate locally "
            "approved control measures when necessary."
        ),
    },
}


# =========================================================
# LOAD MODEL
# =========================================================
rice_model = None
MODEL_LOAD_ERROR = None

try:
    print("MODEL PATH:", MODEL_PATH)
    print("MODEL EXISTS:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Rice model not found at {MODEL_PATH}"
        )

    # Inference only; no optimizer/training state is needed.
    rice_model = load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Rice model loaded successfully")
    print("Input shape:", rice_model.input_shape)
    print("Output shape:", rice_model.output_shape)

    if rice_model.output_shape[-1] != len(CLASS_LABELS):
        raise ValueError(
            f"Model returns {rice_model.output_shape[-1]} classes "
            f"but CLASS_LABELS contains {len(CLASS_LABELS)} classes."
        )

except Exception as error:
    MODEL_LOAD_ERROR = f"{type(error).__name__}: {error}"
    rice_model = None

    print("MODEL LOAD ERROR:", MODEL_LOAD_ERROR)
    traceback.print_exc()


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Rice Crop Disease Detection API is running",
        "model_loaded": rice_model is not None,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_error": MODEL_LOAD_ERROR,
        "number_of_classes": len(CLASS_LABELS),
    }


# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    if rice_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "API is running but the rice model is not loaded.",
                "model_exists": MODEL_PATH.exists(),
                "model_path": str(MODEL_PATH),
                "model_error": MODEL_LOAD_ERROR,
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "input_shape": str(rice_model.input_shape),
        "output_shape": str(rice_model.output_shape),
    }


# =========================================================
# RICE PREDICTION
#
# IMPORTANT:
# The mobile app now sends ONLY:
#
#   file
#
# There is no plant parameter anymore.
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if rice_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Rice disease model is not loaded. "
                f"{MODEL_LOAD_ERROR or ''}"
            ).strip(),
        )

    # Allow normal phone image MIME types.
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file is not an image: {file.content_type}",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        try:
            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file could not be decoded as an image.",
            )

        # =====================================================
        # DO NOT RESIZE OR NORMALIZE HERE.
        #
        # rice_model.keras already contains:
        #   Resizing(256, 256)
        #   Rescaling(1./255)
        # =====================================================
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        print(
            "Prediction request:",
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "shape_before_model": image_array.shape,
            },
        )

        predictions = rice_model.predict(
            image_array,
            verbose=0,
        )

        prediction_values = np.asarray(
            predictions
        )[0]

        if prediction_values.shape[-1] != len(CLASS_LABELS):
            raise RuntimeError(
                f"Expected {len(CLASS_LABELS)} model outputs, "
                f"received {prediction_values.shape[-1]}."
            )

        predicted_class_index = int(
            np.argmax(prediction_values)
        )

        confidence = float(
            prediction_values[predicted_class_index]
        ) * 100.0

        predicted_class = CLASS_LABELS[
            predicted_class_index
        ]

        display_name = DISPLAY_NAMES[
            predicted_class
        ]

        disease_details = DISEASE_INFO[
            predicted_class
        ]

        return {
            "success": True,

            "prediction": {
                "class_index": predicted_class_index,
                "class_name": predicted_class,
                "disease": display_name,
                "confidence": round(confidence, 2),
            },

            "possible_cause": disease_details["cause"],

            "recommendation": disease_details["recommendation"],
        }

    except HTTPException:
        raise

    except Exception as error:
        print("PREDICTION ERROR:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{type(error).__name__}: {error}"
            ),
        )


# =========================================================
# PROFILE PHOTO UPLOAD
#
# Kept because your React Native profile still uses it.
# =========================================================
@app.post("/upload/profile-photo")
async def upload_profile_photo(
    request: Request,
    uid: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed",
        )

    ext = os.path.splitext(
        file.filename or ""
    )[1] or ".jpg"

    filename = (
        f"{uid}_{uuid.uuid4().hex}{ext}"
    )

    file_path = PROFILE_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    image_url = (
        str(request.base_url)
        + f"uploads/profile/{filename}"
    )

    return {
        "message": "Profile photo uploaded successfully",
        "image_url": image_url,
    }


# =========================================================
# SCAN IMAGE UPLOAD
#
# Rice-only version.
# No plant parameter is required anymore.
# =========================================================
@app.post("/upload/scan-image")
async def upload_scan_image(
    request: Request,
    uid: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed",
        )

    ext = os.path.splitext(
        file.filename or ""
    )[1] or ".jpg"

    filename = (
        f"{uid}_rice_{uuid.uuid4().hex}{ext}"
    )

    file_path = SCAN_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    image_url = (
        str(request.base_url)
        + f"uploads/scans/{filename}"
    )

    return {
        "message": "Rice scan image uploaded successfully",
        "image_url": image_url,
    }
