from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import numpy as np
import io
import traceback


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Rice Crop Disease Detection API",
    description="API for detecting and classifying rice leaf diseases using a trained CNN model.",
    version="1.1.0",
)


# =========================================================
# MODEL PATH
#
# GitHub / Render project structure:
#
# project-root/
# ├── predictions.py
# ├── models/
# │   └── rice_model.keras
# └── requirements.txt
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rice_model.keras"


# =========================================================
# CLASS LABELS
#
# IMPORTANT:
# Keep this order exactly the same as the class order used
# during model training.
# =========================================================
CLASS_LABELS = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]


# =========================================================
# DISPLAY NAMES
# =========================================================
DISPLAY_NAMES = {
    "Bacterial Leaf Blight": "Rice Bacterial Leaf Blight",
    "Brown Spot": "Rice Brown Spot",
    "Healthy Rice Leaf": "Healthy Rice Leaf",
    "Leaf Blast": "Rice Leaf Blast",
    "Leaf scald": "Rice Leaf Scald",
    "Sheath Blight": "Rice Sheath Blight",
}


# =========================================================
# DISEASE INFORMATION
# =========================================================
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
            "No visible rice leaf disease symptoms were detected "
            "by the model."
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

print("=================================================")
print("Rice Disease API Starting")
print("BASE DIRECTORY:", BASE_DIR)
print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", MODEL_PATH.exists())
print("=================================================")

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    # compile=False is appropriate because the server only performs
    # inference. It also avoids loading optimizer/training state.
    rice_model = load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Rice model loaded successfully.")
    print("Model input shape:", rice_model.input_shape)
    print("Model output shape:", rice_model.output_shape)

    # Confirm that the model returns six class probabilities.
    output_shape = rice_model.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "Expected a single model output, but received multiple outputs."
        )

    if output_shape[-1] != len(CLASS_LABELS):
        raise ValueError(
            f"Model output has {output_shape[-1]} classes, "
            f"but CLASS_LABELS contains {len(CLASS_LABELS)} classes."
        )

except Exception as error:
    MODEL_LOAD_ERROR = f"{type(error).__name__}: {error}"

    print("MODEL LOAD ERROR:", MODEL_LOAD_ERROR)
    traceback.print_exc()

    rice_model = None


# =========================================================
# HOME / SERVER STATUS
# =========================================================
@app.get("/")
def home():
    return {
        "message": "Rice Crop Disease Detection API is running",
        "model_loaded": rice_model is not None,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_error": MODEL_LOAD_ERROR,
        "number_of_classes": len(CLASS_LABELS),
    }


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health():
    if rice_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "The API is running but the rice model is not loaded.",
                "model_exists": MODEL_PATH.exists(),
                "model_path": str(MODEL_PATH),
                "model_error": MODEL_LOAD_ERROR,
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "input_shape": str(rice_model.input_shape),
        "output_shape": str(rice_model.output_shape),
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # -----------------------------------------------------
    # Make sure model loaded correctly
    # -----------------------------------------------------
    if rice_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Rice disease model is not loaded. "
                f"{MODEL_LOAD_ERROR or ''}"
            ).strip(),
        )

    # -----------------------------------------------------
    # Validate MIME type
    #
    # Android image pickers can return image/jpeg,
    # image/png, image/webp, etc.
    # -----------------------------------------------------
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not an image. "
                f"Received content type: {file.content_type}"
            ),
        )

    try:
        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image file is empty.",
            )

        # -------------------------------------------------
        # Decode image
        # -------------------------------------------------
        try:
            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded file could not be decoded as an image. "
                    "Please upload a valid rice leaf image."
                ),
            )

        # -------------------------------------------------
        # PREPROCESSING
        #
        # Your trained rice model already contains:
        #
        # Resizing(256, 256)
        # Rescaling(1./255)
        #
        # Therefore:
        # DO NOT resize here.
        # DO NOT divide pixels by 255 here.
        # -------------------------------------------------
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
                "input_shape": image_array.shape,
            },
        )

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------
        predictions = rice_model.predict(
            image_array,
            verbose=0,
        )

        prediction_values = np.asarray(
            predictions
        )[0]

        # -------------------------------------------------
        # Ensure model output is correct
        # -------------------------------------------------
        if prediction_values.shape[-1] != len(CLASS_LABELS):
            raise RuntimeError(
                "Model output/class mismatch. "
                f"Model returned {prediction_values.shape[-1]} values, "
                f"but the API has {len(CLASS_LABELS)} class labels."
            )

        # -------------------------------------------------
        # Get predicted class
        # -------------------------------------------------
        predicted_index = int(
            np.argmax(prediction_values)
        )

        confidence = float(
            prediction_values[predicted_index]
        ) * 100.0

        predicted_class = CLASS_LABELS[
            predicted_index
        ]

        display_name = DISPLAY_NAMES[
            predicted_class
        ]

        disease_details = DISEASE_INFO[
            predicted_class
        ]

        print(
            "Prediction completed:",
            {
                "class_index": predicted_index,
                "class_name": predicted_class,
                "confidence": confidence,
            },
        )

        # -------------------------------------------------
        # RESPONSE TO MOBILE APP
        # -------------------------------------------------
        return {
            "success": True,

            "prediction": {
                "class_index": predicted_index,
                "class_name": predicted_class,
                "disease": display_name,
                "confidence": round(
                    confidence,
                    2,
                ),
            },

            "possible_cause": disease_details[
                "cause"
            ],

            "recommendation": disease_details[
                "recommendation"
            ],
        }

    except HTTPException:
        raise

    except Exception as error:
        print("PREDICTION ERROR:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{type(error).__name__}: {error}"
            ),
        )


# =========================================================
# OPTIONAL LOCAL RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "predictions:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
