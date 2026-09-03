from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from pathlib import Path


app = FastAPI(
    title="Rice Crop Disease Detection API",
    description="API for detecting rice leaf diseases using a trained CNN model",
    version="1.0.0"
)


# =========================================================
# MODEL PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "rice_model.keras"


# =========================================================
# CLASS LABELS
#
# IMPORTANT:
# Keep this order exactly the same as the class order
# produced during model training.
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
try:
    rice_model = load_model(MODEL_PATH)

except Exception as error:
    print(f"Error loading model: {error}")
    rice_model = None


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
def home():
    return {
        "message": "Rice Crop Disease Detection API is running",
        "model_loaded": rice_model is not None
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------
    if rice_model is None:
        raise HTTPException(
            status_code=500,
            detail="Rice disease model is not loaded"
        )


    # -----------------------------------------------------
    # Validate file type
    # -----------------------------------------------------
    if file.content_type not in [
        "image/jpeg",
        "image/jpg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported"
        )


    try:

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


        # -------------------------------------------------
        # PREPROCESS IMAGE
        #
        # Your trained rice CNN already contains:
        #   Resizing(256, 256)
        #   Rescaling(1./255)
        #
        # Therefore we should NOT resize or divide by 255
        # here again.
        # -------------------------------------------------
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # -------------------------------------------------
        # Predict
        # -------------------------------------------------
        predictions = rice_model.predict(
            image_array,
            verbose=0
        )

        prediction_values = predictions[0]


        # -------------------------------------------------
        # Get highest prediction
        # -------------------------------------------------
        predicted_index = int(
            np.argmax(prediction_values)
        )

        confidence = float(
            np.max(prediction_values)
        ) * 100


        # -------------------------------------------------
        # Get disease class
        # -------------------------------------------------
        predicted_class = CLASS_LABELS[
            predicted_index
        ]

        display_name = DISPLAY_NAMES[
            predicted_class
        ]

        disease_details = DISEASE_INFO[
            predicted_class
        ]


        # -------------------------------------------------
        # Return result to mobile app
        # -------------------------------------------------
        return {
            "success": True,

            "prediction": {
                "class_index": predicted_index,
                "class_name": predicted_class,
                "disease": display_name,
                "confidence": round(
                    confidence,
                    2
                )
            },

            "possible_cause": disease_details[
                "cause"
            ],

            "recommendation": disease_details[
                "recommendation"
            ]
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
