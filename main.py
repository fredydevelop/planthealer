from pathlib import Path
import hashlib

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)


# =========================================================
# MODEL PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Tomato": BASE_DIR / "models" / "tomato" / "tomato_model.keras",

    "Bell Pepper": (
        BASE_DIR
        / "models"
        / "bell_pepper"
        / "retry_bellpepper_model.keras"
    ),

    "Potato": BASE_DIR / "models" / "potato" / "new_potato_model.keras",
}


# =========================================================
# CLASS LABELS
#
# IMPORTANT:
# The class order must remain exactly the same as the order
# used during model training.
# =========================================================
CLASS_LABELS = {
    "Tomato": [
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

    "Bell Pepper": [
        "Bacterial_spot",
        "Healthy",
    ],

    "Potato": [
        "Early_blight",
        "Late_blight",
        "Healthy",
    ],
}


# =========================================================
# DISPLAY NAMES
# =========================================================
DISPLAY_NAMES = {
    "Tomato": {
        "Bacterial_spot": "Tomato Bacterial Spot",
        "Early_blight": "Tomato Early Blight",
        "Healthy": "Healthy Tomato Leaf",
        "Late_blight": "Tomato Late Blight",
        "Leaf_Mold": "Tomato Leaf Mold",
        "Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
        "Spider_mites": "Tomato Spider Mites",
        "Target_Spot": "Tomato Target Spot",
        "YellowLeaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "mosaic_virus": "Tomato Mosaic Virus",
    },

    "Bell Pepper": {
        "Bacterial_spot": "Bell Pepper Bacterial Spot",
        "Healthy": "Healthy Bell Pepper Leaf",
    },

    "Potato": {
        "Early_blight": "Potato Early Blight",
        "Late_blight": "Potato Late Blight",
        "Healthy": "Healthy Potato Leaf",
    },
}


# =========================================================
# DISEASE INFORMATION
# =========================================================
DISEASE_INFO = {
    # -----------------------------------------------------
    # TOMATO
    # -----------------------------------------------------
    "Tomato": {
        "Bacterial_spot": {
            "cause": (
                "A bacterial infection commonly associated with "
                "Xanthomonas species. Warm, wet conditions and "
                "splashing water can encourage its spread."
            ),
            "recommendation": (
                "Remove severely affected leaves, avoid overhead "
                "watering, maintain good field sanitation and airflow, "
                "and use appropriate locally approved disease-control "
                "measures when necessary."
            ),
        },

        "Early_blight": {
            "cause": (
                "A fungal disease commonly associated with Alternaria "
                "solani. Warm temperatures, leaf moisture, and infected "
                "plant debris can favour its development."
            ),
            "recommendation": (
                "Remove infected leaves and crop debris, improve airflow "
                "around plants, avoid prolonged leaf wetness, practise "
                "crop rotation, and use an appropriate locally approved "
                "fungicide when necessary."
            ),
        },

        "Healthy": {
            "cause": (
                "No disease symptoms were detected by the model."
            ),
            "recommendation": (
                "Continue good crop management, regular monitoring, "
                "proper watering, balanced nutrition, and field sanitation."
            ),
        },

        "Late_blight": {
            "cause": (
                "A disease caused by Phytophthora infestans. Cool, humid, "
                "and wet conditions can favour rapid development and spread."
            ),
            "recommendation": (
                "Remove and safely dispose of infected plant material, "
                "reduce prolonged leaf wetness, improve airflow, monitor "
                "nearby plants, and apply locally recommended late-blight "
                "control measures where appropriate."
            ),
        },

        "Leaf_Mold": {
            "cause": (
                "A fungal disease commonly caused by Passalora fulva. "
                "It is particularly favoured by high humidity and poor "
                "air circulation."
            ),
            "recommendation": (
                "Improve ventilation and plant spacing, reduce humidity "
                "around foliage, avoid unnecessary leaf wetness, remove "
                "affected leaves, and use appropriate locally approved "
                "control measures when needed."
            ),
        },

        "Septoria_leaf_spot": {
            "cause": (
                "A fungal disease caused by Septoria lycopersici. "
                "Warm conditions, wet foliage, and infected crop residue "
                "can favour infection."
            ),
            "recommendation": (
                "Remove infected lower leaves and crop debris, avoid "
                "overhead irrigation, improve spacing and airflow, "
                "practise crop rotation, and use suitable locally approved "
                "disease-control products when necessary."
            ),
        },

        "Spider_mites": {
            "cause": (
                "Feeding damage from spider mites, commonly the "
                "two-spotted spider mite. Hot and dry conditions can "
                "favour mite populations."
            ),
            "recommendation": (
                "Inspect the undersides of leaves, remove heavily affected "
                "foliage, reduce plant stress, maintain good field hygiene, "
                "and use an appropriate locally approved mite-control "
                "method when infestation is significant."
            ),
        },

        "Target_Spot": {
            "cause": (
                "A fungal disease commonly associated with Corynespora "
                "cassiicola. Warm, humid conditions and prolonged leaf "
                "wetness can support infection."
            ),
            "recommendation": (
                "Remove affected foliage and plant debris, improve airflow, "
                "avoid prolonged leaf wetness, rotate crops where possible, "
                "and use locally approved disease-control measures when needed."
            ),
        },

        "YellowLeaf_Curl_Virus": {
            "cause": (
                "A viral disease commonly spread by whiteflies. "
                "Infected planting material and whitefly activity can "
                "contribute to disease spread."
            ),
            "recommendation": (
                "Remove severely infected plants, control whitefly "
                "populations using appropriate local practices, manage "
                "weeds that may host the virus or vector, and use healthy "
                "planting material."
            ),
        },

        "mosaic_virus": {
            "cause": (
                "A viral infection such as Tomato mosaic virus. "
                "It can spread through infected plant material, "
                "contaminated hands or tools, and mechanical contact."
            ),
            "recommendation": (
                "Remove infected plants, disinfect tools and hands after "
                "handling affected plants, avoid unnecessary plant-to-plant "
                "contact, and use healthy or certified planting material."
            ),
        },
    },

    # -----------------------------------------------------
    # BELL PEPPER
    # -----------------------------------------------------
    "Bell Pepper": {
        "Bacterial_spot": {
            "cause": (
                "A bacterial infection commonly associated with "
                "Xanthomonas species. Warm, wet conditions and water "
                "splash can favour its spread."
            ),
            "recommendation": (
                "Remove badly affected leaves, avoid overhead watering, "
                "improve field sanitation and airflow, use healthy planting "
                "material, and apply locally approved bacterial-disease "
                "management measures when necessary."
            ),
        },

        "Healthy": {
            "cause": (
                "No disease symptoms were detected by the model."
            ),
            "recommendation": (
                "Continue good crop management, regular monitoring, "
                "proper watering, balanced nutrition, and field sanitation."
            ),
        },
    },

    # -----------------------------------------------------
    # POTATO
    # -----------------------------------------------------
    "Potato": {
        "Early_blight": {
            "cause": (
                "A fungal disease commonly associated with Alternaria "
                "solani. Warm conditions, leaf moisture, and infected "
                "crop debris can favour infection."
            ),
            "recommendation": (
                "Remove infected foliage and crop debris, improve airflow, "
                "practise crop rotation, avoid prolonged leaf wetness, and "
                "use a suitable locally approved fungicide when necessary."
            ),
        },

        "Late_blight": {
            "cause": (
                "A disease caused by Phytophthora infestans. Cool, wet, "
                "and humid conditions can allow the disease to develop "
                "and spread quickly."
            ),
            "recommendation": (
                "Remove and safely dispose of infected foliage, avoid "
                "prolonged leaf wetness, improve airflow, monitor nearby "
                "plants closely, and use locally recommended late-blight "
                "control measures when appropriate."
            ),
        },

        "Healthy": {
            "cause": (
                "No disease symptoms were detected by the model."
            ),
            "recommendation": (
                "Continue regular crop monitoring, good sanitation, "
                "proper irrigation, balanced nutrition, and preventive "
                "crop-management practices."
            ),
        },
    },
}


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    loaded_models = {}

    for crop, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file for {crop} was not found at: {model_path}"
            )

        loaded_models[crop] = load_model(model_path)

    return loaded_models


# =========================================================
# PREPROCESS IMAGE
# =========================================================
def preprocess_image(uploaded_file):
    """
    Convert uploaded image to RGB,
    resize to 256 x 256,
    normalize pixel values,
    and add batch dimension.
    """

    image_bytes = uploaded_file.getvalue()

    from io import BytesIO

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Same size used by the prediction backend
    image = image.resize((256, 256))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# =========================================================
# MAKE PREDICTION
# =========================================================
def predict_disease(
    model,
    image_array,
    labels
):
    predictions = model.predict(
        image_array,
        verbose=0
    )

    prediction_values = predictions[0]

    predicted_index = int(
        np.argmax(prediction_values)
    )

    confidence = float(
        np.max(prediction_values)
    ) * 100

    predicted_class = labels[
        predicted_index
    ]

    return predicted_class, confidence


# =========================================================
# CREATE INPUT SIGNATURE
#
# This allows us to know when the crop or uploaded image
# has changed.
# =========================================================
def build_input_signature(
    crop,
    uploaded_file
):
    if uploaded_file is None:
        return (
            crop,
            None
        )

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    return (
        crop,
        uploaded_file.name,
        len(file_bytes),
        file_hash,
    )


# =========================================================
# SESSION STATE
# =========================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_signature" not in st.session_state:
    st.session_state.prediction_signature = None


# =========================================================
# LOAD ALL MODELS
# =========================================================
try:
    models = load_models()

except Exception as error:

    st.error(
        f"Unable to load the disease-detection models: {error}"
    )

    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    /* Main page spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Main heading */
    h1 {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Prediction title */
    h2 {
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }

    /* Improve button appearance */
    div.stButton > button {
        border-radius: 8px;
        min-height: 46px;
        font-weight: 600;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        margin-bottom: 0.6rem;
    }

    /* Uploaded image */
    [data-testid="stImage"] {
        margin-top: 0.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE TITLE
# =========================================================
st.title(
    "🌿 Plant Disease Detection System"
)

st.write(
    "Select a plant and upload a clear image of its leaf "
    "to detect a possible disease."
)


# =========================================================
# TWO-COLUMN LAYOUT
#
# LEFT:
# Plant selector
# Upload button
# Image
# Predict button
#
# RIGHT:
# Prediction result
# =========================================================
left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT COLUMN
# =========================================================
with left_column:

    crop = st.selectbox(
        "Select Plant",
        options=[
            "Potato",
            "Tomato",
            "Bell Pepper",
        ],
        index=0,
        placeholder="Select a plant",
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=False,
    )


    # -----------------------------------------------------
    # Build current input signature
    # -----------------------------------------------------
    current_signature = build_input_signature(
        crop,
        uploaded_file
    )


    # -----------------------------------------------------
    # Clear old result when crop or image changes
    # -----------------------------------------------------
    if (
        st.session_state.prediction_result
        is not None
        and
        st.session_state.prediction_signature
        != current_signature
    ):

        st.session_state.prediction_result = None

        st.session_state.prediction_signature = None


    # -----------------------------------------------------
    # Variables used for prediction
    # -----------------------------------------------------
    image = None
    image_array = None


    # -----------------------------------------------------
    # SHOW IMAGE DIRECTLY UNDER UPLOAD BUTTON
    # -----------------------------------------------------
    if uploaded_file is not None:

        try:

            image, image_array = preprocess_image(
                uploaded_file
            )

            image_left, image_center, image_right = st.columns(
                [1, 1.2, 1]
            )

            with image_center:
                st.image(
                    image,
                    caption="Uploaded Leaf Image",
                    width=180,
                )

        except Exception:

            st.error(
                "The uploaded file could not be read "
                "as a valid image."
            )


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------
    predict_button = st.button(
        "Predict Disease",
        type="primary",
        use_container_width=True,
        disabled=(
            uploaded_file is None
            or image_array is None
        ),
    )


    # -----------------------------------------------------
    # PERFORM PREDICTION
    # -----------------------------------------------------
    if (
        predict_button
        and image_array is not None
    ):

        try:

            predicted_class, confidence = (
                predict_disease(
                    model=models[crop],
                    image_array=image_array,
                    labels=CLASS_LABELS[crop],
                )
            )

            display_name = (
                DISPLAY_NAMES[crop][
                    predicted_class
                ]
            )

            disease_details = (
                DISEASE_INFO[crop][
                    predicted_class
                ]
            )


            # Save result
            st.session_state.prediction_result = {
                "crop": crop,
                "class": predicted_class,
                "display_name": display_name,
                "confidence": confidence,
                "cause": disease_details[
                    "cause"
                ],
                "recommendation": (
                    disease_details[
                        "recommendation"
                    ]
                ),
            }


            # Save the crop/image combination
            # associated with this result
            st.session_state.prediction_signature = (
                current_signature
            )


        except Exception as error:

            st.error(
                "Prediction could not be completed. "
                f"{error}"
            )


# =========================================================
# RIGHT COLUMN
# =========================================================
with right_column:

    result = (
        st.session_state.prediction_result
    )


    # Only display the result if it belongs
    # to the currently selected crop and image
    if (
        result is not None
        and
        st.session_state.prediction_signature
        == current_signature
    ):

        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------
        st.success(
            "Prediction completed successfully."
        )


        # -------------------------------------------------
        # RESULT HEADING
        # -------------------------------------------------
        st.markdown(
            "### Prediction Result"
        )


        # -------------------------------------------------
        # PREDICTED DISEASE TITLE
        # Example:
        # Potato Late Blight
        # -------------------------------------------------
        st.markdown(
            f"## {result['display_name']}"
        )


        # -------------------------------------------------
        # DISEASE
        # -------------------------------------------------
        st.markdown(
            f"**Disease:** "
            f"{result['display_name']}"
        )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------
        st.markdown(
            f"**Confidence:** "
            f"{result['confidence']:.2f}%"
        )


        # -------------------------------------------------
        # POSSIBLE CAUSE
        # -------------------------------------------------
        st.markdown(
            "**Possible Cause:**"
        )

        st.write(
            result["cause"]
        )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------
        st.markdown(
            "**Recommendation:**"
        )

        st.write(
            result["recommendation"]
        )
