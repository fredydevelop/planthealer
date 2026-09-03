from pathlib import Path
import hashlib

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Rice Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "rice_model.keras",
    BASE_DIR / "rice_model.keras",
]

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
            "Xanthomonas oryzae pv. oryzae. Warm, humid conditions, "
            "rain, irrigation water, and infected plant material can "
            "support disease development and spread."
        ),
        "recommendation": (
            "Use clean and healthy planting materials, avoid excessive "
            "nitrogen application, maintain good field sanitation, and "
            "use resistant rice varieties where available. Seek guidance "
            "from a local agricultural extension officer when the disease "
            "is severe."
        ),
    },
    "Brown Spot": {
        "cause": (
            "A fungal disease commonly associated with Bipolaris oryzae. "
            "Plant stress, poor soil fertility, infected seed, and warm, "
            "humid conditions can favour its development."
        ),
        "recommendation": (
            "Improve crop nutrition and field management, use healthy "
            "seed, remove infected crop residues, reduce plant stress, "
            "and use an appropriate locally approved disease-control "
            "measure when necessary."
        ),
    },
    "Healthy Rice Leaf": {
        "cause": "No visible rice leaf disease symptoms were detected by the model.",
        "recommendation": (
            "Continue regular crop monitoring, balanced fertilization, "
            "proper irrigation, field sanitation, and other good rice "
            "production practices."
        ),
    },
    "Leaf Blast": {
        "cause": (
            "A fungal disease commonly caused by Magnaporthe oryzae. "
            "High humidity, prolonged leaf wetness, susceptible varieties, "
            "and excessive nitrogen can favour disease development."
        ),
        "recommendation": (
            "Manage infected crop residues, avoid excessive nitrogen "
            "fertilization, maintain appropriate water management, use "
            "resistant varieties where available, and apply a locally "
            "approved control measure when recommended."
        ),
    },
    "Leaf scald": {
        "cause": (
            "A fungal rice disease commonly associated with Microdochium "
            "oryzae. Humid conditions, infected crop residues, and stressed "
            "plants may encourage disease development."
        ),
        "recommendation": (
            "Use healthy planting material, manage infected residues, "
            "maintain balanced crop nutrition, improve field sanitation, "
            "and use locally recommended disease-control measures when necessary."
        ),
    },
    "Sheath Blight": {
        "cause": (
            "A fungal disease commonly caused by Rhizoctonia solani. "
            "Dense crop stands, high humidity, warm temperatures, and "
            "excessive nitrogen can favour its development."
        ),
        "recommendation": (
            "Avoid excessive nitrogen application, maintain suitable plant "
            "spacing, manage infected crop residues, and use an appropriate "
            "locally approved control measure when necessary."
        ),
    },
}

def get_model_path():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path
    checked = "\n".join(str(p) for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "The rice disease model was not found. Checked:\n" + checked
    )

@st.cache_resource
def load_rice_model():
    return load_model(get_model_path())

def preprocess_image(uploaded_file):
    from io import BytesIO

    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # The saved rice model already contains Resizing(256,256)
    # and Rescaling(1./255), so do not normalize again here.
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array

def predict_disease(model, image_array):
    predictions = model.predict(image_array, verbose=0)
    values = predictions[0]

    predicted_index = int(np.argmax(values))
    confidence = float(np.max(values)) * 100
    predicted_class = CLASS_LABELS[predicted_index]

    return predicted_class, confidence

def build_input_signature(uploaded_file):
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    return (
        uploaded_file.name,
        len(file_bytes),
        file_hash,
    )

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_signature" not in st.session_state:
    st.session_state.prediction_signature = None

try:
    rice_model = load_rice_model()
except Exception as error:
    st.error(f"Unable to load the rice disease-detection model: {error}")
    st.stop()

st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1 {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    h2 {
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }

    div.stButton > button {
        border-radius: 8px;
        min-height: 46px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        margin-bottom: 0.6rem;
    }

    [data-testid="stImage"] {
        margin-top: 0.2rem;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

st.title("🌾 Rice Crop Disease Detection System")

st.write(
    "Upload a clear image of a rice leaf to detect its possible disease condition."
)

left_column, right_column = st.columns(
    [1, 1],
    gap="large",
)

with left_column:
    st.markdown("### Rice Leaf Image")

    uploaded_file = st.file_uploader(
        "Upload Rice Leaf Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    current_signature = build_input_signature(uploaded_file)

    if (
        st.session_state.prediction_result is not None
        and st.session_state.prediction_signature != current_signature
    ):
        st.session_state.prediction_result = None
        st.session_state.prediction_signature = None

    image = None
    image_array = None

    if uploaded_file is not None:
        try:
            image, image_array = preprocess_image(uploaded_file)

            image_left, image_center, image_right = st.columns([1, 1.2, 1])

            with image_center:
                st.image(
                    image,
                    caption="Uploaded Rice Leaf Image",
                    width=180,
                )

        except Exception:
            st.error(
                "The uploaded file could not be read as a valid image."
            )

    predict_button = st.button(
        "Predict Rice Disease",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or image_array is None),
    )

    if predict_button and image_array is not None:
        try:
            predicted_class, confidence = predict_disease(
                model=rice_model,
                image_array=image_array,
            )

            display_name = DISPLAY_NAMES[predicted_class]
            disease_details = DISEASE_INFO[predicted_class]

            st.session_state.prediction_result = {
                "class": predicted_class,
                "display_name": display_name,
                "confidence": confidence,
                "cause": disease_details["cause"],
                "recommendation": disease_details["recommendation"],
            }

            st.session_state.prediction_signature = current_signature

        except Exception as error:
            st.error(
                "Prediction could not be completed. "
                f"{error}"
            )

with right_column:
    result = st.session_state.prediction_result

    if (
        result is not None
        and st.session_state.prediction_signature == current_signature
    ):
        st.success("Prediction completed successfully.")

        st.markdown("### Prediction Result")

        st.markdown(f"## {result['display_name']}")

        st.markdown(
            f"**Disease:** {result['display_name']}"
        )

        st.markdown(
            f"**Confidence:** {result['confidence']:.2f}%"
        )

        st.markdown("**Possible Cause:**")
        st.write(result["cause"])

        st.markdown("**Recommendation:**")
        st.write(result["recommendation"])
