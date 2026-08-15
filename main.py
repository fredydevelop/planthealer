import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL PATHS
# ============================================================
MODEL_PATHS = {
    "Tomato": (
        BASE_DIR
        / "models"
        / "tomato"
        / "tomato_model.keras"
    ),

    "Bell Pepper": (
        BASE_DIR
        / "models"
        / "bell_pepper"
        / "retry_bellpepper_model.keras"
    ),

    "Potato": (
        BASE_DIR
        / "models"
        / "potato"
        / "new_potato_model.keras"
    ),
}


# ============================================================
# CLASS NAMES
#
# IMPORTANT:
# The order of these labels MUST match the class order
# used during training.
# ============================================================
CLASS_NAMES = {

    "Potato": [
        "Potato Early Blight",
        "Potato Late Blight",
        "Potato Healthy"
    ],

    "Bell Pepper": [
        "Bell Pepper Bacterial Spot",
        "Bell Pepper Healthy"
    ],

    "Tomato": [
        "Tomato Bacterial Spot",
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Leaf Mold",
        "Tomato Septoria Leaf Spot",
        "Tomato Spider Mites",
        "Tomato Target Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus",
        "Tomato Healthy"
    ]
}


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_crop_model(model_path):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    return tf.keras.models.load_model(
        model_path
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    # Change this if your models use another input size
    image = image.resize(
        (224, 224)
    )

    # Convert to numpy array
    image_array = np.array(
        image
    ).astype("float32")

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MAKE PREDICTION
# ============================================================
def make_prediction(
    model,
    image_array,
    crop
):

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = int(
        np.argmax(
            prediction,
            axis=1
        )[0]
    )

    confidence = float(
        np.max(
            prediction
        )
    )

    class_names = CLASS_NAMES.get(
        crop,
        []
    )

    if predicted_class < len(
        class_names
    ):
        disease_name = class_names[
            predicted_class
        ]
    else:
        disease_name = (
            f"Class {predicted_class}"
        )

    return (
        predicted_class,
        disease_name,
        confidence
    )


# ============================================================
# SESSION STATE
# ============================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "previous_crop" not in st.session_state:
    st.session_state.previous_crop = None

if "previous_file" not in st.session_state:
    st.session_state.previous_file = None


# ============================================================
# HEADER
# ============================================================
st.title(
    "🌿 Plant Disease Detection System"
)

st.write(
    "Select a crop type, upload an image of the crop leaf, "
    "and use the trained model to predict its condition."
)

st.divider()


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================
left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# PREDICTION ACTIONS
# ============================================================
with left_column:

    st.subheader(
        "Make a Prediction"
    )

    # --------------------------------------------------------
    # SELECT CROP
    # --------------------------------------------------------
    selected_crop = st.selectbox(
        "Select Crop Type",
        [
            "Select crop",
            "Tomato",
            "Potato",
            "Bell Pepper"
        ]
    )


    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Crop Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    # --------------------------------------------------------
    # CURRENT FILE
    # --------------------------------------------------------
    current_file_name = (
        uploaded_file.name
        if uploaded_file is not None
        else None
    )


    # --------------------------------------------------------
    # RESET RESULT IF INPUT CHANGES
    # --------------------------------------------------------
    if (
        selected_crop
        != st.session_state.previous_crop
        or
        current_file_name
        != st.session_state.previous_file
    ):

        st.session_state.prediction_result = None

        st.session_state.previous_crop = (
            selected_crop
        )

        st.session_state.previous_file = (
            current_file_name
        )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------
    predict_button = st.button(
        "Predict Disease",
        type="primary",
        use_container_width=True
    )


    # --------------------------------------------------------
    # HANDLE PREDICTION
    # --------------------------------------------------------
    if predict_button:

        if selected_crop == "Select crop":

            st.warning(
                "Please select a crop type."
            )

        elif uploaded_file is None:

            st.warning(
                "Please upload a crop leaf image."
            )

        else:

            try:

                model_path = MODEL_PATHS[
                    selected_crop
                ]

                with st.spinner(
                    f"Analyzing {selected_crop} leaf..."
                ):

                    # ----------------------------------------
                    # LOAD SELECTED MODEL
                    # ----------------------------------------
                    model = load_crop_model(
                        model_path
                    )


                    # ----------------------------------------
                    # OPEN IMAGE
                    # ----------------------------------------
                    uploaded_file.seek(0)

                    image = Image.open(
                        uploaded_file
                    )


                    # ----------------------------------------
                    # PREPROCESS IMAGE
                    # ----------------------------------------
                    processed_image = (
                        preprocess_image(
                            image
                        )
                    )


                    # ----------------------------------------
                    # MAKE PREDICTION
                    # ----------------------------------------
                    (
                        predicted_class,
                        disease_name,
                        confidence
                    ) = make_prediction(
                        model,
                        processed_image,
                        selected_crop
                    )


                    # ----------------------------------------
                    # STORE RESULT
                    # ----------------------------------------
                    st.session_state.prediction_result = {
                        "crop": selected_crop,
                        "class": predicted_class,
                        "disease": disease_name,
                        "confidence": confidence
                    }


            except FileNotFoundError as error:

                st.error(
                    str(error)
                )

                st.write(
                    "Streamlit is looking for the model at:"
                )

                st.code(
                    str(
                        MODEL_PATHS[
                            selected_crop
                        ]
                    )
                )


            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ============================================================
# RIGHT COLUMN
# IMAGE AND RESULT
# ============================================================
with right_column:

    st.subheader(
        "Image & Result"
    )


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------
    if uploaded_file is not None:

        try:

            uploaded_file.seek(0)

            preview_image = Image.open(
                uploaded_file
            )


            # Smaller image and centered
            image_left, image_center, image_right = st.columns(
                [1, 2, 1]
            )

            with image_center:

                st.image(
                    preview_image,
                    caption=f"{selected_crop} Leaf Image",
                    width=240
                )


        except Exception:

            st.error(
                "Unable to display the uploaded image."
            )

    else:

        st.info(
            "Your uploaded crop image will appear here."
        )


    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------
    result = (
        st.session_state.prediction_result
    )


    if result is not None:

        st.divider()

        st.success(
            "Prediction completed successfully."
        )

        st.subheader(
            "Prediction Result"
        )


        # Main disease result
        st.write(
            f"## {result['disease']}"
        )


        # Crop and confidence
        result_col1, result_col2 = st.columns(
            2
        )


        with result_col1:

            st.metric(
                label="Crop",
                value=result["crop"]
            )


        with result_col2:

            st.metric(
                label="Confidence",
                value=(
                    f"{result['confidence'] * 100:.2f}%"
                )
            )


        # Detailed information
        with st.expander(
            "Prediction Details"
        ):

            st.write(
                f"**Predicted Class:** "
                f"{result['class']}"
            )

            st.write(
                f"**Disease:** "
                f"{result['disease']}"
            )

            st.write(
                f"**Confidence:** "
                f"{result['confidence'] * 100:.2f}%"
            )


    elif uploaded_file is not None:

        st.info(
            "Click **Predict Disease** to analyze the uploaded image."
        )
