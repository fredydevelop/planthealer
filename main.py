import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# Replace these paths with the SAME paths you use
# in your FastAPI model-loading code.
# ============================================================
MODEL_PATHS = {
    "Tomato": "models/tomato_model.keras",
    "Potato": "models/potato_model.keras",
    "Bell Pepper": "models/bell_pepper_model.keras"
}


# ============================================================
# CLASS LABELS
# IMPORTANT:
# Replace these with the EXACT class order used during training.
#
# If you do not provide the class labels, prediction can still
# work, but the result will only show the class index.
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

    # Replace with your exact Tomato class order
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
# Cached so Streamlit does not reload the model every time
# the user interacts with the page.
# ============================================================
@st.cache_resource
def load_crop_model(model_path):
    return tf.keras.models.load_model(model_path)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # IMPORTANT:
    # Change this if your models were trained using another size.
    image = image.resize((224, 224))

    # Convert image to numpy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def make_prediction(model, image_array, crop):

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = int(
        np.argmax(prediction, axis=1)[0]
    )

    confidence = float(
        np.max(prediction)
    )

    class_names = CLASS_NAMES.get(crop, [])

    if predicted_class < len(class_names):
        disease_name = class_names[predicted_class]
    else:
        disease_name = f"Class {predicted_class}"

    return predicted_class, disease_name, confidence


# ============================================================
# SESSION STATE
# Used so prediction disappears when the user changes
# crop or uploaded image.
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
st.title("🌿 Plant Disease Detection System")

st.write(
    "Upload an image of a crop leaf and use the trained "
    "deep-learning model to identify its condition."
)

st.divider()


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================
left_column, right_column = st.columns(
    [1, 1.15],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# Prediction controls
# ============================================================
with left_column:

    st.subheader("Make a Prediction")

    # --------------------------------------------------------
    # CROP SELECTION
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
    # IMAGE UPLOAD
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
    # RESET RESULT WHEN INPUT CHANGES
    # --------------------------------------------------------
    current_file_name = (
        uploaded_file.name
        if uploaded_file is not None
        else None
    )

    if (
        selected_crop != st.session_state.previous_crop
        or
        current_file_name != st.session_state.previous_file
    ):

        st.session_state.prediction_result = None

        st.session_state.previous_crop = selected_crop

        st.session_state.previous_file = current_file_name


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------
    predict_button = st.button(
        "Predict Disease",
        type="primary",
        use_container_width=True
    )


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

                with st.spinner(
                    f"Analyzing {selected_crop} leaf..."
                ):

                    # ----------------------------------------
                    # Get correct model
                    # ----------------------------------------
                    model_path = MODEL_PATHS[
                        selected_crop
                    ]

                    model = load_crop_model(
                        model_path
                    )


                    # ----------------------------------------
                    # Open image
                    # ----------------------------------------
                    image = Image.open(
                        uploaded_file
                    )


                    # ----------------------------------------
                    # Preprocess
                    # ----------------------------------------
                    processed_image = (
                        preprocess_image(image)
                    )


                    # ----------------------------------------
                    # Predict
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
                    # Save result
                    # ----------------------------------------
                    st.session_state.prediction_result = {
                        "crop": selected_crop,
                        "class": predicted_class,
                        "disease": disease_name,
                        "confidence": confidence
                    }


            except FileNotFoundError:

                st.error(
                    f"The model for {selected_crop} "
                    f"could not be found.\n\n"
                    f"Model path: "
                    f"{MODEL_PATHS[selected_crop]}"
                )


            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ============================================================
# RIGHT COLUMN
# Image + prediction result
# ============================================================
with right_column:

    st.subheader("Image & Result")


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------
    if uploaded_file is not None:

        try:

            uploaded_file.seek(0)

            preview_image = Image.open(
                uploaded_file
            )

            st.image(
                preview_image,
                caption=f"{selected_crop} Leaf Image",
                use_container_width=True
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
    result = st.session_state.prediction_result

    if result is not None:

        st.divider()

        st.success(
            "Prediction completed successfully."
        )

        st.subheader(
            "Prediction Result"
        )


        st.write(
            f"### {result['disease']}"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Crop",
                result["crop"]
            )


        with result_col2:

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )


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
            "Click **Predict Disease** to analyze "
            "the uploaded image."
        )
