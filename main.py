import streamlit as st
import requests
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# API CONFIGURATION
# =========================================================
# Change this if your FastAPI is deployed online.
# Example:
# API_URL = "https://your-api.onrender.com/predict"

API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# TITLE
# =========================================================
st.title("🌿 Plant Disease Detection System")

st.write(
    "Select a crop type, upload an image of the crop leaf, "
    "and use the trained machine learning model to predict its condition."
)

st.divider()


# =========================================================
# CROP SELECTION
# =========================================================
st.subheader("1. Select Crop Type")

crop_options = {
    "Tomato": "tomato",
    "Potato": "potato",
    "Bell Pepper": "bell_pepper"
}

selected_crop = st.selectbox(
    "Crop Type",
    options=["Select crop"] + list(crop_options.keys())
)


# =========================================================
# IMAGE UPLOAD
# =========================================================
if selected_crop != "Select crop":

    st.subheader("2. Upload Crop Image")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Preview image
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption=f"Uploaded {selected_crop} leaf",
            use_container_width=True
        )

        st.subheader("3. Make Prediction")

        if st.button(
            "Predict Disease",
            type="primary",
            use_container_width=True
        ):

            try:
                # Reset file pointer before sending
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                data = {
                    "plant": crop_options[selected_crop]
                }

                with st.spinner("Analyzing crop image..."):

                    response = requests.post(
                        API_URL,
                        data=data,
                        files=files,
                        timeout=60
                    )

                # =================================================
                # SUCCESSFUL RESPONSE
                # =================================================
                if response.status_code == 200:

                    result = response.json()

                    plant = result.get("plant")
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")

                    st.success("Prediction completed successfully.")

                    st.subheader("Prediction Result")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="Predicted Class",
                            value=str(prediction)
                        )

                    with col2:
                        if confidence is not None:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence * 100:.2f}%"
                            )
                        else:
                            st.metric(
                                label="Confidence",
                                value="N/A"
                            )

                    st.write(f"**Crop:** {selected_crop}")

                # =================================================
                # API ERROR
                # =================================================
                else:

                    try:
                        error_message = response.json().get(
                            "detail",
                            "Prediction failed."
                        )
                    except Exception:
                        error_message = response.text

                    st.error(
                        f"API Error ({response.status_code}): "
                        f"{error_message}"
                    )

            # =====================================================
            # CONNECTION ERROR
            # =====================================================
            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the prediction API. "
                    "Make sure the FastAPI server is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The prediction request took too long. "
                    "Please try again."
                )

            except Exception as e:

                st.error(f"An unexpected error occurred: {str(e)}")

    else:
        st.info("Upload a crop leaf image to continue.")

else:
    st.info("Select a crop type to begin.")
