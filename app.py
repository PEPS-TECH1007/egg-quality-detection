import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from io import BytesIO


# Download trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="PepsTechRnD/egg-quality-yolo11x",
    filename="best.pt"
)

# Load trained YOLO11x model
model = YOLO(model_path)


# Render bridge
BRIDGE_URL = "https://egg-quality-detection.onrender.com"


st.title("🥚 Egg Quality Detection")
st.write("Raspberry Pi Camera / Manual Upload → YOLO11x Detection")


# -----------------------------
# IMAGE SOURCE
# -----------------------------

source = st.radio(
    "Choose image source:",
    [
        "📷 Raspberry Pi Camera",
        "📁 Upload Image"
    ]
)


# -----------------------------
# RASPBERRY PI IMAGE
# -----------------------------

if source == "📷 Raspberry Pi Camera":

    if st.button("📷 Get Latest Egg Image"):

        response = requests.get(
            f"{BRIDGE_URL}/latest"
        )

        if response.status_code == 200:

            try:
                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                st.session_state["egg_image"] = image

            except Exception:
                st.error("No valid image is available from the Raspberry Pi.")

        else:
            st.error("Could not get image from Raspberry Pi.")


# -----------------------------
# MANUAL IMAGE UPLOAD
# -----------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload an egg image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.session_state["egg_image"] = image


# -----------------------------
# DISPLAY IMAGE
# -----------------------------

if "egg_image" in st.session_state:

    image = st.session_state["egg_image"]

    st.image(
        image,
        caption="Egg Image",
        use_container_width=True
    )


    # -----------------------------
    # DETECTION
    # -----------------------------

    if st.button("🔍 Detect Egg"):

        image_array = np.array(image)

        results = model(image_array)

        result_image = results[0].plot()

        st.image(
            result_image,
            caption="Detection Result",
            use_container_width=True
        )
