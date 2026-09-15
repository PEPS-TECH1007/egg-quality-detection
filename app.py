import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download


# Download trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="PepsTechRnD/egg-quality-yolo11x",
    filename="best.pt"
)

# Load trained YOLO11x model
model = YOLO(model_path)


# Page title
st.title("🥚 Egg Quality Detection")

st.write("Upload an egg image to test the trained YOLO11x model.")


# Upload image
uploaded_file = st.file_uploader(
    "Upload an egg image",
    type=["jpg", "jpeg", "png"]
)


# If image is uploaded
if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="Uploaded Image")

    # Detection button
    if st.button("🔍 Detect Egg"):

        # Convert PIL image to NumPy array
        image_array = np.array(image)

        # Run YOLO
        results = model(image_array)

        # Draw detections
        result_image = results[0].plot()

        # Display result
        st.image(
            result_image,
            caption="Detection Result"
        )
