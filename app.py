import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from io import BytesIO
import hashlib


# Download trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="PepsTechRnD/egg-quality-yolo11x",
    filename="best.pt"
)

model = YOLO(model_path)

BRIDGE_URL = "https://egg-quality-detection.onrender.com"

st.title("🥚 Egg Quality Detection")
st.write("Raspberry Pi Camera / Manual Upload → YOLO11x Detection")


source = st.radio(
    "Choose image source:",
    [
        "📷 Raspberry Pi Camera",
        "📁 Upload Image"
    ]
)


# =========================================================
# RASPBERRY PI CAMERA
# =========================================================

if source == "📷 Raspberry Pi Camera":

    @st.fragment(run_every="2s")
    def camera_dashboard():

        response = requests.get(
            f"{BRIDGE_URL}/latest",
            timeout=10
        )

        if response.status_code == 200:

            try:
                image_bytes = response.content

                # Check whether a valid image exists
                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                image_hash = hashlib.md5(
                    image_bytes
                ).hexdigest()

                # Store latest image
                if st.session_state.get("image_hash") != image_hash:

                    st.session_state["egg_image"] = image
                    st.session_state["image_hash"] = image_hash
                    st.session_state["detection_result"] = None

                # Display latest Pi image
                if "egg_image" in st.session_state:

                    st.image(
                        st.session_state["egg_image"],
                        caption="Live Image from Raspberry Pi",
                        use_container_width=True
                    )

                    if st.button("🔍 Detect Egg"):

                        image_array = np.array(
                            st.session_state["egg_image"]
                        )

                        results = model(image_array)

                        st.session_state["detection_result"] = (
                            results[0].plot()
                        )

                    # Show detection result
                    if st.session_state.get(
                        "detection_result"
                    ) is not None:

                        st.image(
                            st.session_state["detection_result"],
                            caption="Detection Result",
                            use_container_width=True
                        )

            except Exception:
                st.info(
                    "Waiting for an image from Raspberry Pi..."
                )

        else:
            st.info(
                "Waiting for an image from Raspberry Pi..."
            )


    camera_dashboard()


# =========================================================
# MANUAL UPLOAD
# =========================================================

else:

    uploaded_file = st.file_uploader(
        "Upload an egg image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Egg Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Egg"):

            image_array = np.array(image)

            results = model(image_array)

            result_image = results[0].plot()

            st.image(
                result_image,
                caption="Detection Result",
                use_container_width=True
            )
