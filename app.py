import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from io import BytesIO
import streamlit.components.v1 as components


BRIDGE_URL = "https://egg-quality-detection.onrender.com"


# ----------------------------------------
# LOAD MODEL ONLY ONCE
# ----------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="PepsTechRnD/egg-quality-yolo11x",
        filename="best.pt"
    )

    return YOLO(model_path)


model = load_model()


# ----------------------------------------
# DASHBOARD
# ----------------------------------------

st.title("🥚 Egg Quality Detection")

st.write(
    "Raspberry Pi Camera / Manual Upload → YOLO11x Detection"
)


source = st.radio(
    "Choose image source:",
    [
        "📷 Raspberry Pi Camera",
        "📁 Upload Image"
    ]
)


# ========================================
# RASPBERRY PI MODE
# ========================================

if source == "📷 Raspberry Pi Camera":

    st.subheader("📷 Raspberry Pi Camera")

    st.write(
        "Waiting for the latest image from Raspberry Pi..."
    )


    # Browser-only automatic image refresh.
    # This does NOT rerun Streamlit or YOLO.
    components.html(
        f"""
        <div style="text-align:center;">
            <img
                id="eggImage"
                src="{BRIDGE_URL}/latest?t=0"
                style="
                    max-width:100%;
                    max-height:500px;
                    border-radius:10px;
                "
            >
        </div>

        <script>

        const image = document.getElementById("eggImage");

        setInterval(function() {{

            image.src =
                "{BRIDGE_URL}/latest?t="
                + new Date().getTime();

        }}, 2000);

        </script>
        """,
        height=520
    )


    # ------------------------------------
    # DETECT LATEST PI IMAGE
    # ------------------------------------

    if st.button("🔍 Detect Egg"):

        response = requests.get(
            f"{BRIDGE_URL}/latest",
            timeout=15
        )

        if response.status_code == 200:

            try:

                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                st.image(
                    image,
                    caption="Image from Raspberry Pi",
                    use_container_width=True
                )


                image_array = np.array(image)

                results = model(image_array)

                result_image = results[0].plot()

                st.image(
                    result_image,
                    caption="Detection Result",
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "The latest response is not a valid image."
                )

        else:

            st.warning(
                "No image has been uploaded by the Raspberry Pi yet."
            )


# ========================================
# MANUAL UPLOAD MODE
# ========================================

else:

    st.subheader("📁 Manual Image Upload")

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
