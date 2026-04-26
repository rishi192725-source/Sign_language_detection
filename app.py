import streamlit as st
from PIL import Image
import pandas as pd
from utils.predict import predict_image

st.set_page_config(page_title="Sign Language Detection", layout="wide")

st.title("Sign Language Detection")
st.write("Upload an image and the model will detect the sign language letter.")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        annotated_image, predictions = predict_image(image, conf_threshold)

        st.subheader("Prediction")
        st.image(annotated_image, use_container_width=True)

    st.subheader("Detected Results")

    if predictions:
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True)

        top_pred = max(predictions, key=lambda x: x["confidence"])
        st.success(
            f"Top Prediction: {top_pred['class_name']} "
            f"(Confidence: {top_pred['confidence']:.2f})"
        )
    else:
        st.warning("No sign detected.")
else:
    st.info("Please upload an image to start prediction.")