import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


# -------- LOAD MODEL --------
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


# -------- IMAGE PREPROCESSING --------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# -------- IMAGE CLASSIFICATION --------
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded = decode_predictions(predictions, top=3)[0]
        return decoded

    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


# -------- STREAMLIT APP --------
def main():
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("AI Image Classifier")
    st.write("Upload your image and let AI detect what it is!")

    # Cache model so it doesn't reload every time
    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        btn = st.button("Classify Image")

        if btn:

            with st.spinner("Analyzing Image..."):

                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")

                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")


# -------- RUN APP --------
if __name__ == "__main__":
    main()