import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = '671010158_mnist_model.keras'

st.set_page_config(page_title="MNIST Digit Predictor", page_icon=None)

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit (0-9) to get a prediction.")


@st.cache_resource
def load_mnist_model(path):
    return tf.keras.models.load_model(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the model is saved correctly.")
    st.stop()

model = load_mnist_model(MODEL_PATH)

st.sidebar.header("Preprocess options")
invert = st.sidebar.checkbox(
    "Invert colors (use when the digit is dark on a white background)",
    value=False,
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 1. Load the image
        img = Image.open(uploaded_file)

        # 2. Convert to grayscale (MNIST images are grayscale)
        img = img.convert('L')

        # 3. MNIST digits are white strokes on a black background
        if invert:
            img = ImageOps.invert(img)

        # 4. Resize to 28x28 pixels, as required by the model
        img = img.resize((28, 28))

        # 5. Convert to a NumPy array and normalize [0, 255] -> [0, 1]
        img_array = np.array(img).astype("float32") / 255.0

        # 6. Reshape for model prediction (add batch dimension)
        img_array = img_array.reshape(1, 28, 28)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        with col2:
            st.image(img, caption='Preprocessed 28x28 (model input)', use_container_width=True)

        st.write("Classifying...")

        # 7. Make a prediction
        prediction = model.predict(img_array)

        # 8. Get the predicted digit and its confidence
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.success(f"The model predicts the digit is: **{predicted_digit}**  (confidence {confidence:.2%})")

        st.subheader("Probability for each digit")
        st.bar_chart({"probability": prediction[0]})

    except Exception as e:
        st.error(
            f"An error occurred during prediction: {e}. "
            "Please ensure the uploaded image is valid and the model is correctly loaded."
        )
else:
    st.info("Tip: a clear, well-cropped digit works best. If your digit is black on white, tick 'Invert colors' in the sidebar.")
