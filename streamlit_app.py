import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Deep Learning MNIST Digit Recognizer")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mnist_model.keras",
        compile=False
    )


try:
    model = load_model()
    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a digit image (28x28 recommended)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((28, 28))

    img_array = np.array(image)

    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_digit}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.bar_chart(prediction[0])