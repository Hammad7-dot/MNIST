import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Deep Learning MNIST Digit Recognizer")


# Cache the model so it only loads once into memory on startup
@st.cache_resource
def load_model():
    # FIXED: Added compile=False to bypass incompatible metadata during loading
    return tf.keras.models.load_model("mnist_model.h5", compile=False)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'mnist_model.h5'. Verify it is in the root folder: {e}")


def preprocess_image(img) -> np.ndarray:
    """Preprocess image to match the input layout of the trained ANN."""
    # Handle alpha channels (transparency) by pasting onto a solid white canvas
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        clean_bg = Image.new("RGBA", img.size, (255, 255, 255))
        clean_bg.paste(img, (0, 0), img.convert("RGBA"))
        img = clean_bg

    # Convert to 8-bit Grayscale and downsample to 28x28 pixels
    img = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)

    # Invert the background if it is white/light (MNIST expects white on black)
    corners = [
        img_array[0, 0],
        img_array[0, 27],
        img_array[27, 0],
        img_array[27, 27],
    ]
    if np.mean(corners) > 127:
        img_array = 255 - img_array

    # Normalize pixel array to [0.0, 1.0] and flatten to a 1D vector (1, 784)
    final_tensor = img_array.astype("float32") / 255.0
    return final_tensor.reshape(1, 784)


uploaded_file = st.file_uploader(
    "Upload a handwritten digit image...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=180)

    if st.button("Analyze Pattern", type="primary"):
        with st.spinner("Analyzing..."):
            processed_data = preprocess_image(image)

            # Generate prediction metrics from neural network
            predictions = model.predict(processed_data)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            # Markdown header formatting applied correctly
            st.success(f"### Predicted Value: {predicted_class}")
            st.metric(
                label="Classification Confidence", value=f"{confidence:.2%}"
            )

            # Chart breakdown distribution
            chart_data = {
                str(i): float(prob) for i, prob in enumerate(predictions[0])
            }
            st.bar_chart(chart_data)