import json
import h5py
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Deep Learning MNIST Digit Recognizer")


@st.cache_resource
def load_model_safely():
    model_path = "mnist_model.h5"

    # 1. Attempt a standard load with compile disabled
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.warning(
            "Version structural mismatch detected. Patching HDF5 metadata configuration..."
        )

        # 2. Fallback: Open the raw H5 file and strip out 'quantization_config'
        try:
            with h5py.File(model_path, "r+") as f:
                if "model_config" in f.attrs:
                    config_data = json.loads(
                        f.attrs["model_config"].decode("utf-8")
                    )

                    # Look through sequential or functional layer configs
                    if (
                        "config" in config_data
                        and "layers" in config_data["config"]
                    ):
                        for layer in config_data["config"]["layers"]:
                            if (
                                "config" in layer
                                and "quantization_config" in layer["config"]
                            ):
                                layer["config"].pop("quantization_config", None)

                    # Save the cleaned layout config back into the file attributes
                    f.attrs["model_config"] = json.dumps(config_data).encode(
                        "utf-8"
                    )

            # Try loading the newly modified file structure
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as deep_error:
            raise RuntimeError(
                f"Failed to automatically patch the model architecture: {deep_error}"
            )


try:
    model = load_model_safely()
    st.success("🎯 Neural Network loaded successfully!")
except Exception as e:
    st.error(f"Critical error loading model file: {e}")


def preprocess_image(img) -> np.ndarray:
    """Preprocess image to match the input layout of the trained ANN."""
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        clean_bg = Image.new("RGBA", img.size, (255, 255, 255))
        clean_bg.paste(img, (0, 0), img.convert("RGBA"))
        img = clean_bg

    img = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)

    corners = [
        img_array[0, 0],
        img_array[0, 27],
        img_array[27, 0],
        img_array[27, 27],
    ]
    if np.mean(corners) > 127:
        img_array = 255 - img_array

    return (img_array.astype("float32") / 255.0).reshape(1, 784)


uploaded_file = st.file_uploader(
    "Upload a handwritten digit image...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=180)

    if st.button("Analyze Pattern", type="primary"):
        with st.spinner("Analyzing..."):
            processed_data = preprocess_image(image)
            predictions = model.predict(processed_data)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            st.success(f"### Predicted Value: {predicted_class}")
            st.metric(
                label="Classification Confidence", value=f"{confidence:.2%}"
            )

            chart_data = {
                str(i): float(prob) for i, prob in enumerate(predictions[0])
            }
            st.bar_chart(chart_data)