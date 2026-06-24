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

        # 2. Fallback: Open raw H5 file and strip out 'quantization_config' safely
        try:
            with h5py.File(model_path, "r+") as f:
                if "model_config" in f.attrs:
                    raw_config = f.attrs["model_config"]

                    # FIX: Safely decode only if the data is retrieved as bytes
                    if isinstance(raw_config, bytes):
                        config_str = raw_config.decode("utf-8")
                    else:
                        config_str = raw_config

                    config_data = json.loads(config_str)

                    # Traverse through layers to safely pop the unrecognized parameter
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

                    # Save cleaned layout configuration back into file attributes
                    if isinstance(raw_config, bytes):
                        f.attrs["model_config"] = json.dumps(
                            config_data
                        ).encode("utf-8")
                    else:
                        f.attrs["model_config"] = json.dumps(config_data)

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

# ... rest of your code (preprocess_image, file_uploader, etc.) remains exactly the same!