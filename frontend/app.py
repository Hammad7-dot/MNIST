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

    # 1. Attempt a standard load first
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.warning(
            "Version structural mismatch detected. Patching HDF5 metadata configuration..."
        )

        # 2. Fallback: Recursively strip 'quantization_config' out of the metadata dictionary
        try:

            def strip_quantization_key(data):
                """Recursively search and destroy 'quantization_config' keys in any dict or list."""
                if isinstance(data, dict):
                    data.pop("quantization_config", None)
                    for key, value in data.items():
                        strip_quantization_key(value)
                elif isinstance(data, list):
                    for item in data:
                        strip_quantization_key(item)

            # Open and instantly save changes by wrapping inside a 'with' context manager
            with h5py.File(model_path, "r+") as f:
                if "model_config" in f.attrs:
                    raw_config = f.attrs["model_config"]
                    config_str = (
                        raw_config.decode("utf-8")
                        if isinstance(raw_config, bytes)
                        else raw_config
                    )
                    config_data = json.loads(config_str)

                    # Strip the problematic key wherever it hides in the architecture
                    strip_quantization_key(config_data)

                    # Save cleaned structure back
                    updated_json = json.dumps(config_data)
                    f.attrs["model_config"] = (
                        updated_json.encode("utf-8")
                        if isinstance(raw_config, bytes)
                        else updated_json
                    )

            # 3. Reload the cleanly patched model file structure
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