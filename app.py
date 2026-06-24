import io
import json
import h5py
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
import tensorflow as tf

app = FastAPI(title="MNIST Digit Classification API", version="1.0")

MODEL_PATH = "mnist_model.h5"


def load_model_safely():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)

    except Exception:

        def remove_quantization(obj):
            if isinstance(obj, dict):
                obj.pop("quantization_config", None)
                for value in obj.values():
                    remove_quantization(value)

            elif isinstance(obj, list):
                for item in obj:
                    remove_quantization(item)

        with h5py.File(MODEL_PATH, "r+") as f:
            if "model_config" in f.attrs:
                config = f.attrs["model_config"]

                if isinstance(config, bytes):
                    config = config.decode("utf-8")

                config_json = json.loads(config)

                remove_quantization(config_json)

                f.attrs["model_config"] = json.dumps(config_json)

        return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = load_model_safely()


def preprocess_image(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes))

    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        bg = Image.new("RGBA", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), img.convert("RGBA"))
        img = bg

    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    corners = [
        img_array[0, 0],
        img_array[0, 27],
        img_array[27, 0],
        img_array[27, 27],
    ]

    if np.mean(corners) > 127:
        img = ImageOps.invert(img)
        img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    # ANN input
    img_array = img_array.reshape(1, 784)

    return img_array


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "tensorflow": tf.__version__
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    x = preprocess_image(contents)

    prediction = model.predict(x, verbose=0)

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "prediction": digit,
        "confidence": round(confidence * 100, 2)
    }