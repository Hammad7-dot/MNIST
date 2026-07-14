import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
import tensorflow as tf

app = FastAPI(
    title="MNIST Digit Classification API",
    version="1.0"
)

MODEL_PATH = "mnist_model.keras"


# Load model once when the API starts
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load model '{MODEL_PATH}': {e}"
    )


def preprocess_image(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes))

    # Handle transparent images
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        bg = Image.new("RGBA", img.size, (255, 255, 255))
        bg.paste(img, (0, 0), img.convert("RGBA"))
        img = bg

    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # Detect white background and invert
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

    # ANN model expects 784 inputs
    img_array = img_array.reshape(1, 784)

    return img_array


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "tensorflow_version": tf.__version__
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        x = preprocess_image(contents)

        prediction = model.predict(
            x,
            verbose=0
        )

        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "prediction": digit,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

