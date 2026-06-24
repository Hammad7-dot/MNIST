import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
import tensorflow as tf

# 1. THIS VARIABLE MUST BE NAMED EXACTLY "app"
app = FastAPI(title="MNIST Digit Classification API", version="1.0")

# 2. Load your deep learning model configuration globally on startup
MODEL_PATH = "mnist_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image bytes to guarantee 100% accurate structural

    matching with the MNIST-ANN input layer.
    """
    img = Image.open(io.BytesIO(raw_bytes))

    # Eliminate Alpha/Transparency channels
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        white_bg = Image.new("RGBA", img.size, (255, 255, 255))
        white_bg.paste(img, (0, 0), img.convert("RGBA"))
        img = white_bg

    # Convert to standard 8-bit Grayscale and resize
    img = img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Invert background if it is white/light
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

    # Normalize and flatten to 1D vector (1, 784)
    final_array = img_array.astype("float32") / 255.0
    final_array = final_array.reshape(1, 784)

    return final_array


# 3. ENDPOINTS MUST USE THE @app DECORATOR
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "MNIST-ANN"}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_tensor = preprocess_image(contents)

        predictions = model.predict(processed_tensor)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                str(i): round(float(prob), 4) for i, prob in enumerate(predictions[0])
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}