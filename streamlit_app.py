import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from scipy import ndimage

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered",
)

st.title("🔢 Deep Learning MNIST Digit Recognizer")
st.caption("Draw a digit or upload an image, and a CNN trained on MNIST will guess it.")

MODEL_PATH = "mnist_model.keras"


# ----------------------------------------------------------------------------
# Model loading (cached so it only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Couldn't load the model file `{MODEL_PATH}`. "
        f"Make sure it's committed to the repo and the path is correct.\n\nDetails: {e}"
    )
    st.stop()

# Figure out what input shape the model actually expects (flat 784 vs 28x28x1)
# so the app works regardless of how the model was built.
model_input_shape = model.input_shape  # e.g. (None, 784) or (None, 28, 28, 1)
expects_flat = len(model_input_shape) == 2


# ----------------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert an arbitrary PIL image of a handwritten digit into the
    28x28 normalized, centered array format MNIST models expect.
    """
    img = pil_image.convert("L")

    # Auto-invert: MNIST digits are white-on-black. If the image looks like
    # dark ink on a light background (mean pixel is bright), invert it.
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img).astype("float32")

    # Crop to the bounding box of the digit so it isn't tiny in the frame
    threshold = arr.max() * 0.1 if arr.max() > 0 else 0
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Resize the cropped digit to fit in a 20x20 box (MNIST convention),
    # preserving aspect ratio, then pad to 28x28.
    h, w = arr.shape
    if h > w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_w, new_h = 20, max(1, round(h * 20 / w))

    digit_img = Image.fromarray(arr.astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    upper_left = ((28 - new_w) // 2, (28 - new_h) // 2)
    padded.paste(digit_img, upper_left)

    arr = np.array(padded).astype("float32")

    # Center the digit using its center of mass, like the original MNIST
    # preprocessing pipeline — this noticeably improves accuracy on
    # off-center user drawings.
    if arr.sum() > 0:
        cy, cx = ndimage.center_of_mass(arr)
        shift_y, shift_x = np.round(14 - cy).astype(int), np.round(14 - cx).astype(int)
        arr = ndimage.shift(arr, shift=(shift_y, shift_x), mode="constant", cval=0)

    arr = arr / 255.0
    return arr


def predict(arr_28x28: np.ndarray):
    if expects_flat:
        model_input = arr_28x28.reshape(1, 784)
    else:
        model_input = arr_28x28.reshape(1, 28, 28, 1)

    prediction = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    return predicted_digit, confidence, prediction


def show_result(arr_28x28: np.ndarray, prediction_source: str):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(arr_28x28, caption="Model input (28x28)", width=140, clamp=True)
    with col2:
        predicted_digit, confidence, prediction = predict(arr_28x28)
        st.metric("Prediction", predicted_digit)
        st.write(f"Confidence: **{confidence:.2f}%**")
        if confidence < 50:
            st.warning("Low confidence — try drawing the digit larger and more centered.")

    st.bar_chart(
        {"probability": prediction},
        x_label="digit",
        y_label="probability",
    )


# ----------------------------------------------------------------------------
# UI: draw or upload
# ----------------------------------------------------------------------------
tab_draw, tab_upload = st.tabs(["✏️ Draw a digit", "📁 Upload an image"])

with tab_draw:
    st.write("Draw a single digit (0-9) in the box below, then click **Predict**.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_clicked = st.button("Predict", type="primary", key="predict_draw")

    if predict_clicked:
        if canvas_result.image_data is None or np.array(canvas_result.image_data)[:, :, :3].sum() == 0:
            st.info("Please draw a digit first.")
        else:
            rgba = canvas_result.image_data.astype("uint8")
            pil_img = Image.fromarray(rgba, mode="RGBA").convert("L")
            processed = preprocess_image(pil_img)
            show_result(processed, "canvas")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a digit image (any size — it'll be auto-resized)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)
        processed = preprocess_image(image)
        show_result(processed, "upload")

st.divider()
with st.expander("ℹ️ About this app"):
    st.write(
        "This app uses a convolutional neural network trained on the MNIST "
        "handwritten digit dataset (60,000 training images of digits 0-9). "
        "Draw or upload a digit and the model predicts which digit it is, "
        "along with a confidence score for every class."
    )
