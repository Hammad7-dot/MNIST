import io
import os
import requests
import streamlit as st
from PIL import Image

# Network coordinates targeting internal container communications
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000/predict")

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 Deep Learning MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader(
    "Choose a digit image file...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Display the uploaded visual context directly to the browser view pane
    image = Image.open(uploaded_file)
    st.image(
        image, caption="Uploaded File Context", width=180, use_column_width=False
    )

    if st.button("Analyze Pattern", type="primary"):
        with st.spinner("Processing through Neural Network..."):
            try:
                # Convert active file reference context into payload byte buffer
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)

                # Route post transaction payloads up stream towards API endpoints
                files = {"file": ("image.png", buffer, "image/png")}
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("success"):
                        # Render analytics data summaries cleanly
                        st.success(f"### Predicted Value: {result['prediction']}")
                        st.metric(
                            label="Classification Confidence Level",
                            value=f"{result['confidence']:.2%}",
                        )

                        # Render complete analytical softmax score breakdowns
                        with st.expander("Show Probability Distribution"):
                            st.bar_chart(result["probabilities"])
                    else:
                        st.error(f"Processing Failure: {result.get('error')}")
                else:
                    st.error(
                        f"API Endpoint Unreachable (Status: {response.status_code})"
                    )

            except Exception as e:
                st.error(f"System Connection Error: {str(e)}")