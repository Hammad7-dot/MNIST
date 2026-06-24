FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy all files from your repository into the container
COPY . /app

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# CRITICAL FIX: Point directly to the frontend/app.py directory path
CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]