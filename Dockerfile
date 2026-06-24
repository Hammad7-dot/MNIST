FROM python:3.10-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all application files to the container tracking workdir
COPY . .

# Expose port 7860 (Hugging Face strict requirement)
EXPOSE 7860

# Run Streamlit directly on port 7860 with absolute file pathways
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]