FROM python:3.11-slim

WORKDIR /app

# System dependencies

RUN apt-get update && apt-get install -y --no-install-recommends 
build-essential 
curl 
&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files

COPY . .

# Streamlit runs on 7860 in Hugging Face Spaces

EXPOSE 7860

# Health check

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Start Streamlit

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
