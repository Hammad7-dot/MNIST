# ==========================================
# Base Environment Configuration Stage
# ==========================================
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies needed for compiling or image manipulation extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Backend API Release Target Stage
# ==========================================
FROM base AS backend
COPY ./backend /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Frontend Dashboard Release Target Stage
# ==========================================
FROM base AS frontend
COPY ./frontend /app
EXPOSE 8501
CMD ["streamlit", "run", "app:app", "--server.port=8501", "--server.address=0.0.0.0"]