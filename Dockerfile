FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
EXPOSE 8000

CMD sh -c "uvicorn api:api.py --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 7860"