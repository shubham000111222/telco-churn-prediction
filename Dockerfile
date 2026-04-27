FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train if models don't exist
RUN python src/models/train.py && python src/models/generate_results.py

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_demo/app.py --server.port 8501 --server.address 0.0.0.0"]
