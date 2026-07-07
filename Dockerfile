# Olist recommender serving image (capstone Step 8)
FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/
COPY data/raw/ data/raw/
COPY data/processed/geo_zip_centroids.csv data/processed/geo_zip_centroids.csv

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
