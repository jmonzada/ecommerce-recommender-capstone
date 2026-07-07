# Olist recommender serving image (capstone Step 8)
FROM python:3.11-slim

WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY app/main.py app/main.py
COPY configs/windows.yaml configs/windows.yaml
# only the artefacts the serving path loads (see src/pipeline.py + app/main.py)
COPY models/chosen_config.yaml models/chosen_config.yaml
COPY models/ranker_xgboost.joblib models/ranker_xgboost.joblib
COPY models/feature_columns.json models/feature_columns.json
COPY models/explanations_cache.json models/explanations_cache.json
COPY data/raw/ data/raw/
COPY data/processed/geo_zip_centroids.csv data/processed/geo_zip_centroids.csv

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
