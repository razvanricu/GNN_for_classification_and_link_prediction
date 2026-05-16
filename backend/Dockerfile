FROM python:3.13-slim

WORKDIR /app

# System deps for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid pulling the large GPU variant
RUN pip install --no-cache-dir \
    torch==2.7.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy data files
COPY saved_models/ ./saved_models/
COPY precomputed/ ./precomputed/
COPY catalog.duckdb ./catalog.duckdb

ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
