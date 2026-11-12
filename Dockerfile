
# Customer Churn Prediction System
# Python 3.12 Docker Image

FROM python:3.12-slim

# Python environment configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# Copy backend application
COPY app/ ./app/

# Copy frontend
COPY frontend/ ./frontend/

# Copy trained machine learning model
COPY models/churn_model.joblib ./models/churn_model.joblib

# Copy model evaluation metrics
COPY reports/model_metrics.json ./reports/model_metrics.json

# Create database directory and non-root user
RUN groupadd --system appgroup \
    && useradd --system \
       --uid 10001 \
       --gid appgroup \
       appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app

# Run application as non-root user
USER appuser

# Internal application port
EXPOSE 8000

# Start FastAPI backend and frontend
CMD ["python", "-m", "uvicorn", "app.web:app", \
     "--host", "0.0.0.0", "--port", "8000"]
