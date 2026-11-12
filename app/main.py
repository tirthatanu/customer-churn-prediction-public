
import json
import logging
import sqlite3

from pathlib import Path
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from app.database import (
    init_db,
    save_prediction,
    get_prediction_history
)


# 1. Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 2. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"

METRICS_PATH = PROJECT_DIR / "reports" / "model_metrics.json"


# 3. Initialize database at application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("SQLite database initialized successfully")
    yield


# 4. Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction System",
    description="Machine Learning API for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan
)


# 5. Customer input validation
class CustomerInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "tenure": 12,
                "monthly_charges": 75.5,
                "total_charges": 906,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic"
            }
        }
    )

    tenure: int = Field(
        ...,
        ge=0,
        description="Customer tenure in months"
    )

    monthly_charges: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False
    )

    total_charges: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False
    )

    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]


# 6. Prediction response model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str


# 7. Prediction history record model
class PredictionHistoryRecord(BaseModel):
    id: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    internet_service: str
    prediction: str
    probability: float
    risk_level: str
    created_at: str


# 8. Prediction history response model
class PredictionHistoryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    predictions: list[PredictionHistoryRecord]


# 9. Load trained ML model
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logger.info("Machine learning model loaded successfully")

    return model


# 10. Determine risk level
def get_risk_level(probability: float) -> str:

    if probability < 0.30:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    return "High"


# 11. Health check endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Customer Churn Prediction API is running"
    }


# 12. Customer churn prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerInput):

    try:
        # Load trained model
        model = load_model()

        # Prepare customer information
        customer_data = pd.DataFrame([{
            "tenure": customer.tenure,
            "monthly_charges": customer.monthly_charges,
            "total_charges": customer.total_charges,
            "contract": customer.contract,
            "internet_service": customer.internet_service
        }])

        # Generate prediction
        prediction = int(
            model.predict(customer_data)[0]
        )

        # Identify churn class index
        churn_index = list(
            model.classes_
        ).index(1)

        # Calculate probability
        probability = float(
            model.predict_proba(customer_data)[0][churn_index]
        )

        probability = round(probability, 4)

        # Convert prediction to readable text
        prediction_label = (
            "Churn" if prediction == 1 else "No Churn"
        )

        # Determine risk classification
        risk_level = get_risk_level(probability)

        # Save prediction to SQLite
        record_id = save_prediction(
            tenure=customer.tenure,
            monthly_charges=customer.monthly_charges,
            total_charges=customer.total_charges,
            contract=customer.contract,
            internet_service=customer.internet_service,
            prediction=prediction_label,
            probability=probability,
            risk_level=risk_level
        )

        logger.info(
            "Prediction saved successfully. Record ID: %s",
            record_id
        )

        # Return prediction
        return {
            "prediction": prediction_label,
            "probability": probability,
            "risk_level": risk_level
        }

    except FileNotFoundError as exc:

        logger.error("Model missing: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available"
        ) from exc

    except sqlite3.Error as exc:

        logger.exception("Database operation failed")

        raise HTTPException(
            status_code=503,
            detail="Prediction history database is unavailable"
        ) from exc

    except Exception as exc:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction"
        ) from exc


# 13. Model information endpoint
@app.get("/model/info")
def get_model_info():

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained model is not available"
        )

    try:
        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            model_info = json.load(file)

        if not isinstance(model_info, dict):
            raise ValueError(
                "Invalid model information format"
            )

        logger.info(
            "Model information retrieved successfully"
        )

        return model_info

    except FileNotFoundError as exc:

        logger.error(
            "Model metrics file missing: %s",
            exc
        )

        raise HTTPException(
            status_code=503,
            detail="Model information is not available"
        ) from exc

    except (json.JSONDecodeError, ValueError, OSError) as exc:

        logger.exception(
            "Unable to retrieve model information"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve model information"
        ) from exc


# 14. Prediction history endpoint
@app.get(
    "/predictions",
    response_model=PredictionHistoryResponse
)
def get_predictions(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of records to retrieve"
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip"
    )
):

    try:
        # Fetch records from SQLite
        total, predictions = get_prediction_history(
            limit=limit,
            offset=offset
        )

        logger.info(
            "Retrieved %s prediction records",
            len(predictions)
        )

        # Return prediction history
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "predictions": predictions
        }

    except sqlite3.Error as exc:

        logger.exception(
            "Failed to retrieve prediction history"
        )

        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve prediction history"
        ) from exc
