
import pytest

from fastapi.testclient import TestClient

import app.database as database
import app.main as api

from app.web import app


# Sample customer for prediction testing
SAMPLE_CUSTOMER = {
    "tenure": 12,
    "monthly_charges": 75.5,
    "total_charges": 906,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic"
}


# 1. Create an isolated testing database
@pytest.fixture
def client(tmp_path, monkeypatch):

    test_db = tmp_path / "test_predictions.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_db
    )

    # Run FastAPI startup and shutdown correctly
    with TestClient(app) as test_client:
        yield test_client


# 2. Test API health
def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# 3. Test frontend homepage
def test_frontend_homepage(client):

    response = client.get("/")

    assert response.status_code == 200

    assert "Customer Churn Prediction" in response.text

    assert "predictionForm" in response.text


# 4. Test frontend CSS and JavaScript
def test_frontend_assets(client):

    css_response = client.get("/style.css")

    assert css_response.status_code == 200

    assert ".workspace" in css_response.text

    js_response = client.get("/script.js")

    assert js_response.status_code == 200

    assert "loadPredictionHistory" in js_response.text


# 5. Test model information
def test_model_info_endpoint(client):

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == (
        "Balanced Logistic Regression"
    )

    assert data["total_samples"] == 7043

    assert len(data["features"]) == 5

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:
        assert 0 <= data[metric] <= 1


# 6. Test prediction and SQLite saving
def test_prediction_and_database(client):

    # Database must initially be empty
    initial = client.get("/predictions")

    assert initial.status_code == 200

    assert initial.json()["total"] == 0

    # Submit customer information
    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [
        "Churn",
        "No Churn"
    ]

    assert 0 <= data["probability"] <= 1

    assert data["risk_level"] in [
        "Low",
        "Medium",
        "High"
    ]

    # Risk category must match score
    assert data["risk_level"] == api.get_risk_level(
        data["probability"]
    )

    # Verify SQLite history
    history_response = client.get("/predictions")

    assert history_response.status_code == 200

    history = history_response.json()

    assert history["total"] == 1

    assert len(history["predictions"]) == 1

    saved = history["predictions"][0]

    assert saved["tenure"] == 12

    assert saved["contract"] == "Month-to-month"

    assert saved["prediction"] == data["prediction"]

    assert saved["probability"] == data["probability"]

    assert saved["risk_level"] == data["risk_level"]

    assert saved["created_at"]


# 7. Test invalid customer information
@pytest.mark.parametrize(
    "invalid_changes",
    [
        {"tenure": -1},
        {"tenure": 2.5},
        {"monthly_charges": -50},
        {"total_charges": -100},
        {"contract": "Weekly"},
        {"internet_service": "Satellite"},
        {"unknown_field": "invalid"}
    ]
)
def test_invalid_customer_input(
    client,
    invalid_changes
):

    payload = SAMPLE_CUSTOMER.copy()

    payload.update(invalid_changes)

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422

    # Invalid submissions must not be saved
    history = client.get("/predictions").json()

    assert history["total"] == 0


# 8. Test missing required fields
@pytest.mark.parametrize(
    "missing_field",
    [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract",
        "internet_service"
    ]
)
def test_missing_customer_fields(
    client,
    missing_field
):

    payload = SAMPLE_CUSTOMER.copy()

    payload.pop(missing_field)

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# 9. Test prediction history pagination
def test_prediction_history_pagination(client):

    # Create three predictions
    for tenure in [12, 24, 36]:

        customer = SAMPLE_CUSTOMER.copy()

        customer["tenure"] = tenure

        response = client.post(
            "/predict",
            json=customer
        )

        assert response.status_code == 200

    # Request two records, skipping the newest
    response = client.get(
        "/predictions?limit=2&offset=1"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 3

    assert data["limit"] == 2

    assert data["offset"] == 1

    assert len(data["predictions"]) == 2

    # Newest-first ordering
    ids = [
        record["id"]
        for record in data["predictions"]
    ]

    assert ids == sorted(ids, reverse=True)


# 10. Test invalid pagination
@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1"
    ]
)
def test_invalid_pagination(client, query):

    response = client.get(
        f"/predictions?{query}"
    )

    assert response.status_code == 422


# 11. Test risk classification boundaries
@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "Low"),
        (0.2999, "Low"),
        (0.30, "Medium"),
        (0.6999, "Medium"),
        (0.70, "High"),
        (1.0, "High")
    ]
)
def test_risk_classification(
    probability,
    expected
):

    result = api.get_risk_level(probability)

    assert result == expected


# 12. Test unavailable model handling
def test_missing_model(
    client,
    monkeypatch,
    tmp_path
):

    missing_model = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(
        api,
        "MODEL_PATH",
        missing_model
    )

    # Clear cached model for this test
    api.load_model.cache_clear()

    try:
        response = client.post(
            "/predict",
            json=SAMPLE_CUSTOMER
        )

        assert response.status_code == 503

        info_response = client.get("/model/info")

        assert info_response.status_code == 503

        # Failed predictions must not be saved
        history = client.get("/predictions").json()

        assert history["total"] == 0

    finally:
        api.load_model.cache_clear()
