
# Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts
whether a telecommunications customer is likely to churn.

Built using Python, Scikit-learn, FastAPI, SQLite, HTML,
CSS, JavaScript, and Docker.

## 1. Project Overview

Customer churn is an important challenge for subscription-based
businesses because losing customers can reduce revenue and
increase customer acquisition costs.

This application analyzes customer subscription information
and predicts whether a customer is likely to leave the service.

The system displays:
- Churn prediction (Churn / No Churn)
- Model churn score
- Risk category (Low / Medium / High)
- Machine learning model performance
- Recent prediction history

## 2. Features

### Machine Learning
- Data loading and exploration
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Training and testing data split
- Logistic Regression classification
- Balanced Logistic Regression
- Decision Tree classification
- Random Forest classification
- Five-fold cross-validation
- Model evaluation and comparison
- Scikit-learn preprocessing pipeline
- Trained model saved using Joblib

### Backend
- REST API developed using FastAPI
- Customer data validation using Pydantic
- Integration with the trained ML model
- Prediction and model information endpoints
- SQLite prediction history
- API logging and error handling

### Frontend
- Responsive HTML and CSS interface
- Interactive JavaScript customer form
- Customer churn prediction
- Churn score progress bar
- Risk classification display
- Model performance information
- Recent prediction history
- Sample data and form reset buttons

### Deployment and Testing
- Dockerfile
- Docker Compose
- Persistent SQLite storage using Docker volume
- Automated API and integration tests using Pytest
- Git and GitHub version control

## 3. Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Backend | FastAPI |
| Input Validation | Pydantic |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, HTTPX |
| Containerization | Docker, Docker Compose |
| Version Control | Git, GitHub |

## 4. Dataset


## 4. Dataset

This project uses the **IBM Telco Customer Churn Dataset**.

### Original Dataset Sources

- [IBM Official GitHub Repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [IBM Dataset CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Dataset Characteristics

| Characteristic | Value |
|---|---|
| Customer records | 7,043 |
| Original columns | 21 |
| Target variable | Churn |
| Churned customers | 1,869 |
| Retained customers | 5,174 |
| Churn rate | 26.54% |
| Retention rate | 73.46% |

### Download the Dataset

The original and cleaned CSV files are intentionally excluded
from this public repository.

To reproduce the data analysis, download the dataset from
the original IBM source.

From the project root, run:

```bash
mkdir -p data

curl -fL \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv \
  -o data/telco_churn.csv
```

The downloaded file will be saved as:

`data/telco_churn.csv`

The cleaned dataset can be generated using the project's
data cleaning script:

```bash
python ml/clean_data.py
```

The expected cleaned dataset path is:

`data/telco_churn_clean.csv`

The trained prediction model is already included in the
repository, so downloading the dataset is not required
to run the prediction web application.

### Dataset Licensing

The original dataset is attributed to its respective
rights holders.

This repository does not redistribute the original or
cleaned customer datasets.

The software license for this project does not
automatically apply to third-party data.

Users should review the applicable terms of the
original dataset before downloading, modifying,
or redistributing it.

For additional information, see [DATASET_NOTICE.md](DATASET_NOTICE.md).

## 5. Data Cleaning

The following data cleaning steps were performed:

1. Loaded the original CSV dataset.
2. Removed unnecessary whitespace.
3. Converted TotalCharges into numeric values.
4. Identified 11 invalid TotalCharges entries.
5. Verified that all 11 affected customers had tenure = 0.
6. Replaced those missing TotalCharges values with zero.
7. Checked for duplicate customer IDs.
8. Verified that no missing values remained.
9. Saved the cleaned dataset.

Cleaning script:

ml/clean_data.py

Run:

    python -m ml.clean_data

## 6. Exploratory Data Analysis

EDA was performed to understand customer behavior.

The generated visualizations include:

1. Customer churn distribution
2. Churn rate by contract type
3. Customer tenure distribution
4. Monthly charges comparison

Important observation:

| Contract Type | Churn Rate |
|---------------|------------|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Month-to-month customers had the highest observed churn rate.

EDA script:

ml/eda.py

Run:

    python -m ml.eda

Charts are available in:

reports/figures/

## 7. Machine Learning Methodology

The model uses five input features:

| Feature | Description |
|---------|-------------|
| tenure | Customer tenure in months |
| monthly_charges | Monthly subscription charges |
| total_charges | Total customer charges |
| contract | Customer contract type |
| internet_service | Internet service type |

Target variable:

Churn

Encoding:
- No Churn = 0
- Churn = 1

Dataset splitting:
- Training: 80%
- Testing: 20%

Training samples: 5,634
Testing samples: 1,409

Preprocessing includes:
- Numeric missing-value imputation
- Numeric feature scaling
- Categorical missing-value imputation
- One-hot encoding

Preprocessing is fitted on training data only.

A Scikit-learn Pipeline combines preprocessing and
classification to ensure consistent transformations during
model training and API prediction.

## 8. Model Comparison

Four classification configurations were evaluated using
five-fold cross-validation on the training dataset.

| Model | Accuracy | Recall | F1-score |
|-------|----------|--------|----------|
| Logistic Regression | 79.22% | 49.36% | 55.75% |
| Balanced Logistic Regression | 72.88% | 80.54% | 61.17% |
| Decision Tree | 79.06% | 51.84% | 56.68% |
| Random Forest | 79.32% | 50.43% | 56.41% |

Selected model:

Balanced Logistic Regression

Selection criterion:

Highest cross-validation F1-score.

The balanced model also achieved higher churn recall,
which is useful when identifying customers who may leave.

Model comparison script:

ml/compare_models.py

## 9. Final Model Evaluation

The selected model was evaluated on a held-out test set.

| Metric | Result |
|--------|--------|
| Accuracy | 72.39% |
| Precision | 48.78% |
| Recall | 80.21% |
| F1-score | 60.67% |

Confusion matrix:

    [[720, 315],
     [ 74, 300]]

Interpretation:

- True Negatives: 720
- False Positives: 315
- False Negatives: 74
- True Positives: 300

The model identified 300 of 374 actual churn customers
in the test dataset.

The higher recall comes with additional false positives.

The model is saved at:

models/churn_model.joblib

Evaluation files:

reports/model_metrics.json
reports/evaluation_report.txt
reports/model_comparison.csv

Training command:

    python -m ml.train_model

## 10. Risk Classification

The application uses the following provisional risk levels:

| Model Score | Risk Level |
|-------------|------------|
| Below 30% | Low |
| 30% to below 70% | Medium |
| 70% and above | High |

The classification threshold is 0.50.

Important limitation:

Balanced Logistic Regression outputs have not been
probability-calibrated. The displayed score should not
be interpreted as a verified real-world churn probability.

Risk thresholds are application-defined and have not
been independently validated.

## 11. API Endpoints

FastAPI provides the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Check application health |
| POST | /predict | Predict customer churn |
| GET | /model/info | Retrieve model information |
| GET | /predictions | Retrieve prediction history |

### Prediction Request

POST /predict

Example JSON:

    {
      "tenure": 12,
      "monthly_charges": 75.5,
      "total_charges": 906,
      "contract": "Month-to-month",
      "internet_service": "Fiber optic"
    }

### Prediction Response

Example result from the trained model:

    {
      "prediction": "Churn",
      "probability": 0.8103,
      "risk_level": "High"
    }

### Prediction History

GET /predictions

Supports:
- limit
- offset

Example:

    /predictions?limit=10&offset=0

History records are stored in SQLite.

## 12. Installation on Ubuntu

Requirements:
- Python 3.11 or newer
- Git
- pip
- Python virtual environment support

Clone the repository:

    git clone https://github.com/tirthatanu/customer-churn-prediction.git

Open the project:

    cd customer-churn-prediction

Create a virtual environment:

    python3 -m venv .venv

Activate the environment:

    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

## 13. Run the Application Locally

From the project root directory, run:

    python -m uvicorn app.web:app --reload --port 8001

Open the website:

http://127.0.0.1:8001/

FastAPI documentation:

http://127.0.0.1:8001/docs

Health check:

http://127.0.0.1:8001/health

The SQLite database is initialized automatically.

The application uses the saved model in:

models/churn_model.joblib

Stop the development server using Ctrl+C.

## 14. Docker Deployment

Docker and Docker Compose are supported.

Build and start:

    docker compose up --build -d

Check container status:

    docker compose ps

Check logs:

    docker compose logs --tail=50

Open the Docker website:

http://127.0.0.1:8002/

Docker API documentation:

http://127.0.0.1:8002/docs

Stop the container:

    docker compose down

SQLite prediction history is preserved using a
Docker named volume.

The Docker database is separate from the local
development database.

## 15. Automated Testing

Automated tests are implemented using Pytest and HTTPX.

Run:

    python -m pytest tests/test_app.py -v

Verified result:

28 tests passed.

Testing includes:
- API health checks
- Frontend accessibility
- CSS and JavaScript availability
- Model information
- Prediction response validation
- SQLite prediction storage
- Invalid customer inputs
- Missing required fields
- Prediction history pagination
- Risk classification boundaries
- Missing model handling

Tests use temporary SQLite databases to prevent
modifying the real prediction history.

## 16. Project Structure

    customer-churn-prediction/
    |
    |-- app/
    |   |-- __init__.py
    |   |-- main.py
    |   |-- database.py
    |   |-- web.py
    |
    |-- data/
    |   |-- telco_churn.csv
    |   |-- telco_churn_clean.csv
    |
    |-- frontend/
    |   |-- index.html
    |   |-- style.css
    |   |-- script.js
    |
    |-- ml/
    |   |-- __init__.py
    |   |-- explore_data.py
    |   |-- clean_data.py
    |   |-- eda.py
    |   |-- preprocessing.py
    |   |-- compare_models.py
    |   |-- train_model.py
    |
    |-- models/
    |   |-- churn_model.joblib
    |
    |-- reports/
    |   |-- evaluation_report.txt
    |   |-- model_metrics.json
    |   |-- model_comparison.csv
    |   |-- figures/
    |
    |-- tests/
    |   |-- test_app.py
    |
    |-- Dockerfile
    |-- compose.yaml
    |-- .dockerignore
    |-- .gitignore
    |-- requirements.txt
    |-- README.md

## 17. Screenshots and Demo

The final project submission should include screenshots of:

1. Customer information form
2. Successful churn prediction
3. Model evaluation metrics
4. Prediction history table
5. FastAPI Swagger documentation
6. Docker container running successfully
7. Automated test results

A short demonstration video may also be included.

## 18. Current Limitations

- The model uses five selected customer features.
- Churn probabilities are not yet calibrated.
- Risk thresholds are provisional.
- The training dataset represents a specific telecom sample.
- Performance may differ on other real-world datasets.
- The application does not currently include authentication.
- Prediction history is intended for local demonstration.
- Public deployment would require additional security controls.

## 19. Future Improvements

Possible future enhancements include:

- Probability calibration and threshold optimization
- Additional customer features
- More advanced ML algorithms
- Feature importance visualization
- Authentication and authorization
- Monitoring and model drift detection
- Improved retention recommendations
- Cloud deployment

## 20. Repository

GitHub:

https://github.com/tirthatanu/customer-churn-prediction

Project Name:

Customer Churn Prediction System

Project Type:

End-to-End Machine Learning Web Application
