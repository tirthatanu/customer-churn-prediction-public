
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# 1. Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "telco_churn_clean.csv"


# 2. Define feature names
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges"
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# 3. Load dataset and select features
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Rename columns to match FastAPI input fields
    df = df.rename(columns={
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract",
        "InternetService": "internet_service"
    })

    # Verify required columns exist
    required_columns = FEATURE_COLUMNS + ["Churn"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select input features
    X = df[FEATURE_COLUMNS].copy()

    # Ensure numeric columns have numeric types
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Encode target labels
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    if y.isna().any():
        raise ValueError("Invalid values found in Churn column.")

    return X, y.astype(int)


# 4. Create preprocessing pipeline
def build_preprocessor():

    # Numeric preprocessing
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# 5. Test preprocessing
if __name__ == "__main__":

    X, y = load_data()

    print("\nDataset Loaded Successfully!")

    print("\nSelected Features:")
    print(X.columns.tolist())

    print("\nFeature Dataset Shape:")
    print(X.shape)

    print("\nTarget Distribution:")
    print(y.value_counts())

    # Split dataset before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Dataset Shape:")
    print(X_train.shape)

    print("\nTesting Dataset Shape:")
    print(X_test.shape)

    # Build the preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit preprocessing on training data only
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Apply learned transformations to testing data
    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nProcessed Training Shape:")
    print(X_train_processed.shape)

    print("\nProcessed Testing Shape:")
    print(X_test_processed.shape)

    print("\nPreprocessing Completed Successfully!")
