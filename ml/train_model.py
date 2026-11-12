
import json
from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ml.preprocessing import load_data, build_preprocessor


# 1. Define project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load customer dataset
print("\nLoading Customer Dataset...")

X, y = load_data()

print("Dataset Shape:", X.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 4. Create preprocessing pipeline
preprocessor = build_preprocessor()


# 5. Initialize selected classification model
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# 6. Build complete machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# 7. Train the selected model
print("\nTraining Balanced Logistic Regression...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# 8. Generate predictions on test data
y_pred = model.predict(X_test)

churn_class_index = list(model.classes_).index(1)

y_prob = model.predict_proba(X_test)[
    :, churn_class_index
]


# 9. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)


# 10. Display evaluation results
print("\n========== FINAL MODEL EVALUATION ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# 11. Save the trained pipeline
model_path = MODEL_DIR / "churn_model.joblib"

joblib.dump(model, model_path)

print("\nFinal Trained Model Saved To:")
print(model_path)


# 12. Create model information and metrics
model_info = {
    "model_name": "Balanced Logistic Regression",
    "model_version": "1.0",
    "selection_method": "5-fold CV F1-score",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "total_samples": len(X),
    "features": X.columns.tolist(),
    "target": "Churn",
    "prediction_threshold": 0.5,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}


# 13. Save model metrics as JSON
metrics_path = REPORT_DIR / "model_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(model_info, file, indent=4)

print("\nModel Metrics Saved Successfully!")


# 14. Save detailed evaluation report
report_path = REPORT_DIR / "evaluation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("CUSTOMER CHURN PREDICTION SYSTEM\n")
    file.write("================================\n\n")

    file.write("Model: Balanced Logistic Regression\n\n")

    file.write("Evaluation Metrics:\n")
    file.write(f"Accuracy:  {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-score:  {f1:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))

    file.write("\n\nClassification Report:\n")
    file.write(report)

print("Evaluation Report Saved Successfully!")


# 15. Verify saved model
loaded_model = joblib.load(model_path)

sample_customer = X_test.iloc[[0]]

sample_prediction = loaded_model.predict(
    sample_customer
)[0]

sample_probability = loaded_model.predict_proba(
    sample_customer
)[0, churn_class_index]


# 16. Display sample prediction
print("\n========== TEST PREDICTION ==========")

print(
    "Prediction:",
    "Churn" if sample_prediction == 1 else "No Churn"
)

print(
    f"Churn Probability: {sample_probability:.2%}"
)

print("\nStep 12 Completed Successfully!")
