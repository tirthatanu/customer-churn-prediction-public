
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ml.preprocessing import load_data, build_preprocessor


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
print("\nLoading Customer Dataset...")

X, y = load_data()

print("Dataset Shape:", X.shape)


# 3. Split data using the same settings
# as our previous training script
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 4. Define classification algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}


# 5. Configure cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 6. Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# 7. Train and evaluate each algorithm
results = []

print("\nStarting Model Comparison...")

for model_name, classifier in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    results.append({
        "Model": model_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean()
    })

    print("Evaluation Completed!")


# 8. Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)


# 9. Display results
print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# 10. Identify highest cross-validation F1-score
best_model_name = results_df.iloc[0]["Model"]

print("\nHighest CV F1-score:")
print(best_model_name)


# 11. Save comparison report
output_file = REPORT_DIR / "model_comparison.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nComparison Report Saved To:")
print(output_file)

print("\nStep 11 Completed Successfully!")
