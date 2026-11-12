
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define project paths
project_dir = Path(__file__).resolve().parent.parent

data_file = project_dir / "data" / "telco_churn_clean.csv"
figures_dir = project_dir / "reports" / "figures"

# Create output folder
figures_dir.mkdir(parents=True, exist_ok=True)

# 2. Load cleaned dataset
df = pd.read_csv(data_file)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# 3. Analyze churn distribution
churn_counts = df["Churn"].value_counts()
churn_percentage = (
    df["Churn"].value_counts(normalize=True) * 100
)

print("\nChurn Distribution:")
print(churn_counts)

print("\nChurn Percentage:")
print(churn_percentage.round(2))

# 4. Chart 1: Churn Distribution
plt.figure(figsize=(7, 5))

churn_counts = churn_counts.reindex(["No", "Yes"])

bars = plt.bar(
    churn_counts.index,
    churn_counts.values,
    color=["steelblue", "salmon"]
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.bar_label(bars)

plt.tight_layout()
plt.savefig(
    figures_dir / "churn_distribution.png",
    dpi=150
)
plt.close()

print("\nChart 1 Saved!")

# 5. Analyze churn by contract type
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nChurn Rate by Contract Type (%):")
print(contract_churn.round(2))

# 6. Chart 2: Contract Churn Rate
plt.figure(figsize=(8, 5))

bars = plt.bar(
    contract_churn.index,
    contract_churn.values,
    color="cornflowerblue"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.bar_label(bars, fmt="%.1f%%")

plt.tight_layout()
plt.savefig(
    figures_dir / "contract_churn.png",
    dpi=150
)
plt.close()

print("Chart 2 Saved!")

# 7. Chart 3: Tenure Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df.loc[df["Churn"] == "No", "tenure"],
    bins=20,
    alpha=0.6,
    label="No Churn"
)

plt.hist(
    df.loc[df["Churn"] == "Yes", "tenure"],
    bins=20,
    alpha=0.6,
    label="Churn"
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()

plt.tight_layout()
plt.savefig(
    figures_dir / "tenure_distribution.png",
    dpi=150
)
plt.close()

print("Chart 3 Saved!")

# 8. Chart 4: Monthly Charges Comparison
plt.figure(figsize=(7, 5))

monthly_no = df.loc[
    df["Churn"] == "No", "MonthlyCharges"
]

monthly_yes = df.loc[
    df["Churn"] == "Yes", "MonthlyCharges"
]

plt.boxplot([monthly_no, monthly_yes])

plt.xticks([1, 2], ["No Churn", "Churn"])

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Customer Status")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig(
    figures_dir / "monthly_charges.png",
    dpi=150
)
plt.close()

print("Chart 4 Saved!")

print("\nEDA Completed Successfully!")
print("Charts saved in:", figures_dir)
