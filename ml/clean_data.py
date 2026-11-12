
import pandas as pd
from pathlib import Path
from pandas.api.types import is_string_dtype

# 1. Define project paths
project_dir = Path(__file__).resolve().parent.parent

input_file = project_dir / "data" / "telco_churn.csv"
output_file = project_dir / "data" / "telco_churn_clean.csv"

# 2. Load the original dataset
df = pd.read_csv(input_file)

print("Original Dataset Shape:", df.shape)

# 3. Remove whitespace from column names
df.columns = df.columns.str.strip()

# 4. Remove whitespace from text columns
# Compatible with newer Pandas versions
for column in df.columns:
    if is_string_dtype(df[column]):
        df[column] = df[column].str.strip()

# 5. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 6. Identify missing TotalCharges
missing_mask = df["TotalCharges"].isna()

print("Missing TotalCharges:", missing_mask.sum())

# 7. Check for unexpected missing values
unexpected_missing = missing_mask & (df["tenure"] != 0)

if unexpected_missing.any():
    raise ValueError(
        "Missing TotalCharges found for customers "
        "with nonzero tenure."
    )

# 8. Fill missing TotalCharges for new customers
df.loc[missing_mask, "TotalCharges"] = 0.0

# 9. Check duplicate customer IDs
duplicates = df["customerID"].duplicated().sum()

print("Duplicate Customer IDs:", duplicates)

if duplicates > 0:
    raise ValueError("Duplicate customer IDs detected.")

# 10. Check remaining missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

if df.isnull().any().any():
    raise ValueError("Dataset still contains missing values.")

# 11. Save the cleaned dataset
df.to_csv(output_file, index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned Dataset Shape:", df.shape)
print("Cleaned Dataset Saved To:", output_file)
