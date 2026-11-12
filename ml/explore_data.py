
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/telco_churn.csv")

# 2. Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# 3. Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# 4. Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# 5. Display data types
print("\nData Types:")
print(df.dtypes)

# 6. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 7. Check invalid TotalCharges values
print("\nInvalid TotalCharges:")
print(
    pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).isna().sum()
)

# 8. Check churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 9. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 10. Identify customers with missing TotalCharges
print("\nCustomers with Missing TotalCharges:")

missing_customers = df[df["TotalCharges"].isna()]

print(
    missing_customers[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
    ]
)

# 11. Check tenure of affected customers
print("\nTenure Values of Missing Customers:")
print(missing_customers["tenure"].value_counts())
