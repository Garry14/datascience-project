import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path="./cleaned_data.csv"):
    df = pd.read_csv(path)
    # Fix Total Charges
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    return df


def split_features_target(df, target="Churn"):
    FEATURES = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
    ]
    X = df[FEATURES]
    y = df[target]
    return X, y


def get_column_types(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor
