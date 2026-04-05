import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path="./cleaned_data.csv"):
    return pd.read_csv(path)


def split_features_target(df, target="Churn"):
    X = df.drop(columns=[target])
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
