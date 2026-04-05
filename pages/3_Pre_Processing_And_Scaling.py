import streamlit as st
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
import joblib

st.title("⚙️ Pre-Processing and Scaling")
st.markdown(
    """
    Pada bagian ini, kita akan melakukan pra-pemrosesan data dan penskalaan fitur untuk mempersiapkan dataset sebelum digunakan dalam model machine learning.
    """
)

# Load dataset
file_path = "./cleaned_data.csv"

if pathlib.Path(file_path).exists():
    data = pd.read_csv(file_path)
    st.subheader("📂 Dataset Overview")
    st.dataframe(data.head())
else:
    st.error(
        f"{file_path} not found. Please run the EDA page first to generate the cleaned data."
    )

# Split features and target
st.subheader("🔍 Feature and Target Separation")
X = data.drop(columns=["Churn"])
y = data["Churn"]
st.write(f"Features shape: {X.shape}")
st.write(f"Target shape: {y.shape}")

# Split testing and training data
st.subheader("🔍 Train-Test Split")
st.write(
    "Splitting data into training and testing sets with test size = 20% and random state = 42."
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Testing set size: {X_test.shape[0]} samples")

# Separate numerical and categorical columns
st.subheader("🔍 Numerical and Categorical Columns")
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns
st.write(f"Numerical columns: {X['Country'].dtype}")
st.write(f"Numerical columns: {list(num_cols)}")
st.write(f"Categorical columns: {list(cat_cols)}")

# Preprocessing pipeline
st.subheader("⚙️ Preprocessing Pipeline")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
st.write(
    "Preprocessing completed. Numerical features scaled and categorical features encoded."
)
st.write(f"Transformed training features shape: {X_train_transformed.shape}")

# Handling imbalanced data
st.subheader("⚖️ Handling Imbalanced Data")
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(
    X_train_transformed, y_train
)
st.write(f"Original training set size: {X_train_transformed.shape[0]} samples")
st.write(f"Resampled training set size: {X_train_resampled.shape[0]} samples")

# Save preprocessor
preprocessor_path = "preprocessor.pkl"
if pathlib.Path(preprocessor_path).exists():
    st.warning(
        f"{preprocessor_path} already exists. It will be overwritten with the new preprocessor."
    )
else:
    st.write(f"{preprocessor_path} does not exist. It will be created.")
joblib.dump(preprocessor, preprocessor_path)
st.write(f"Preprocessor saved to {preprocessor_path}")
