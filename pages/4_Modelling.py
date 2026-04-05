import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
import joblib

from preprocessor import (
    load_data,
    split_features_target,
    get_column_types,
    build_preprocessor,
)

# Load and preprocess data
df = load_data()

# Modelling
st.title("📌 Modelling")
st.markdown(
    """
    Pada bagian ini, kita akan membangun model machine learning untuk memprediksi customer churn berdasarkan data yang telah diproses sebelumnya.
    """
)

# Split features and target
st.subheader("🔍 Feature and Target Separation")
X, y = split_features_target(df)
st.write(f"Features shape: {X.shape}")
st.write(f"Target shape: {y.shape}")

# Split testing and training data
st.subheader("🔍 Train-Test Split")
st.write(
    "Splitting data into training and testing sets with test size = 20% and random state = 42."
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Testing set size: {X_test.shape[0]} samples")


# Separate numerical and categorical columns
st.subheader("🔍 Numerical and Categorical Columns")
num_cols, cat_cols = get_column_types(X_train)
st.write(f"Numerical columns: {list(num_cols)}")
st.write(f"Categorical columns: {list(cat_cols)}")

# Preprocessing pipeline
st.subheader("⚙️ Preprocessing Pipeline")
preprocessor = build_preprocessor(num_cols, cat_cols)
st.write(
    "Preprocessing pipeline created with StandardScaler for numerical features and OneHotEncoder for categorical features."
)
st.write("Preprocessor details:")
st.write(preprocessor)

# FULL PIPELINE
st.subheader("⚙️ Full Pipeline with SMOTE-Tomek and Logistic Regression")
pipeline = ImbPipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTETomek(random_state=42)),
        ("model", LogisticRegression()),
    ]
)

# Train
pipeline.fit(X_train, y_train)
st.write("Model trained successfully.")
st.write("Pipeline details:")
st.write(pipeline)

# Save model
st.subheader("💾 Saving the Model")
joblib.dump(pipeline, "model_churn.pkl")
st.write("Model saved as 'model_churn.pkl'.")
