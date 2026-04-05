import streamlit as st
import pandas as pd
import joblib

# Load Model and Preprocessor
pipeline = joblib.load("./model_churn.pkl")

st.title("📌 Prediction")
st.markdown(
    """Pada bagian ini, kita akan menggunakan model yang telah dibangun untuk memprediksi apakah seorang pelanggan akan churn atau tidak berdasarkan input fitur yang diberikan."""
)
# data = pd.DataFrame(
#     {
#         "tenure": [12],
#         "Monthly Charges": [70.35],
#         "Contract": ["Month-to-month"],
#         "Internet Service": ["Fiber optic"],
#     }
# )

data = pd.DataFrame(
    {
        "Gender": ["Male"],
        "Senior Citizen": ["1"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "Tenure Months": [1],
        "Phone Service": ["No"],
        "Multiple Lines": ["No"],
        "Internet Service": ["Fiber optic"],
        "Online Security": ["No"],
        "Online Backup": ["Yes"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["Yes"],
        "Contract": ["Month-to-month"],
        "Paperless Billing": ["Yes"],
        "Payment Method": ["Electronic check"],
        "Monthly Charges": [100],
        "Total Charges": [1400],
    }
)

st.subheader("📂 Input Data")
st.dataframe(data)
st.subheader("🔍 Preprocessing Input Data")

# Prediction
st.subheader("🎯 Prediction Result")
prediction = pipeline.predict(data)
probability = pipeline.predict_proba(data)[0][1]
st.write(f"Predicted Probability of Churn: {probability:.2f}")
st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
