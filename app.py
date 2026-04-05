import streamlit as st

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        📊 Telco Customer Churn Prediction
    </h1>
    <h4 style='text-align: center;'>
        Analisis & Prediksi Customer yang Berpotensi Churn
    </h4>
    <hr>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
Gunakan menu di sidebar untuk navigasi:
- Introduction
- EDA
- Modeling
- Prediction
"""
)
