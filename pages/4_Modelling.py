import streamlit as st
import pandas as pd
import pathlib
import joblib

# Load preprocessed data
file_path = "./preprocessor.pkl"

preprocessor = joblib.load(file_path)

# Modelling
st.title("📌 Modelling")
st.markdown(
    """
    Pada bagian ini, kita akan membangun model machine learning untuk memprediksi customer churn berdasarkan data yang telah diproses sebelumnya.
    """
)
