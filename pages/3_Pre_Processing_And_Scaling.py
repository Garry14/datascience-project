import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pathlib

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
    st.error(f"{file_path} not found. Please run the EDA page first to generate the cleaned data.")


