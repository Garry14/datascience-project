import streamlit as st
import pandas as pd

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown(
    """
    Pada bagian ini, kita akan melakukan eksplorasi data untuk memahami distribusi, korelasi, dan pola-pola yang ada dalam dataset.
    """
)
# Load dataset
data = pd.read_excel("./Telco_customer_churn.xlsx")
st.subheader("📂 Dataset Overview")
st.dataframe(data.head())

st.subheader("📈 Distribusi Target Variable (Churn)")
st.bar_chart(data["Churn Value"].value_counts())

st.subheader("📉 Distribusi Fitur Numerik")
st.dataframe(data.describe())

st.subheader("📊 Distribusi Fitur Kategorikal")
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    st.write(f"### {col}")
    st.bar_chart(data[col].value_counts())
