import streamlit as st
from preprocessor import load_data

df = load_data()

st.subheader("📂 Dataset")
st.dataframe(df.head())

st.write("Shape:", df.shape)
