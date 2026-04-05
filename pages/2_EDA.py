import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

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

st.subheader("🔍 Data Quality Check")
st.write(f"Shape of the dataset: {data.shape}")

st.write("Info:")
st.text(data.info())

st.write("Description:")
st.dataframe(data.describe())

st.write("Missing Values:")
st.dataframe(data.isnull().sum())

# Handle Missing
st.subheader("🧹 Handling Missing Values")
# Fill missing values in 'Total Charges' with median
data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
median_total_charges = data["Total Charges"].median()
data["Total Charges"] = data["Total Charges"].fillna(median_total_charges)

# Fill missing values in 'Churn Reason' with 'No Churn'
data["Churn Reason"] = data["Churn Reason"].fillna("No Churn")
st.write("Missing values in 'Total Charges' filled with median: ", median_total_charges)
st.write("Missing values in 'Churn Reason' filled with 'No Churn'.")
st.dataframe(data.isnull().sum())

# Clear Dataframe
st.subheader("🧹 Data Cleaning")
# Drop CustomerID
data = data.drop(columns=["CustomerID"])
st.write("Dropped 'CustomerID' column.")
st.write("Shape after dropping: ", data.shape)

st.subheader("📊 Boxplot")
# Target Column
data['Churn'] = data['Churn Label']

st.title("📊 Boxplot Analysis")

# Only numeric columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Exclude target and related columns
exclude = ['Churn Value', 'Churn Score']
num_cols = [col for col in num_cols if col not in exclude]

# Plot boxplots in a grid layout
n_cols = 2

for i in range(0, len(num_cols), n_cols):
    cols = st.columns(n_cols)
    
    for j, col in enumerate(num_cols[i:i+n_cols]):
        with cols[j]:
            fig = px.box(
                data,
                x="Churn",
                y=col,
                color="Churn",
                title=f"{col} vs Churn"
            )
            st.plotly_chart(fig, use_container_width=True)

# Save cleaned data for future use
file_path = "cleaned_data.csv"

if not os.path.exists(file_path):
    data.to_csv(file_path, index=False)
    st.write(f"Cleaned data saved to {file_path}")
else:
    st.write(f"{file_path} already exists. Skipping save.")
