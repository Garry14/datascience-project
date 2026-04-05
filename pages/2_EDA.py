import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    st.write(f"### {col}")
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=col, y="Total Charges", data=data)
    plt.xticks(rotation=45)
    st.pyplot(plt)
