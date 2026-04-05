import streamlit as st

st.title("📌 Introduction")

st.markdown(
    """
Perusahaan berbasis subscripition sangat bergantung pada keberlangsungan pelanggan...
            """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Business Problem")
    st.markdown(
        """
        - Banyak pelanggan churn
        - Revenue menurun
        - Biaya akuisisi pelanggan baru lebih mahal daripada mempertahankan pelanggan lama
        """
    )

with col2:
    st.subheader("🎯 Objective")
    st.markdown(
        """
        Memprediksi kemungkinan customer churn menggunakan Machine Learning
        """
    )

st.subheader("📊 Model Output")
st.markdown(
    """
- 0 = Tidak Churn
- 1 = Churn
    """
)
