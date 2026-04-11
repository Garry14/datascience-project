import streamlit as st
import pandas as pd
import pickle

# =============================
# LOAD MODEL
# =============================
# Ganti dengan path model lu
with open("./v4_rf_generated_model.pkl", "rb") as f:
    model = pickle.load(f)


# =============================
# ENCODER FUNCTION (PUNYA LU)
# =============================
def encode_input(data):
    columns = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "senior_citizen_no",
        "senior_citizen_yes",
        "partner_no",
        "partner_yes",
        "dependents_no",
        "dependents_yes",
        "phone_service_no",
        "phone_service_yes",
        "multiple_lines_no",
        "multiple_lines_no_phone_service",
        "multiple_lines_yes",
        "internet_service_dsl",
        "internet_service_fiber_optic",
        "internet_service_no",
        "online_security_no",
        "online_security_no_internet_service",
        "online_security_yes",
        "online_backup_no",
        "online_backup_no_internet_service",
        "online_backup_yes",
        "device_protection_no",
        "device_protection_no_internet_service",
        "device_protection_yes",
        "tech_support_no",
        "tech_support_no_internet_service",
        "tech_support_yes",
        "streaming_tv_no",
        "streaming_tv_no_internet_service",
        "streaming_tv_yes",
        "streaming_movies_no",
        "streaming_movies_no_internet_service",
        "streaming_movies_yes",
        "contract_month_to_month",
        "contract_one_year",
        "contract_two_year",
        "paperless_billing_no",
        "paperless_billing_yes",
        "payment_method_bank_transfer_automatic",
        "payment_method_credit_card_automatic",
        "payment_method_electronic_check",
        "payment_method_mailed_check",
    ]

    encoded = pd.DataFrame(0, index=[0], columns=columns)

    # numeric
    encoded["tenure_months"] = data["Tenure Months"][0]
    encoded["monthly_charges"] = data["Monthly Charges"][0]
    encoded["total_charges"] = data["Total Charges"][0]

    def encode_yes_no(col, prefix):
        val = data[col][0]
        encoded[f"{prefix}_yes"] = 1 if val == "Yes" else 0
        encoded[f"{prefix}_no"] = 1 if val == "No" else 0

    encode_yes_no("Senior Citizen", "senior_citizen")
    encode_yes_no("Partner", "partner")
    encode_yes_no("Dependents", "dependents")
    encode_yes_no("Phone Service", "phone_service")
    encode_yes_no("Paperless Billing", "paperless_billing")

    ml = data["Multiple Lines"][0]
    if ml == "Yes":
        encoded["multiple_lines_yes"] = 1
    elif ml == "No":
        encoded["multiple_lines_no"] = 1
    else:
        encoded["multiple_lines_no_phone_service"] = 1

    internet = data["Internet Service"][0]
    if internet == "DSL":
        encoded["internet_service_dsl"] = 1
    elif internet == "Fiber optic":
        encoded["internet_service_fiber_optic"] = 1
    else:
        encoded["internet_service_no"] = 1

    def encode_service(col, prefix):
        val = data[col][0]
        if val == "Yes":
            encoded[f"{prefix}_yes"] = 1
        elif val == "No":
            encoded[f"{prefix}_no"] = 1
        else:
            encoded[f"{prefix}_no_internet_service"] = 1

    encode_service("Online Security", "online_security")
    encode_service("Online Backup", "online_backup")
    encode_service("Device Protection", "device_protection")
    encode_service("Tech Support", "tech_support")
    encode_service("Streaming TV", "streaming_tv")
    encode_service("Streaming Movies", "streaming_movies")

    contract = data["Contract"][0]
    if contract == "Month-to-month":
        encoded["contract_month_to_month"] = 1
    elif contract == "One year":
        encoded["contract_one_year"] = 1
    else:
        encoded["contract_two_year"] = 1

    payment = data["Payment Method"][0]
    if payment == "Bank transfer (automatic)":
        encoded["payment_method_bank_transfer_automatic"] = 1
    elif payment == "Credit card (automatic)":
        encoded["payment_method_credit_card_automatic"] = 1
    elif payment == "Electronic check":
        encoded["payment_method_electronic_check"] = 1
    else:
        encoded["payment_method_mailed_check"] = 1

    return encoded


# =============================
# UI
# =============================
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.write("Isi data di sidebar lalu klik Predict")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("Customer Input")

# Numeric
tenure = st.sidebar.number_input("Tenure Months", 0, 100, 1)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 1000.0, 50.0)
total = st.sidebar.number_input("Total Charges", 0.0, 100000.0, 50.0)


# Yes/No
def yes_no(label):
    return st.sidebar.selectbox(label, ["Yes", "No"])


senior = yes_no("Senior Citizen")
partner = yes_no("Partner")
dependents = yes_no("Dependents")
phone = yes_no("Phone Service")
paperless = yes_no("Paperless Billing")

# Multiple Lines
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)

# Internet
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])


def service(label):
    return st.sidebar.selectbox(label, ["Yes", "No", "No internet service"])


online_security = service("Online Security")
online_backup = service("Online Backup")
device_protection = service("Device Protection")
tech_support = service("Tech Support")
streaming_tv = service("Streaming TV")
streaming_movies = service("Streaming Movies")

# Contract
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Payment
payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
)

# =============================
# DATAFRAME
# =============================
input_data = pd.DataFrame(
    {
        "Tenure Months": [tenure],
        "Monthly Charges": [monthly],
        "Total Charges": [total],
        "Senior Citizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Phone Service": [phone],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless],
        "Payment Method": [payment],
    }
)


# =============================
# DECODE ONE HOT ENCODED DATA
# =============================
def decode_one_hot(encoded_df):
    result = {}

    mapping = {
        # binary
        "senior_citizen": "Senior Citizen",
        "partner": "Partner",
        "dependents": "Dependents",
        "phone_service": "Phone Service",
        "paperless_billing": "Paperless Billing",
        # multiple lines
        "multiple_lines": "Multiple Lines",
        # internet service
        "internet_service": "Internet Service",
        # services
        "online_security": "Online Security",
        "online_backup": "Online Backup",
        "device_protection": "Device Protection",
        "tech_support": "Tech Support",
        "streaming_tv": "Streaming TV",
        "streaming_movies": "Streaming Movies",
        # contract
        "contract": "Contract",
        # payment
        "payment_method": "Payment Method",
    }

    for col in encoded_df.columns:
        value = encoded_df[col].values[0]

        # =========================
        # 1. NUMERIC
        # =========================
        if col == "tenure_months":
            result["Tenure"] = f"{value} Month(s)"
            continue

        if col in ["monthly_charges", "total_charges"]:
            result[col.replace("_", " ").title()] = f"${value:.2f}"
            continue


        # =========================
        # 2. ONE HOT
        # =========================
        if value == 1:
            parts = col.split("_")

            # ambil prefix group
            if col.startswith("payment_method"):
                group = "payment_method"
            elif col.startswith("internet_service"):
                group = "internet_service"
            elif col.startswith("multiple_lines"):
                group = "multiple_lines"
            elif col.startswith("contract"):
                group = "contract"
            elif col.startswith("senior_citizen"):
                group = "senior_citizen"
            elif col.startswith("paperless_billing"):
                group = "paperless_billing"
            else:
                group = "_".join(parts[:2])

            label = mapping.get(group, group.replace("_", " ").title())

            # ambil value (YES/NO/DSL/etc)
            value_label = col.replace(group + "_", "")
            value_label = value_label.replace("_", " ").title()

            # special cleanup biar bagus UI
            if value_label == "":
                value_label = "Yes"
            if value_label == "No Internet Service":
                value_label = "No Internet Service"

            result[label] = value_label

    return result


# =============================
# PREDICT
# =============================
if st.sidebar.button("🚀 Predict"):

    encoded = encode_input(input_data)

    prediction = model.predict(encoded)[0]
    proba = model.predict_proba(encoded)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error(f"Customer akan CHURN ❌ (Kemungkinan Churn: {proba:.2%})")
    else:
        st.success(f"Customer TIDAK churn ✅ (Kemungkinan Churn: {proba:.2%})")

    st.write("### 🔍 Encoded Data")
    st.subheader("Customer Summary")

    decoded = decode_one_hot(encoded)

    items = list(decoded.items())

    col1, col2 = st.columns(2)

    for i, (k, v) in enumerate(items):
        if i % 2 == 0:
            with col1:
                st.write(f"**{k} :** {v}")
        else:
            with col2:
                st.write(f"**{k} :** {v}")
