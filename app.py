import streamlit as st
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #00D4FF;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================
st.markdown(
    '<div class="title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether customer will churn or stay</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================
# INPUT SECTION
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1500.0
    )

with col2:
    st.subheader("Service Details")

    partner = st.selectbox(
        "Partner",
        [0, 1]
    )

    dependents = st.selectbox(
        "Dependents",
        [0, 1]
    )

    phone_service = st.selectbox(
        "Phone Service",
        [0, 1]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [0, 1]
    )

# ==========================
# PREDICT BUTTON
# ==========================
if st.button("Predict Churn 🚀"):

    # Model expects 30 features
    input_data = np.zeros((1, 30))

    # Fill known values
    input_data[0, 0] = senior_citizen
    input_data[0, 1] = tenure
    input_data[0, 2] = monthly_charges
    input_data[0, 3] = total_charges
    input_data[0, 4] = partner
    input_data[0, 5] = dependents
    input_data[0, 6] = phone_service
    input_data[0, 7] = paperless_billing

    # Scale input
    scaled_features = scaler.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        scaled_features
    )[0]

    st.divider()

    if prediction == 1:
        st.error(
            "⚠ Customer Will Churn"
        )
    else:
        st.success(
            "✅ Customer Will Stay"
        )

# ==========================
# FOOTER
# ==========================
st.markdown(
    '<div class="footer">Built with Streamlit | Customer Churn Prediction</div>',
    unsafe_allow_html=True
)