import streamlit as st
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Customer Churn Predictor",
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

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #111827
    );
    color: white;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #ec4899
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 20px;
    margin-bottom: 25px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Button */
.stButton>button {
    width: 100%;
    height: 60px;
    border: none;
    border-radius: 15px;
    background: linear-gradient(
        90deg,
        #3b82f6,
        #9333ea
    );
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown(
    '<div class="main-title">📊 Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered prediction to check if a customer will stay or churn</div>',
    unsafe_allow_html=True
)

# ==========================
# INPUT SECTION
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("👤 Customer Details")

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        24
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=2000.0
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📞 Service Details")

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

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ==========================
# PREDICT BUTTON
# ==========================
if st.button("🚀 Predict Customer Churn"):

    # Model expects 30 features
    input_data = np.zeros((1, 30))

    input_data[0, 0] = senior_citizen
    input_data[0, 1] = tenure
    input_data[0, 2] = monthly_charges
    input_data[0, 3] = total_charges
    input_data[0, 4] = partner
    input_data[0, 5] = dependents
    input_data[0, 6] = phone_service
    input_data[0, 7] = paperless_billing

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(
        scaled_data
    )[0]

    try:
        probability = (
            model.predict_proba(
                scaled_data
            )[0][1] * 100
        )
    except:
        probability = 50

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠ Customer Will Churn\n\nRisk Probability: {probability:.2f}%"
        )

        st.progress(
            int(probability)
        )

    else:
        st.success(
            f"✅ Customer Will Stay\n\nConfidence Score: {100-probability:.2f}%"
        )

        st.progress(
            int(100-probability)
        )

# ==========================
# FOOTER
# ==========================
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit | Customer Churn Prediction</div>',
    unsafe_allow_html=True
)