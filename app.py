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
# LOAD MODEL & SCALER
# ==========================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# ==========================
# UNIVERSAL USER-FRIENDLY CSS
# ==========================
st.markdown("""
<style>
/* 1. Global Font Integration (Har system par chalega) */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* 2. Premium Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #152238 50%, #0b1120 100%) !important;
    color: #f3f4f6 !important;
}

/* 3. Text & Headers Styling */
.main-title {
    text-align: center;
    font-size: clamp(32px, 5vw, 52px); /* Responsive Font Size */
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: clamp(14px, 2vw, 18px);
    margin-bottom: 40px;
    font-weight: 400;
}

/* 4. Streamlit Widget Overrides (Force High Contrast Colors) */
div[data-testid="stWidgetFormLabel"] p, label p {
    color: #e5e7eb !important;
    font-weight: 500 !important;
    font-size: 15px !important;
}

/* Custom Card Wrapper using Streamlit's native blocks */
div[data-testid="column"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    margin-bottom: 20px;
}

/* 5. High Contrast Input Fields Fixing */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

.stNumberInput input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* 6. Perfectly Centered & Styled Button */
.stButton > button {
    width: 100% !important;
    height: 54px !important;
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease-in-out !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
    border: none !important;
}

/* 7. Footer Styling */
.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #6b7280;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown('<div class="main-title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered prediction to check if a customer will stay or churn</div>', unsafe_allow_html=True)

# Error handling if model is missing
if model is None or scaler is None:
    st.error("❌ 'model.pkl' ya 'scaler.pkl' file nahi mili! Please files ko same folder mein rakhein.")
    st.stop()

# ==========================
# INPUT SECTION (Responsive Columns)
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color:#38bdf8; font-size:22px; margin-top:0;'>👤 Customer Details</h3>", unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes")
    tenure = st.slider("Tenure (Months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=2000.0)

with col2:
    st.markdown("<h3 style='color:#818cf8; font-size:22px; margin-top:0;'>📞 Service Details</h3>", unsafe_allow_html=True)
    partner = st.selectbox("Partner", [0, 1], help="0 = No, 1 = Yes")
    dependents = st.selectbox("Dependents", [0, 1], help="0 = No, 1 = Yes")
    phone_service = st.selectbox("Phone Service", [0, 1], help="0 = No, 1 = Yes")
    paperless_billing = st.selectbox("Paperless Billing", [0, 1], help="0 = No, 1 = Yes")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# PREDICT BUTTON & OUTPUT
# ==========================
if st.button("🚀 Predict Customer Churn"):
    
    # Preprocessing
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
    prediction = model.predict(scaled_data)[0]

    try:
        probability = model.predict_proba(scaled_data)[0][1] * 100
    except:
        probability = 50.0

    st.divider()

    # Dynamic Alert boxes according to prediction
    if prediction == 1:
        st.error(f"### ⚠ Customer Will Churn\n\n**Risk Probability:** {probability:.2f}%")
        st.progress(int(probability))
    else:
        st.success(f"### ✅ Customer Will Stay\n\n**Confidence Score:** {100-probability:.2f}%")
        st.progress(int(100-probability))

# ==========================
# FOOTER
# ==========================
st.markdown('<div class="footer">Built with ❤️ using Streamlit | Customer Churn Prediction</div>', unsafe_allow_html=True)
