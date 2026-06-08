import streamlit as st
import joblib
import numpy as np

# ==========================
# 1. PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ==========================
# 2. LOAD MODEL & SCALER (Safe Loading)
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
# 3. UNIVERSAL STABLE CSS (Har System Ke Liye Fixed)
# ==========================
st.markdown("""
<style>
/* Global Font Integration - Direct from Google Server */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Premium Dark Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #111827 100%) !important;
    color: #f3f4f6 !important;
}

/* ---- HEADER FIXED SECTION (No Blue Box, No Text Cut) ---- */
.header-container {
    text-align: center;
    padding: 25px 10px;
    margin-bottom: 30px;
    width: 100%;
    display: block;
    clear: both;
    background: transparent !important;
}

.main-title {
    font-size: clamp(28px, 4.5vw, 48px) !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    margin: 0px auto 12px auto !important;
    padding: 0 !important;
    
    /* Cross-browser Gradient Text Fix */
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #ec4899 100%) !important;
    background-clip: text !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    
    display: inline-block !important;
}

.subtitle {
    color: #9ca3af !important;
    font-size: clamp(14px, 2vw, 17px) !important;
    font-weight: 400 !important;
    margin-top: 5px !important;
    margin-bottom: 0px !important;
    line-height: 1.5 !important;
    background: transparent !important;
}
/* -------------------------------------------------------- */

/* Streamlit Native Columns Hijack (Symmetrical Glassmorphism Cards) */
div[data-testid="column"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 28px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 20px;
}

/* High Contrast Input Labels */
div[data-testid="stWidgetFormLabel"] p, label p, .stSlider p {
    color: #e5e7eb !important;
    font-weight: 500 !important;
    font-size: 15px !important;
}

/* Custom Dropdown/Selectbox Wrapper for Dark Theme */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* Number Input Box Fix */
.stNumberInput input {
    background-color: rgba(15, 23, 42, 0.7) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* Centered Glowing Predict Button */
.stButton > button {
    width: 100% !important;
    height: 55px !important;
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease-in-out !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
    margin-top: 15px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
    border: none !important;
    color: #ffffff !important;
}

/* Footer Styling */
.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #6b7280;
    font-size: 13px;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 4. FIXED HEADER SECTION
# ==========================
st.markdown("""
<div class="header-container">
    <div class="main-title">📊 Customer Churn Predictor</div>
    <div class="subtitle">AI-powered prediction to check if a customer will stay or churn</div>
</div>
""", unsafe_allow_html=True)

# File Validation Check
if model is None or scaler is None:
    st.error("❌ **'model.pkl'** ya **'scaler.pkl'** aapke current folder mein nahi mili! Pehle files upload karein.")
    st.stop()

# ==========================
# 5. INPUT FIELDS SECTION
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color:#38bdf8; font-size:22px; margin-top:0; font-weight:700; margin-bottom:20px;'>👤 Customer Details</h3>", unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes")
    tenure = st.slider("Tenure (Months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=2000.0)

with col2:
    st.markdown("<h3 style='color:#818cf8; font-size:22px; margin-top:0; font-weight:700; margin-bottom:20px;'>📞 Service Details</h3>", unsafe_allow_html=True)
    partner = st.selectbox("Partner", [0, 1], help="0 = No, 1 = Yes")
    dependents = st.selectbox("Dependents", [0, 1], help="0 = No, 1 = Yes")
    phone_service = st.selectbox("Phone Service", [0, 1], help="0 = No, 1 = Yes")
    paperless_billing = st.selectbox("Paperless Billing", [0, 1], help="0 = No, 1 = Yes")

st.write("")

# ==========================
# 6. ML PREDICTION ENGINE
# ==========================
if st.button("🚀 Predict Customer Churn"):
    
    # Base array for 30 features expected by your model
    input_data = np.zeros((1, 30))
    
    # Mapping inputs to correct index
    input_data[0, 0] = senior_citizen
    input_data[0, 1] = tenure
    input_data[0, 2] = monthly_charges
    input_data[0, 3] = total_charges
    input_data[0, 4] = partner
    input_data[0, 5] = dependents
    input_data[0, 6] = phone_service
    input_data[0, 7] = paperless_billing

    # Scaling and Prediction
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    try:
        probability = model.predict_proba(scaled_data)[0][1] * 100
    except:
        probability = 50.0

    st.divider()

    # Dynamic Alert boxes with high visibility text
    if prediction == 1:
        st.error(f"### ⚠ Customer Will Churn\n\n**Risk Probability:** {probability:.2f}%")
        st.progress(int(probability))
    else:
        st.success(f"### ✅ Customer Will Stay\n\n**Confidence Score:** {100-probability:.2f}%")
        st.progress(int(100-probability))

# ==========================
# 7. FOOTER SECTION
# ==========================
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit | Customer Churn Prediction</div>',
    unsafe_allow_html=True
)
