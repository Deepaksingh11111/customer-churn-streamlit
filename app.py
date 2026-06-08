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
# 2. LOAD MODEL & SCALER
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
# 3. FUTURISTIC GLASSMISM CSS
# ==========================
st.markdown("""
<style>
/* Global Font Integration */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Deep Space Nebula Background mimicking the image */
.stApp {
    background: radial-gradient(circle at 20% 30%, #2e1065 0%, #0f172a 50%, #020617 100%) !important;
    background-attachment: fixed !important;
    color: #f8fafc !important;
}

/* ---- HEADER SECTION ---- */
.header-container {
    text-align: center;
    padding: 30px 10px;
    margin-bottom: 20px;
    width: 100%;
}

.main-title {
    font-size: clamp(30px, 5vw, 52px) !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    margin-bottom: 8px !important;
    
    /* Neon Pink & Cyan Gradient Text */
    background: linear-gradient(90deg, #38bdf8 0%, #c084fc 50%, #f472b6 100%) !important;
    background-clip: text !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    display: inline-block !important;
}

.subtitle {
    color: #94a3b8 !important;
    font-size: clamp(14px, 2vw, 18px) !important;
    font-weight: 400;
}

/* ---- FUTURISTIC GLASS UI CONTAINER HACK ---- */
div[data-testid="column"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 35px !important;
    backdrop-filter: blur(25px) ipsl-effect !important;
    -webkit-backdrop-filter: blur(25px) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 
                inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
    margin-bottom: 25px;
}

/* Form Headings */
.section-heading-1 {
    color: #38bdf8 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 25px !important;
    text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
}

.section-heading-2 {
    color: #c084fc !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 25px !important;
    text-shadow: 0 0 15px rgba(192, 132, 252, 0.4);
}

/* High Contrast Readable Input Labels */
div[data-testid="stWidgetFormLabel"] p, label p, .stSlider p {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px;
}

/* Glass Inputs (Selectbox & Number inputs) */
.stSelectbox div[data-baseweb="select"], .stNumberInput input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px) !important;
}

.stSelectbox div[data-baseweb="select"]:hover, .stNumberInput input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
}

/* Neon Glowing Active Blue Button mimicking 'Upgrade Pro' */
.stButton > button {
    width: 100% !important;
    height: 58px !important;
    background: linear-gradient(90deg, #1d4ed8 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    margin-top: 20px !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    background: linear-gradient(90deg, #2563eb 0%, #8b5cf6 100%) !important;
    box-shadow: 0 12px 30px rgba(139, 92, 246, 0.6),
                0 0 15px rgba(56, 189, 248, 0.4) !important;
    color: #ffffff !important;
}

/* Alerts Overwrite for Futuristic Feel */
div[data-testid="stNotification"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(15px) !important;
}

/* Footer Styling */
.footer {
    text-align: center;
    margin-top: 80px;
    padding-top: 25px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 4. FIXED HEADER SECTION
# ==========================
st.markdown("""
<div class="header-container">
    <div class="main-title">Futuristic Glass UI Churn Predictor</div>
    <div class="subtitle">Next-gen analytical engine predicting real-time user retention</div>
</div>
""", unsafe_allow_html=True)

# File Validation Check
if model is None or scaler is None:
    st.error("❌ **'model.pkl'** or **'scaler.pkl'** missing in root folder! Please add your assets.")
    st.stop()

# ==========================
# 5. INPUT FIELDS SECTION
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-heading-1">👤 Customer Profile</div>', unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen Status", [0, 1], help="0 = No, 1 = Yes")
    tenure = st.slider("Account Tenure (Months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges Billing ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Lifetime Accumulation ($)", min_value=0.0, value=2000.0)

with col2:
    st.markdown('<div class="section-heading-2">📊 Analytics & Network</div>', unsafe_allow_html=True)
    partner = st.selectbox("Has Registered Partner", [0, 1], help="0 = No, 1 = Yes")
    dependents = st.selectbox("Has Dependents", [0, 1], help="0 = No, 1 = Yes")
    phone_service = st.selectbox("Activated Phone Line Service", [0, 1], help="0 = No, 1 = Yes")
    paperless_billing = st.selectbox("Digital Paperless Billing", [0, 1], help="0 = No, 1 = Yes")

st.write("")

# ==========================
# 6. ML PREDICTION ENGINE
# ==========================
if st.button("⚡ Execute Prediction Metrics"):
    
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

    # Highly-styled glass alerts matching UI
    if prediction == 1:
        st.error(f"### ⚠ Alert: High Risk Churn Trend Identified\n\n**Risk Probability Index:** {probability:.2f}%")
        st.progress(int(probability))
    else:
        st.success(f"### ✅ Positive Stability: Account Will Stay Active\n\n**Confidence Index Score:** {100-probability:.2f}%")
        st.progress(int(100-probability))

# ==========================
# 7. FOOTER SECTION
# ==========================
st.markdown(
    '<div class="footer">Engineered with Streamlit UI Engine | Core Analytics Matrix</div>',
    unsafe_allow_html=True
)
