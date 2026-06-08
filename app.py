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
# 3. LUXURY OBSIDIAN & GOLD MATTE CSS (FULLY ANIMATED)
# ==========================
st.markdown("""
<style>
/* Global Font Integration */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Premium Matte Dark Charcoal/Obsidian Canvas */
.stApp {
    background: #1e1e1e !important;
    background-image: radial-gradient(circle at 10% 20%, rgba(38, 33, 30, 0.7) 0%, rgba(22, 22, 22, 1) 90%) !important;
    background-attachment: fixed !important;
    color: #ffffff !important;
}

/* ---- KEYFRAMES FOR SMOOTH ANIMATIONS ---- */
@keyframes goldGlow {
    0% { border-color: rgba(212, 163, 115, 0.2); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    50% { border-color: rgba(212, 163, 115, 0.4); box-shadow: 0 10px 35px rgba(212, 163, 115, 0.1); }
    100% { border-color: rgba(212, 163, 115, 0.2); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
}

@keyframes pulseButton {
    0% { box-shadow: 0 4px 15px rgba(212, 163, 115, 0.3); }
    50% { box-shadow: 0 4px 25px rgba(212, 163, 115, 0.6); }
    100% { box-shadow: 0 4px 15px rgba(212, 163, 115, 0.3); }
}

/* ---- HEADER SECTION ---- */
.header-container {
    text-align: center;
    padding: 35px 10px;
    margin-bottom: 25px;
    width: 100%;
}

.main-title {
    font-size: clamp(32px, 5vw, 54px) !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    margin-bottom: 10px !important;
    
    /* Luxury Burnished Gold Gradient Text */
    background: linear-gradient(135deg, #f5e3c3 0%, #d4a373 50%, #aa7c4c 100%) !important;
    background-clip: text !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    display: inline-block !important;
}

.subtitle {
    color: #a3a3a3 !important;
    font-size: clamp(14px, 2vw, 18px) !important;
    font-weight: 400;
    letter-spacing: 0.2px;
}

/* ---- MATTE OBSIDIAN CONTAINERS (AMPLIFIED VISIBILITY) ---- */
div[data-testid="column"] {
    background: rgba(30, 30, 30, 0.75) !important;
    border: 1px solid rgba(212, 163, 115, 0.2) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    margin-bottom: 25px;
    
    /* Smooth Continuous Glow Animation */
    animation: goldGlow 6s infinite ease-in-out;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

div[data-testid="column"]:hover {
    transform: translateY(-4px);
    border-color: rgba(212, 163, 115, 0.5) !important;
}

/* Typography Headings */
.section-heading-1, .section-heading-2 {
    color: #e5c39e !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 25px !important;
    letter-spacing: 0.5px;
    border-bottom: 2px solid rgba(212, 163, 115, 0.15);
    padding-bottom: 8px;
}

/* Universally Clear Form Field Labels */
div[data-testid="stWidgetFormLabel"] p, label p, .stSlider p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}

/* Ultra-Visible Text Helper Descriptions under inputs */
div[data-testid="stMarkdownContainer"] p {
    color: #d1d5db !important;
}

/* Custom Styled Input Fields (Selectbox, Sliders, Number Inputs) */
.stSelectbox div[data-baseweb="select"], .stNumberInput input {
    background-color: #121212 !important;
    border: 1px solid rgba(212, 163, 115, 0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    height: 48px !important;
}

.stSelectbox div[data-baseweb="select"]:hover, .stNumberInput input:focus {
    border-color: #f5e3c3 !important;
}

/* Custom Accent Sync for Sliders */
div[data-testid="stSlider"] [data-disabled="false"] {
    background-color: #d4a373 !important;
}

/* ---- PREMIUM BURNISHED GOLD EXECUTE BUTTON ---- */
.stButton > button {
    width: 100% !important;
    height: 60px !important;
    background: linear-gradient(90deg, #b58957 0%, #d4a373 50%, #b58957 100%) !important;
    color: #121212 !important; /* High contrast black font color on gold surface */
    border: none !important;
    border-radius: 14px !important;
    font-size: 19px !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    animation: pulseButton 3s infinite ease-in-out;
    margin-top: 25px !important;
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(90deg, #d4a373 0%, #f5e3c3 100%) !important;
    color: #000000 !important;
    box-shadow: 0 15px 35px rgba(212, 163, 115, 0.6) !important;
}

/* Styled Alert Modules for Output Display */
div[data-testid="stNotification"] {
    background: rgba(22, 22, 22, 0.9) !important;
    border: 1px solid rgba(212, 163, 115, 0.3) !important;
    border-radius: 14px !important;
}

/* Universal Progress Bar Color Overrides */
div[data-testid="stProgress"] div[role="progressbar"] > div {
    background-color: #d4a373 !important;
}

/* Footer Section styling */
.footer {
    text-align: center;
    margin-top: 90px;
    padding-top: 25px;
    border-top: 1px solid rgba(212, 163, 115, 0.1);
    color: #8c8c8c;
    font-size: 13px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 4. FIXED HEADER SECTION
# ==========================
st.markdown("""
<div class="header-container">
    <div class="main-title"> Customer Churn Predictor </div>
    <div class="subtitle">Next-gen analytical engine predicting real-time user retention</div>
</div>
""", unsafe_allow_html=True)

# File Validation Check
if model is None or scaler is None:
    st.error("❌ **'model.pkl'** or **'scaler.pkl'** missing in root folder! Please add your assets.")
    st.stop()

# ==========================
# 5. INPUT FIELDS SECTION WITH FEATURE EXPLANATIONS
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-heading-1">👤 Customer Profile</div>', unsafe_allow_html=True)
    
    # FEATURE: Senior Citizen Status
    senior_citizen = st.selectbox("Senior Citizen Status", [0, 1], help="0 = No, 1 = Yes")
    st.caption("Tells if the user is an older adult. Senior citizens usually seek stability and don't change networks often.")
    
    # FEATURE: Account Tenure
    tenure = st.slider("Account Tenure (Months)", 0, 72, 24)
    st.caption("Tracks relationship length. Long-term customers are loyal; users in their first 1-3 months are the highest risk.")
    
    # FEATURE: Monthly Charges 
    monthly_charges = st.slider("Monthly Charges Billing ($)", 0.0, 200.0, 70.0)
    st.caption("Measures immediate financial load. Higher bills create an immediate motivation to switch to a cheaper competitor.")
    
    # FEATURE: Total Charges
    total_charges = st.number_input("Total Lifetime Accumulation ($)", min_value=0.0, value=2000.0)
    st.caption("Tracks cumulative lifetime financial investment. Acts as an indicator of past customer commitment.")

with col2:
    st.markdown('<div class="section-heading-2">📊 Analytics & Network</div>', unsafe_allow_html=True)
    
    # FEATURE: Partner Status
    partner = st.selectbox("Has Registered Partner", [0, 1], help="0 = No, 1 = Yes")
    st.caption("Indicates family ties. Married/cohabiting individuals show stable behavior profiles and drop services less frequently.")
    
    # FEATURE: Dependents Status
    dependents = st.selectbox("Has Dependents", [0, 1], help="0 = No, 1 = Yes")
    st.caption("Highlights household reliance. Multi-user homes find it operationally disruptive to switch telecom networks.")
    
    # FEATURE: Activated Phone Line Service
    phone_service = st.selectbox("Activated Phone Line Service", [0, 1], help="0 = No, 1 = Yes")
    st.caption("Checks service adoption footprint. Multi-service bundles (Internet + Voice) heavily reduce chances of cancellation.")
    
    # FEATURE: Digital Paperless Billing
    paperless_billing = st.selectbox("Digital Paperless Billing", [0, 1], help="0 = No, 1 = Yes")
    st.caption("Identifies digital engagement level. Tech-savvy users respond quickly to competitors' digital advertisements.")

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
