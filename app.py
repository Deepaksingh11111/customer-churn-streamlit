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
# Make sure model.pkl and scaler.pkl are in the same directory
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================
# CUSTOM CSS (Fonts & Universal Colors Fixed)
# ==========================
st.markdown("""
<style>
/* Import Poppins font directly from Google Fonts for global availability */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Apply font to everything */
html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #111827
    );
    color: #f3f4f6 !important;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #ec4899
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    font-family: 'Poppins', sans-serif;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 35px;
    font-weight: 300;
}

/* Glass Card Container */
.card-box {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* Fix Streamlit Input Labels for Other Systems */
.stWidgetFormLabel, label, p {
    color: #e5e7eb !important;
    font-weight: 400 !important;
}

/* Custom Button Styles */
.stButton>button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(
        90deg,
        #3b82f6,
        #9333ea
    ) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(147, 51, 234, 0.5);
    color: white !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    color: #6b7280;
    font-size: 13px;
    letter-spacing: 0.5px;
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
    # Creating a proper wrapper container using st.container
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#38bdf8; margin-top:0;'>👤 Customer Details</h3>", unsafe_allow_html=True)

        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        tenure = st.slider("Tenure (Months)", 0, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=2000.0)
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#818cf8; margin-top:0;'>📞 Service Details</h3>", unsafe_allow_html=True)

        partner = st.selectbox("Partner", [0, 1])
        dependents = st.selectbox("Dependents", [0, 1])
        phone_service = st.selectbox("Phone Service", [0, 1])
        paperless_billing = st.selectbox("Paperless Billing", [0, 1])
        
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
    prediction = model.predict(scaled_data)[0]

    try:
        probability = model.predict_proba(scaled_data)[0][1] * 100
    except:
        probability = 50.0

    st.divider()

    if prediction == 1:
        st.error(f"⚠ **Customer Will Churn**\n\nRisk Probability: {probability:.2f}%")
        st.progress(int(probability))
    else:
        st.success(f"✅ **Customer Will Stay**\n\nConfidence Score: {100-probability:.2f}%")
        st.progress(int(100-probability))

# ==========================
# FOOTER
# ==========================
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit | Customer Churn Prediction</div>',
    unsafe_allow_html=True
)
