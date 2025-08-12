import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# ===== Load model and scaler =====
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===== Page Config =====
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# ===== Custom CSS =====
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(120deg, #fceabb, #f8b500);
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #d62828;
            padding: 15px;
        }
        .subtitle {
            text-align: center;
            color: #333;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .prediction-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff512f, #dd2476);
            color: white;
            font-size: 18px;
            border-radius: 8px;
            height: 50px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #dd2476, #ff512f);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown("<div class='title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill in the details below to check your risk level</div>", unsafe_allow_html=True)

# ===== Form =====
with st.form("heart_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<span style='color:black; font-weight:bold'>Age</span>", unsafe_allow_html=True)
        age = st.slider("", 18, 100, 50, key="age_slider")

        st.markdown("<span style='color:black; font-weight:bold'>Sex</span>", unsafe_allow_html=True)
        sex = st.radio("", ["Male", "Female"], key="sex_radio")

        st.markdown("<span style='color:black; font-weight:bold'>Chest Pain Type</span>", unsafe_allow_html=True)
        cp = st.selectbox("", [1, 2, 3, 4], key="cp_select")

        st.markdown("<span style='color:black; font-weight:bold'>Resting BP (mm Hg)</span>", unsafe_allow_html=True)
        trestbps = st.number_input("", 80, 200, 120, key="trestbps_input")

    with col2:
        st.markdown("<span style='color:black; font-weight:bold'>Cholesterol (mg/dl)</span>", unsafe_allow_html=True)
        chol = st.number_input("", 100, 600, 200, key="chol_input")

        st.markdown("<span style='color:black; font-weight:bold'>Fasting Blood Sugar > 120 mg/dl</span>", unsafe_allow_html=True)
        fbs = st.radio("", ["Yes", "No"], key="fbs_radio")

        st.markdown("<span style='color:black; font-weight:bold'>Resting ECG</span>", unsafe_allow_html=True)
        restecg = st.selectbox("", [0, 1, 2], key="restecg_select")

        st.markdown("<span style='color:black; font-weight:bold'>Max Heart Rate</span>", unsafe_allow_html=True)
        thalach = st.number_input("", 60, 220, 150, key="thalach_input")

    with col3:
        st.markdown("<span style='color:black; font-weight:bold'>Exercise Induced Angina</span>", unsafe_allow_html=True)
        exang = st.radio("", ["Yes", "No"], key="exang_radio")

        st.markdown("<span style='color:black; font-weight:bold'>Oldpeak (ST Depression)</span>", unsafe_allow_html=True)
        oldpeak = st.number_input("", 0.0, 10.0, 1.0, step=0.1, key="oldpeak_input")

        st.markdown("<span style='color:black; font-weight:bold'>ST Slope</span>", unsafe_allow_html=True)
        slope = st.selectbox("", [1, 2, 3], key="slope_select")

    # ✅ Add submit button inside the form
    submitted = st.form_submit_button("🔍 Predict")

# ===== Prediction =====
if submitted:
    # Encode categorical inputs
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    # Create input array in correct order
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg,
                            thalach, exang_val, oldpeak, slope]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Display result card
    if prediction == 1:
        st.markdown(f"<div class='prediction-card negative'>🚨 High Risk of Heart Disease<br>Risk Score: {probability:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-card positive'>✅ Low Risk of Heart Disease<br>Risk Score: {probability:.2f}%</div>", unsafe_allow_html=True)

    # ===== Gauge Chart =====
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Risk Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 50 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "#b7e4c7"},
                {'range': [30, 60], 'color': "#ffdd8f"},
                {'range': [60, 100], 'color': "#f4978e"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
