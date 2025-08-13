import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np

# ======== PAGE CONFIG ========
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ======== CUSTOM CSS ========
st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #ffe6f0 0%, #e6f7ff 100%);
}
.block-container { padding: 2rem 2rem; animation: fadeIn 1s ease-in; }

/* Header */
.header-wrap {
  display: flex; align-items: center; gap: 14px;
  margin: 0 0 18px 2px;
}
.heart {
  width: 26px; height: 26px;
  display: inline-block;
  transform-origin: center;
  animation: heartbeat 1.2s infinite ease-in-out;
  filter: drop-shadow(0 2px 6px rgba(255,71,126,0.35));
}
.header-title {
  font-size: 1.6rem; font-weight: 800; letter-spacing: .2px;
  background: linear-gradient(90deg, #FF477E 0%, #9d4edd 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* Labels in Patient Details */
label, .stRadio label, .stSelectbox label {
  font-weight: 700; 
  color: #FFFFFF !important; /* white for better visibility */
}

/* Buttons */
.stButton button {
  background: linear-gradient(90deg, #FF477E, #9d4edd);
  color: #fff; font-weight: 700; border-radius: 10px; padding: 10px 24px; border: none;
}
.stButton button:hover { background: linear-gradient(90deg, #9d4edd, #FF477E); }

/* Animations */
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
@keyframes slideUp { from {transform: translateY(30px); opacity: 0;} to {transform: translateY(0); opacity: 1;} }
@keyframes heartbeat {
  0%   { transform: scale(1); }
  14%  { transform: scale(1.18); }
  28%  { transform: scale(1); }
  42%  { transform: scale(1.18); }
  70%  { transform: scale(1); }
  100% { transform: scale(1); }
}
@keyframes pulseGlow {
  0% { filter: drop-shadow(0 0 0 rgba(255,71,126,0.7)); }
  50% { filter: drop-shadow(0 0 15px rgba(255,71,126,0.9)); }
  100% { filter: drop-shadow(0 0 0 rgba(255,71,126,0.7)); }
}
.chart-container {
  animation: slideUp 0.8s ease-out, fadeIn 1.2s ease-in;
}
.gauge-pulse { animation: pulseGlow 1.5s infinite; }

/* Stat Cards */
.stat-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  text-align: center;
}
.stat-number {
  font-size: 2rem;
  font-weight: 800;
  color: #FF477E;
}
.stat-label {
  font-size: 0.9rem;
  color: #555;
}
</style>
""", unsafe_allow_html=True)

# ======== LOAD MODEL & DATA ========
MODEL_DIR = "models"
model = joblib.load(f"{MODEL_DIR}/model.joblib")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_columns.joblib")
df = pd.read_csv(r"C:\Users\rejir\Downloads\heart disease\heart.csv")

# ======== HEADER ========
st.markdown("""
<div class="header-wrap">
  <img src="https://cdn-icons-png.flaticon.com/512/833/833472.png" class="heart">
  <div class="header-title">Heart Disease Prediction Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ======== TABS ========
tab1, tab2 = st.tabs(["💓 Prediction", "📊 Data Insights"])

# ======== TAB 1: PREDICTION ========
with tab1:
    st.subheader("Patient Details")
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.radio("Sex", ["Male", "Female"], key="sex_radio")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], key="cp_sel")
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], key="fbs_radio")
            restecg = st.selectbox("Resting ECG", [0, 1, 2], key="restecg_sel")
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        with col3:
            exang = st.radio("Exercise Induced Angina", ["Yes", "No"], key="exang_radio")
            oldpeak = st.number_input("Oldpeak", 0.0, 6.5, 1.0, step=0.1)
            slope = st.selectbox("ST Slope", [0, 1, 2], key="slope_sel")

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        # Prepare input
        input_dict = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "chest pain type": cp,
            "resting bp s": trestbps,
            "cholesterol": chol,
            "fasting blood sugar": 1 if fbs == "Yes" else 0,
            "resting ecg": restecg,
            "max heart rate": thalach,
            "exercise angina": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "ST slope": slope
        }

        user_df = pd.DataFrame([input_dict])
        user_df = pd.get_dummies(user_df)
        for col in feature_cols:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[feature_cols]

        scaled_input = scaler.transform(user_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0, 1]

        st.subheader("Prediction Result")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Percentage"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prediction == 1 else "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "red"}]
                  }
        ))
        st.plotly_chart(gauge, use_container_width=True)
        st.success("High Risk" if prediction == 1 else "Low Risk")

# ======== TAB 2: DATA INSIGHTS ========
with tab2:
    st.subheader("Dataset Overview")

    # Animated counter function
    def animated_counter(label, value, suffix=""):
        placeholder = st.empty()
        for i in range(0, value + 1, max(1, value // 50)):
            placeholder.markdown(f"<div class='stat-card'><div class='stat-number'>{i}{suffix}</div><div class='stat-label'>{label}</div></div>", unsafe_allow_html=True)
            time.sleep(0.02)

    col1, col2, col3 = st.columns(3)
    with col1:
        animated_counter("Average Age", int(df["age"].mean()))
    with col2:
        animated_counter("High Risk Patients", int(round((df["target"].sum() / len(df)) * 100)), "%")
    with col3:
        animated_counter("Avg Max Heart Rate", int(df["max heart rate"].mean()))

    st.write("### Scatter Plot: Max Heart Rate vs Oldpeak")
    st.plotly_chart(px.scatter(df, x="max heart rate", y="oldpeak", color="target",
                               title="Max Heart Rate vs Oldpeak"), use_container_width=True)

    st.write("### Age Distribution")
    st.plotly_chart(px.histogram(df, x="age", nbins=20, color="target",
                                 title="Age Distribution by Target"), use_container_width=True)
