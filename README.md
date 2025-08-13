# Heart-Disease-Predictor
# 💓 Heart Disease Prediction Dashboard

A Streamlit-based machine learning web application that predicts the risk of heart disease based on patient health data, and provides an interactive data insights dashboard.

## 🚀 Features
- **Heart Disease Risk Prediction** using a trained Random Forest model.
- **Interactive Forms** for entering patient details.
- **Real-time Risk Gauge** showing predicted probability.
- **Data Insights Dashboard** with:
  - Max Heart Rate vs Oldpeak scatter plot
  - Age distribution histogram
  - Animated statistics cards
- **Attractive UI** with custom colors, animations, and responsive layout.
- **Footer with Contact Information**.

## 📂 Project Structure
heart-disease-detector/
│
├── app.py # Main Streamlit application
├── heart.csv # Dataset (if under 50MB)
├── models/
│ ├── model.joblib # Trained Random Forest model
│ ├── scaler.joblib # Feature scaler
│ ├── feature_columns.joblib # Saved feature column names
├── requirements.txt # Python dependencies
└── README.md # Project documentation
