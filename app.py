# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model & scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Heart Health Monitor", layout="centered")

st.title("❤️ Heart Health Monitor")
st.markdown("AI-based Heart Disease Prediction — model ~92% test accuracy (Random Forest)")

st.header("Patient details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=52)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[1, 0])
    cp = st.selectbox("Chest Pain Type (0–3)", options=[0,1,2,3])
    trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=1000, value=240)

with col2:
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", options=[0,1])
    restecg = st.selectbox("Resting ECG results (0–2)", options=[0,1,2])
    thalach = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=160)
    exang = st.selectbox("Exercise induced angina (1 = yes; 0 = no)", options=[0,1])
    oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of peak exercise ST segment (0–2)", options=[0,1,2])
ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", options=[0,1,2,3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible)", options=[1,2,3])

if st.button("Predict"):
    # build input in same column order as training data
    input_arr = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_arr)
    pred = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][pred]

    if pred == 1:
        st.error(f"⚠️ HIGH RISK — Model predicts heart disease (probability {pred_proba:.2f})")
    else:
        st.success(f"✅ LOW RISK — Model predicts NO heart disease (probability {pred_proba:.2f})")

st.markdown("---")
st.markdown("**Tip:** Use realistic medical ranges (e.g., BP ~ 90–180, chol ~ 120–600, oldpeak ~ 0–6).")
