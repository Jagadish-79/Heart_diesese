# ❤️ Heart Health Monitor – Machine Learning Web App

A machine learning–powered web application that predicts the likelihood of heart disease based on clinical parameters.  
Built using **Python, Scikit-Learn, Random Forest (92% accuracy), Streamlit UI**, and deployed online for real-time predictions.


Project Overview

Heart disease is one of the leading causes of death worldwide.  
This project provides a fast, easy-to-use tool for predicting heart disease using medical attributes such as:

- Age  
- Chest pain type  
- Blood pressure  
- Cholesterol  
- Maximum heart rate  
- Exercise-induced angina  
- ST depression (oldpeak)  
- Thalassemia, etc.

The project includes:

✔ Machine Learning model training  
✔ Data preprocessing & feature scaling  
✔ High-accuracy Random Forest model (~92% test accuracy)  
✔ Exporting the model (`heart_model.pkl`) and scaler (`scaler.pkl`)  
✔ Streamlit web interface  
✔ Full deployment on Streamlit Cloud  


Machine Learning Model

Algorithm Used:Random Forest Classifier  
Reason:Outperformed Logistic Regression, SVM, and KNN on this dataset  
Training Accuracy:** ~98%  
Testing Accuracy:** ~92%  
Dataset: Heart Disease UCI-style dataset (`heart_disease_data.csv`)  

Model Features:
The model was trained on the following attributes:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal


heart-health-monitor/

1)app.py # Streamlit web app

2)train_model.py # ML training script

3)requirements.txt # Packages for Streamlit Cloud

4)heart_disease_data.csv # Dataset

5)heart_model.pkl # Saved trained model

6)scaler.pkl # Saved feature scaler

7)README.md # Project documentation


Deployment (Streamlit Cloud)

This app is deployed using Streamlit Cloud:

1.Push all files to GitHub

2.Go to https://share.streamlit.io

3.Select your repo

4.Choose app.py

5.Deploy 🚀
