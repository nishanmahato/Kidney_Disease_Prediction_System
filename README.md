## Kidney Disease Risk Prediction System
A web-based application built with Streamlit that uses a machine learning model to predict the risk of chronic kidney disease (CKD) based on patient clinical and lifestyle data.
##### Overview
This project provides an interactive interface for early assessment of CKD risk. Users enter patient health parameters and receive risk classification, probability, confidence level, and visual probability distribution.
##### Features
•	Streamlit-based interactive UI
•	Random Forest machine learning model
•	Automatic feature scaling and encoding
•	Risk probability dashboard
•	Visual analytics using Altair
•	Full-screen responsive layout
##### Prediction Workflow
1.	User inputs patient data
2.	Features aligned with trained model schema
3.	Data scaled using pre-trained scaler
4.	Model generates prediction and probability
5.	Results displayed with charts and summary cards
##### Input Parameters
###### Clinical Measurements:
Age, 
Blood Pressure, 
Specific Gravity, 
Albumin, 
Sugar, 
Random Blood Glucose,
Blood Urea, 
Serum Creatinine, 
Sodium, Potassium, 
Hemoglobin, 
Packed Cell Volume

###### Medical History & Lifestyle:
Hypertension, 
Diabetes Mellitus, 
Coronary Artery Disease, 
Pedal Edema,
Anemia, 
Family History of CKD, 
Appetite, 
Smoking Status,
Physical Activity, 
Urinary Sediment

##### Tech Stack
Python, Streamlit, Scikit-learn, Pandas, NumPy, Altair, Joblib



<img width="432" height="636" alt="image" src="https://github.com/user-attachments/assets/2d716e41-8c1b-456d-9679-bcb6a93b495f" />
