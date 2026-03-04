import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

FEATURES = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    scaled_array = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURES)
    pred = model.predict(scaled_df)[0]
    return pred


def main():
    st.title("Heart Attack Prediction App")
    age = st.slider("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.slider("Cholesterol", min_value=100, max_value=500, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120? (1 = True)", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])
    oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Predict Heart Attack Risk"):
        result = make_prediction(inputs)
        if result == 1:
            st.error("High Risk of Heart Attack")
        else:
            st.success("Low Risk of Heart Attack")

if __name__ == "__main__":
    main()