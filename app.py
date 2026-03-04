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
    st.write("Masukkan nilai fitur di bawah")
    inputs = []
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    inputs.append(age)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0,1])
    inputs.append(sex)
    cp = st.number_input("Chest Pain Type (cp)", 0,3,1)
    inputs.append(cp)
    trestbps = st.number_input("Resting Blood Pressure", 0,300,120)
    inputs.append(trestbps)
    chol = st.number_input("Cholesterol", 0,600,200)
    inputs.append(chol)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", [0,1])
    inputs.append(fbs)
    restecg = st.number_input("Rest ECG", 0,2,1)
    inputs.append(restecg)
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