import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_diabetes

# ----------------------------
# Conversion Constants
# ----------------------------
_raw   = load_diabetes(scaled=False)
_n     = _raw.data.shape[0]
MEANS  = _raw.data.mean(axis=0)
SCALES = _raw.data.std(axis=0) * np.sqrt(_n)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Header
# ----------------------------
st.title("🩺 Diabetes Disease Progression Predictor")
st.markdown("Predict diabetes progression score using Decision Tree Regression.")
st.markdown("---")

# ----------------------------
# Sidebar — Real Values
# ----------------------------
st.sidebar.header("Patient Features")

age = st.sidebar.slider("Age (years)",              19,   79,   48)
sex = st.sidebar.selectbox("Sex",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female")
bmi = st.sidebar.slider("BMI",                      18.0, 42.0, 26.0, 0.1)
bp  = st.sidebar.slider("Blood Pressure (mmHg)",    62.0,133.0, 94.0, 0.5)
s1  = st.sidebar.slider("S1 - Total Cholesterol",   97.0,301.0,189.0, 1.0)
s2  = st.sidebar.slider("S2 - LDL",                 41.0,242.0,115.0, 1.0)
s3  = st.sidebar.slider("S3 - HDL",                 22.0, 99.0, 50.0, 1.0)
s4  = st.sidebar.slider("S4 - TCH Ratio",            2.0,  9.0,  4.0, 0.1)
s5  = st.sidebar.slider("S5 - Triglycerides (log)",  3.3,  6.1,  4.6, 0.01)
s6  = st.sidebar.slider("S6 - Blood Sugar",          58.0,124.0, 91.0, 1.0)

# ----------------------------
# Conversion Real → Scaled
# ----------------------------
real_values = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6])
scaled      = (real_values - MEANS) / SCALES
features    = scaled.reshape(1, -1)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Progression Score"):
    prediction = model.predict(features)[0]

    st.success(f"Predicted Diabetes Progression Score: **{prediction:.1f}**")

    if prediction < 100:
        st.info("🟢 Low progression score")
    elif prediction < 200:
        st.warning("🟡 Moderate progression score")
    else:
        st.error("🔴 High progression score")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built using Decision Tree Regression · Scikit-learn · Streamlit")
