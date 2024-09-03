import streamlit as st
from joblib import load
import numpy as np

# Load the model
try:
    model = load('best_model.pkl')
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Streamlit app
st.title("Alzheimer's Disease Detection")
st.write("This app predicts the likelihood of Alzheimer's disease based on various health metrics.")

# Input fields
ADL = st.slider("ADL: Level of difficulty in Activities of Daily Living", 0, 4, 0)
FunctionalAssessment = st.slider("Functional Assessment score", 0, 4, 0)
MMSE = st.slider("MMSE: Mini-Mental State Examination score", 0, 3, 0)
BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
MemoryComplaints = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
SleepQuality = st.slider("Sleep Quality", 4, 10, 7)
CholesterolHDL = st.slider("Cholesterol HDL level", 20, 70, 50)

# Prepare the input data
data = np.array([[
    ADL,
    FunctionalAssessment,
    MMSE,
    BehavioralProblems,
    MemoryComplaints,
    SleepQuality,
    CholesterolHDL
]], dtype=float)

# Prediction button
if st.button('Predict'):
    try:
        # Make the prediction
        prediction = model.predict(data)[0]
        
        # Display the prediction result
        if prediction == 1:
            st.error("Prediction: You are likely to have Alzheimer's Disease.")
        else:
            st.success("Prediction: You are not likely to have Alzheimer's Disease.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
