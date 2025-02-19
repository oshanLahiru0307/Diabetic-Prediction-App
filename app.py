import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():   
    with open('diabetes_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load the model
data = load_model()
regressor = data['model']  # Extract the trained model

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("### Enter the patient details below:")

# Input fields
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=200, value=80)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
age = st.slider("Age", 1, 100, 30)

ok = st.button("Check the result")

if ok:
    try:
        # Convert inputs into NumPy array (matching model format)
        input_data = np.array([[glucose, blood_pressure, insulin, bmi, age]], dtype=float)

        # Make prediction
        Diabetes = regressor.predict(input_data)

        # Display result
        if Diabetes[0] == 1:
            st.error("🚨 High Risk of Diabetes! Consult a doctor immediately.")
        else:
            st.success("✅ Low Risk of Diabetes! Keep maintaining a healthy lifestyle.")

    except Exception as e:
        st.error(f"Error: {e}")
