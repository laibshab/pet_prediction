import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Pet Food Consumption Predictor")

age = st.number_input("Enter pet's age (years):", min_value=1, max_value=20, value=5)
weight = st.number_input("Enter pet's weight (kg):", min_value=1.0, max_value=100.0, value=10.0)
breed_size = st.selectbox("Select breed size:", ["small", "medium", "large"])
activity_level = st.selectbox("Select activity level:", ["low", "medium", "high"])

# Encode inputs
breed_size_map = {'small': 1, 'medium': 2, 'large': 3}
activity_level_map = {'low': 1, 'medium': 2, 'high': 3}

breed_size_encoded = breed_size_map[breed_size]
activity_level_encoded = activity_level_map[activity_level]

if st.button("Predict"):
    # Make prediction
    input_data = np.array([[age, weight, breed_size_encoded, activity_level_encoded]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Food Consumption: {prediction[0]:.2f} grams/day")