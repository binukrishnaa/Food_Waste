import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🍽️ Food Waste Prediction App")

# Input fields (customize based on your features)
feature1 = st.number_input("Feature 1")
feature2 = st.selectbox("Feature 2", ["Option1", "Option2"])
# Add all features your model needs...

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])  # shape must match training
    prediction = model.predict(input_data)
    st.success(f"Predicted Output: {prediction[0]}")
