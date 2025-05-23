import streamlit as st
import numpy as np
import pickle

# Load trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ model.pkl not found. Please upload the model file.")
    st.stop()

st.title("🍽️ Food Waste Economic Loss Predictor")
st.markdown("""
This app predicts **Economic Loss (in Million $)** based on:
- Total Waste (Tons)
- Avg Waste per Capita (Kg)
- Population (Million)
- Household Waste (%)
""")

# Input features
total_waste = st.number_input("Total Waste (Tons)", min_value=0.0, step=100.0)
avg_waste_per_capita = st.number_input("Avg Waste per Capita (Kg)", min_value=0.0, step=1.0)
population = st.number_input("Population (Million)", min_value=0.0, step=0.1)
household_waste_pct = st.slider("Household Waste (%)", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Predict Economic Loss"):
    input_features = np.array([[total_waste, avg_waste_per_capita, population, household_waste_pct]])
    try:
        prediction = model.predict(input_features)
        st.success(f"Estimated Economic Loss: ${prediction[0]:,.2f} Million")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
