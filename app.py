import streamlit as st
import requests
import threading
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

# --------- FLASK API (BACKEND) ----------
app = Flask(__name__)

# Load Model & Scaler
model = tf.keras.models.load_model("model/flood_model.h5")
x_scaler = joblib.load("model/scaler_X.pkl")  # Ensure this file exists

@app.route("/")
def home():
    return "Flood Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        rainfall = float(data["rainfall"])
        water_level = float(data["water_level"])
        
        # Preprocess input
        input_data = np.array([[rainfall, water_level]])
        input_data = x_scaler.transform(input_data)

        # Predict flood occurrence
        prediction = model.predict(input_data)
        flood_risk = int(prediction[0][0] > 0.5)  # Convert probability to 0 or 1

        return jsonify({"rainfall": rainfall, "water_level": water_level, "flood_risk": flood_risk})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask in a separate thread
def run_flask():
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# --------- STREAMLIT UI (FRONTEND) ----------
st.title("🌊 Flood Prediction System")

# User Inputs
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, format="%.2f")
water_level = st.number_input("Enter Water Level (m)", min_value=0.0, format="%.2f")

if st.button("Predict Flood Risk 🚀"):
    api_url = "http://127.0.0.1:5000/predict"
    data = {"rainfall": rainfall, "water_level": water_level}

    try:
        response = requests.post(api_url, json=data)
        result = response.json()

        if "flood_risk" in result:
            flood_risk = "🌊 HIGH Risk (Flood Likely)" if result["flood_risk"] == 1 else "✅ LOW Risk (No Flood)"
            st.success(f"Prediction: {flood_risk}")
        else:
            st.error("Error in prediction. Please check API.")
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")

