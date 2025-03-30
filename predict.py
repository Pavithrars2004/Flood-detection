import numpy as np
import tensorflow as tf
import joblib
import pandas as pd


# Load Models
ann_model = tf.keras.models.load_model("model/ann_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

lr_model = joblib.load("model/lr_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")

# User Input
years = input("Enter the years to predict (comma-separated): ")
years = np.array([int(y) for y in years.split(",")]).reshape(-1, 1)

# Predictions
predictions = {
    "Year": years.flatten(),
    "ANN Prediction": ann_model.predict(years).flatten(),
    "Linear Regression": lr_model.predict(years).flatten(),
    "Random Forest": rf_model.predict(years).flatten()
}

# Save Predictions
output_df = pd.DataFrame(predictions)
output_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved in 'predictions.csv'.")
