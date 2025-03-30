import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Predictions
df = pd.read_csv("flood_data.csv")
predictions = pd.read_csv("future_predictions.csv")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Total"], label="Actual Data", marker="o")
plt.plot(predictions["Year"], predictions["Predicted Total"], label="Predicted Data", linestyle="dashed", marker="x")
plt.xlabel("Year")
plt.ylabel("Flood Prediction")
plt.legend()
plt.title("Flood Prediction Trend")
plt.show()

# Confidence Interval Calculation
pred_values = np.array(predictions["Predicted Total"])
lower_bound = np.percentile(pred_values, 2.5)
upper_bound = np.percentile(pred_values, 97.5)

print(f"🔍 95% Confidence Interval: [{lower_bound}, {upper_bound}]")
