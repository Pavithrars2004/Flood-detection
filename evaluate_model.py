import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the actual data
actual_data = pd.read_csv("flood_data.csv")
print("Columns in actual data:", actual_data.columns)

# Load the predicted data
predictions = pd.read_csv("predictions.csv")
print("Columns in predictions:", predictions.columns)

# Merge datasets on 'Year' column
df = pd.merge(actual_data[['Year', 'Total']], predictions, on="Year", how="inner")

print("\nMerged Data Sample:")
print(df.head())

# Models to evaluate
models = ['ANN Prediction', 'Linear Regression', 'Random Forest']

# Store evaluation metrics
evaluation_results = []

# Evaluate each model
for model in models:
    actual_values = df["Total"]
    predicted_values = df[model]

    # Calculate error metrics
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(actual_values, predicted_values)

    # Calculate R² only if we have at least two rows
    if len(df) > 1:
        r2 = r2_score(actual_values, predicted_values)
    else:
        r2 = "N/A"  # Not applicable for single-row evaluation

    print(f"\n✅ {model} Evaluation:")
    print(f"   - MSE: {mse}")
    print(f"   - RMSE: {rmse}")
    print(f"   - MAE: {mae}")
    print(f"   - R² Score: {r2}")

    # Save results
    evaluation_results.append([model, mse, rmse, mae, r2])

# Convert results into a DataFrame
results_df = pd.DataFrame(evaluation_results, columns=['Model', 'MSE', 'RMSE', 'MAE', 'R² Score'])

# Save to CSV
results_df.to_csv("model_evaluation_results.csv", index=False)
print("\n✅ Model evaluation completed! Results saved in 'model_evaluation_results.csv'.")

# 📊 **Visualization: MSE Comparison Bar Chart**
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MSE"], color=['blue', 'green', 'red'])
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Comparison of Different Models")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.savefig("mse_comparison.png")
print("\n✅ MSE comparison chart saved as 'mse_comparison.png'.")
plt.show()

# 📈 **Visualization: Actual vs Predicted Flood Levels (Line Graph)**
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Total"], label="Actual", marker='o', color="black", linestyle="--")

# Only plot predictions if we have multiple data points
if len(df) > 1:
    plt.plot(df["Year"], df["ANN Prediction"], label="ANN Prediction", marker='s', linestyle="-", color="blue")
    plt.plot(df["Year"], df["Linear Regression"], label="Linear Regression", marker='^', linestyle="-", color="green")
    plt.plot(df["Year"], df["Random Forest"], label="Random Forest", marker='D', linestyle="-", color="red")

plt.xlabel("Year")
plt.ylabel("Flood Level")
plt.title("Actual vs Predicted Flood Levels")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("actual_vs_predicted.png")
print("\n✅ Actual vs Predicted flood levels graph saved as 'actual_vs_predicted.png'.")
plt.show()
