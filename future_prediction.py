import numpy as np
import tensorflow as tf

# Load Model
model = tf.keras.models.load_model("model/ann_model.h5")

# Future Years (Next 10 Years)
future_years = np.array([y for y in range(2025, 2035)]).reshape(-1, 1)
predictions = model.predict(future_years)

# Save Future Predictions
output_df = pd.DataFrame({"Year": future_years.flatten(), "Predicted Total": predictions.flatten()})
output_df.to_csv("future_predictions.csv", index=False)
print("✅ Future predictions saved in 'future_predictions.csv'.")
