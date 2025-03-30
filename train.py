import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Data
df = pd.read_csv("flood_data.csv")
X = df[["Year"]].values
y = df["Total"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse')
ann_model.fit(X_train, y_train, epochs=200, verbose=0)
ann_model.save("model/ann_model.h5")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, "model/lr_model.pkl")

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "model/rf_model.pkl")

# LSTM Model
X_train_lstm = X_train.reshape((X_train.shape[0], 1, 1))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(1, 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=200, verbose=0)
lstm_model.save("model/lstm_model.h5")

print("✅ All models trained and saved!")
