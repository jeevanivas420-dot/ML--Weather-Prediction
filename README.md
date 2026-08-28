# Implementation of Random Forest Algorithm for Weather Prediction
## NAME : JEEVA NIVAS M
## REGISTER NUMBER : 212225040148

## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset

 

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1) Load the weather dataset using pandas.
2) Preprocess the data by handling missing values and sorting by time.
3) Select features and create lag variables for temperature and PM2.5.
4) Train Random Forest models to predict temperature and PM2.5 and save the models.
```

## Program:
```
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: MANOJ KUMAR N
Register Number: 212225230168
```
```py
# Implementation of Random Forest Algorithm for Weather Prediction

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("weather-station-eee-block_2024_07_13 (1).csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Shape:", data.shape)
print("\nDataset Columns:")
print(data.columns)

# Select input features
features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr"
]

# Target variable
target = "tem"

# Create X and y
X = data[features]
y = data[target]

# Remove rows only if selected features or target contain missing values
combined = pd.concat([X, y], axis=1)
combined = combined.dropna()

X = combined[features]
y = combined[target]

print("\nRows after removing missing values:", len(X))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display results
print("\n======================================")
print(" RANDOM FOREST WEATHER PREDICTION")
print("======================================")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# Predict temperature for a new weather condition
new_weather = pd.DataFrame([{
    "hum": 80,
    "co2": 500,
    "illumination": 30000,
    "pressure": 1010,
    "pm2_5": 5,
    "pm10": 10,
    "wind_direction_angle": 180,
    "wind_speed": 2,
    "wind_speed_level": 1,
    "tsr": 200
}])

predicted_temperature = model.predict(new_weather)

print("\nPredicted Temperature:",
      round(predicted_temperature[0], 2), "°C")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
importance.plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.xlabel("Weather Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")

plt.tight_layout()
plt.show()
```

## Output:

<img width="904" height="342" alt="image" src="https://github.com/user-attachments/assets/db3bd152-10fc-4f20-a24b-34f2f71caa6f" />
<img width="940" height="386" alt="image" src="https://github.com/user-attachments/assets/6c3253fc-66fb-4e76-b152-b7f24f52047c" />
<img width="650" height="289" alt="image" src="https://github.com/user-attachments/assets/97d7d6b4-6602-45fa-8fed-fc4f1056c2d3" />

<img width="495" height="302" alt="image" src="https://github.com/user-attachments/assets/32355318-0bcc-41cd-b75e-0d2846dd57ad" />

<img width="983" height="590" alt="image" src="https://github.com/user-attachments/assets/998ef0f0-d18f-4437-87bd-03165021b366" />
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/6e85c946-aaed-4fef-af65-b59ffc3418b1" />

## Result:
The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
