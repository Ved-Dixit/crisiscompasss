import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "earthquake_data.csv"  # Ensure the correct path
df_earthquake = pd.read_csv(file_path)

# Prepare features and target variable
X = df_earthquake[['Age of Building (Years)', 'Floors', 'Earthquake Magnitude']]
y = df_earthquake['Earthquake Damage Index (0-100)']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "earthquake_damage_model.pkl")

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model saved successfully as 'earthquake_damage_model.pkl'")
print(f"Performance: MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.4f}")
