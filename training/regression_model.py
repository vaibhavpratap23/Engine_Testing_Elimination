import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("synthetic_engine_data.csv")

# 2. Separate features and regression targets
X = df.drop(columns=['power_output', 'fuel_efficiency', 'failure_risk'])
y_reg = df[['power_output', 'fuel_efficiency']]

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# 4. Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train models
model_power = LinearRegression()
model_efficiency = LinearRegression()

model_power.fit(X_train_scaled, y_train['power_output'])
model_efficiency.fit(X_train_scaled, y_train['fuel_efficiency'])

# 6. Predictions
y_pred_power = model_power.predict(X_test_scaled)
y_pred_efficiency = model_efficiency.predict(X_test_scaled)

# 7. Evaluation
print("📈 Power Output Prediction")
print("MSE:", mean_squared_error(y_test['power_output'], y_pred_power))
print("R² Score:", r2_score(y_test['power_output'], y_pred_power))

print("\n⛽ Fuel Efficiency Prediction")
print("MSE:", mean_squared_error(y_test['fuel_efficiency'], y_pred_efficiency))
print("R² Score:", r2_score(y_test['fuel_efficiency'], y_pred_efficiency))
