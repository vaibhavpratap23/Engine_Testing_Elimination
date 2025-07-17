import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("synthetic_engine_data.csv")

# 2. Feature selection
X = df.drop(columns=['power_output', 'fuel_efficiency', 'failure_risk'])
y = df['fuel_efficiency']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Predict
y_pred = rf_model.predict(X_test_scaled)

# 7. Evaluate
print("🌲 Random Forest Regressor - Fuel Efficiency")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8. Save model and scaler
import joblib
joblib.dump(rf_model, "fuel_efficiency_model.pkl")
joblib.dump(scaler, "fuel_scaler.pkl")

print("\n✅ Model and scaler saved successfully!")

