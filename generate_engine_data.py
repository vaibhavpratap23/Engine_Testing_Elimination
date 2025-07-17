import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 2000

# Generate input features
engine_rpm = np.random.normal(loc=2000, scale=500, size=n_samples).clip(800, 6000)
fuel_flow_rate = np.random.normal(loc=50, scale=10, size=n_samples).clip(20, 100)
intake_air_temp = np.random.normal(loc=30, scale=5, size=n_samples).clip(10, 60)
coolant_temp = np.random.normal(loc=90, scale=10, size=n_samples).clip(60, 120)
manifold_pressure = np.random.normal(loc=100, scale=20, size=n_samples).clip(60, 200)
ambient_temp = np.random.normal(loc=25, scale=7, size=n_samples).clip(-10, 45)
engine_load = np.random.uniform(low=10, high=100, size=n_samples)

# Generate outputs based on feature interactions
power_output = (engine_rpm * fuel_flow_rate * engine_load) / 100000 + np.random.normal(0, 5, n_samples)
exhaust_temp = 0.6 * coolant_temp + 0.4 * intake_air_temp + np.random.normal(0, 5, n_samples)
fuel_efficiency = 5000 / (engine_rpm + 1e-5) * (100 / fuel_flow_rate) + np.random.normal(0, 1, n_samples)

# Failure risk: binary classification (1 = failure likely)
failure_risk = (
    (coolant_temp > 110) |
    (exhaust_temp > 600) |
    (fuel_flow_rate > 90) |
    (engine_rpm > 5500)
).astype(int)

# Combine into DataFrame
df = pd.DataFrame({
    'engine_rpm': engine_rpm,
    'fuel_flow_rate': fuel_flow_rate,
    'intake_air_temp': intake_air_temp,
    'coolant_temp': coolant_temp,
    'manifold_pressure': manifold_pressure,
    'ambient_temp': ambient_temp,
    'engine_load': engine_load,
    'power_output': power_output,
    'exhaust_temp': exhaust_temp,
    'fuel_efficiency': fuel_efficiency,
    'failure_risk': failure_risk
})

# Save to CSV
df.to_csv("synthetic_engine_data.csv", index=False)
print("✅ Synthetic engine dataset generated and saved as 'synthetic_engine_data.csv'")