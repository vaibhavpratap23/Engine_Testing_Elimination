# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
df = pd.read_csv("synthetic_engine_data.csv")

# 3. Separate input features (X) and outputs (y)
X = df.drop(columns=['power_output', 'fuel_efficiency', 'failure_risk'])

# Targets
y_regression = df[['power_output', 'fuel_efficiency']]  # Continuous variables
y_classification = df['failure_risk']                   # Binary classification

# 4. Train/Test Split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# 5. Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Check shapes
print("✅ Shapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_reg_train:", y_reg_train.shape)
print("y_clf_train:", y_clf_train.shape)
