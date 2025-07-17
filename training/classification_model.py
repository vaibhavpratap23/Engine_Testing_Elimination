import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv("../data/synthetic_engine_data.csv")


# 2. Separate features (X) and classification target (y)
X = df.drop(columns=['power_output', 'fuel_efficiency', 'failure_risk'])
y = df['failure_risk']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Logistic Regression classifier
clf = LogisticRegression(class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)

# 6. Predict
y_pred = clf.predict(X_test_scaled)

# 7. Evaluate the model
print("🧠 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save model and scaler for web app use
joblib.dump(clf, "../models/failure_risk_model.pkl")
joblib.dump(scaler, "../scalers/failure_scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
