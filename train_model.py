# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load and preprocess data
X, y = preprocess_data('AirQualityUCI.csv')

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'feature_scaler.pkl')

# Scale target
y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)
joblib.dump({'min': y_min, 'max': y_max}, 'target_scaler.pkl')

# Train-test split of course
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Train model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, test_pred):.4f}")

# Save model
joblib.dump(model, 'aqi_model.pkl')
print("Model saved successfully!")

# Test with good conditions
good_conditions = np.array([[20, 50, 0.5, 1.0, 30, 10, 0]])
good_scaled = scaler.transform(good_conditions)
pred_scaled = model.predict(good_scaled)
pred_actual = pred_scaled[0] * (y_max - y_min) + y_min
print(f"Good conditions prediction: {pred_actual:.2f}")