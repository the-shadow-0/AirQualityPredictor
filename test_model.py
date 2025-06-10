import joblib
import numpy as np

# Load model and scalers
model = joblib.load('aqi_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')
y_min = target_scaler['min']
y_max = target_scaler['max']

# Test inputs
tests = {
    "Excellent": [20, 50, 0.2, 0.3, 10, 10, 0],
    "Good": [22, 55, 0.5, 1.0, 25, 12, 0],
    "Moderate": [25, 65, 1.5, 3.0, 60, 16, 0],
    "Unhealthy": [30, 75, 4.0, 8.0, 150, 18, 0],
    "Hazardous": [35, 85, 8.0, 15.0, 300, 20, 0]
}

for category, values in tests.items():
    features = np.array([values])
    features_scaled = feature_scaler.transform(features)
    prediction_scaled = model.predict(features_scaled)[0]
    prediction_actual = prediction_scaled * (y_max - y_min) + y_min
    aqi = min(500, max(0, prediction_actual / 2))    
    print(f"{category} conditions: Predicted AQI = {aqi:.1f}")