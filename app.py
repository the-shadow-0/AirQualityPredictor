# app.py
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI(title="AirQuality AI", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and scalers
try:
    model = joblib.load("aqi_model.pkl")
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    y_min = target_scaler['min']
    y_max = target_scaler['max']
    print("✅ Model and scalers loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    temperature: float = Form(...),
    humidity: float = Form(...),
    co: float = Form(...),
    benzene: float = Form(...),
    nox: float = Form(...),
    hour: int = Form(...)
):
    try:
        # Create feature array
        features = np.array([[temperature, humidity, co, benzene, nox, hour, 0]])
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Predict
        prediction_scaled = model.predict(features_scaled)[0]
        
        # Rescale prediction
        prediction_actual = prediction_scaled * (y_max - y_min) + y_min
        
        # Convert to AQI scale
        aqi = min(500, max(0, (prediction_actual - 0) / (2000 - 0) * 500))
        
        # Categorize AQI
        if aqi < 50:
            category = "Good"
            color = "good"
        elif aqi < 100:
            category = "Moderate"
            color = "moderate"
        elif aqi < 150:
            category = "Unhealthy for Sensitive Groups"
            color = "unhealthy-sensitive"
        elif aqi < 200:
            category = "Unhealthy"
            color = "unhealthy"
        else:
            category = "Hazardous"
            color = "hazardous"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": round(aqi, 2),
            "category": category,
            "color": color,
            "temperature": temperature,
            "humidity": humidity,
            "co": co,
            "benzene": benzene,
            "nox": nox,
            "hour": hour
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}",
            "temperature": temperature,
            "humidity": humidity,
            "co": co,
            "benzene": benzene,
            "nox": nox,
            "hour": hour
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)