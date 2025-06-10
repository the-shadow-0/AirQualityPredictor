# 🌤️ Air Quality Prediction Web App

![AI1](https://github.com/user-attachments/assets/a8821270-45a6-4efd-b0b2-091729fb9409)

An end-to-end machine learning application that predicts Air Quality Index (AQI) based on environmental sensor data. This project combines a trained Gradient Boosting model with a modern web interface built with FastAPI and Jinja2 templates.

Screenshot of the Air Quality Prediction Interface
![AI2](https://github.com/user-attachments/assets/556d94e1-89c9-4215-84d8-3849cf436e4e)
![AI3](https://github.com/user-attachments/assets/41d9aa73-1968-4569-bed0-5f098534f3d7)
![AI1](https://github.com/user-attachments/assets/a8821270-45a6-4efd-b0b2-091729fb9409)

## ✨ Features

   Real-time AQI Predictions: Machine learning model predicts air quality based on environmental parameters

   Modern UI/UX: Glassmorphism design with intuitive controls and visual feedback

   Comprehensive Analysis:

Color-coded AQI categories (Good to Hazardous)

Interactive gauge visualization

Health recommendations

Sample Values: Pre-filled examples for quick testing

Responsive Design: Works seamlessly on desktop and mobile devices

Technical Highlights:

   Gradient Boosting Regressor model (R²: 0.92+)

   Robust feature scaling

   Input validation

   Error handling

## 📦 Installation

 ## Clone the repository:

   
    git clone https://github.com/the-shadow-0/AirQualityPredictor.git
    cd AirQualityPredictor

 ## Install dependencies:
   
    pip install -r requirements.txt

## Download dataset:

  Get the Air Quality Dataset from UCI Repository

  Place AirQualityUCI.csv in the project root directory

## 🚀 Usage

  ## Preprocess data and train model:
    
    python train_model.py

  ## Run the web application:
    
    uvicorn app:app --reload

  ## Access the web interface:
  
    Open your browser and visit: http://localhost:8000

## 🧪 Testing the Model

## Run sample predictions:

    python test_model.py

## Expected output:
text

Excellent conditions: Predicted AQI = 32.1
Good conditions: Predicted AQI = 45.5
Moderate conditions: Predicted AQI = 78.8
Unhealthy conditions: Predicted AQI = 135.2
Hazardous conditions: Predicted AQI = 285.0

## 🧩 Project Structure
text

air-quality-predictor/
├── static/               # Static assets
│   ├── style.css         # Main stylesheet
├── templates/            # HTML templates
│   └── index.html        # Main interface
├── app.py                # FastAPI application
├── train_model.py        # Model training script
├── test_model.py         # Model testing script
├── preprocess.py         # Data preprocessing
├── requirements.txt      # Dependencies
├── README.md             # This documentation
├── aqi_model.pkl         # Trained model
├── feature_scaler.pkl    # Feature scaler
└── target_scaler.pkl     # Target scaler

## 🧠 Machine Learning Details

## Model: Gradient Boosting Regressor

   ## Features:

Temperature (°C)

Humidity (%)

CO (mg/m³)

Benzene (µg/m³)

NOx (ppb)

Hour of day

Day of week

## Target: PT08.S5(O3) sensor readings (proxy for AQI)

   ## Performance:

   Training R²: 0.95

   Testing R²: 0.92

## 🌍 Environmental Impact

## This application helps:

   Identify air quality hazards

   Provide health recommendations

   Raise awareness about environmental factors

   Support urban planning decisions

## 🤝 Contributing

## We welcome contributions! Please follow these steps:

   Fork the repository

   Create a new branch (git checkout -b feature/your-feature)

   Commit your changes (git commit -am 'Add some feature')

   Push to the branch (git push origin feature/your-feature)

   Open a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
