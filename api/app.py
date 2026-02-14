from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np

import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_model import extract_features
from risk_logic import calculate_risk_and_alerts
from artifact_detection import detect_and_clean_artifacts

app = FastAPI(title="Smart Ambulance Anomaly Detection API")

class VitalsData(BaseModel):
    timestamp: float
    heart_rate: float
    spo2: float
    bp_systolic: float
    bp_diastolic: float
    vibration: float

class PredictionResponse(BaseModel):
    anomaly: bool
    risk_score: float
    confidence: float
    details: str

import traceback

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Smart Ambulance API"}

from fastapi import Request

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(data: List[VitalsData]):
    """
    Expects a window of vitals data (typically 30-60 seconds at 1Hz).
    """
    try:
        if len(data) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data points for stable window analysis.")
        
        # Convert input to DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        # 1. Artifact Detection & Cleaning
        df_clean = detect_and_clean_artifacts(df)
        
        # 2. Feature Extraction (latest window)
        feats = extract_features(df_clean)
        
        # 3. Anomaly & Risk Logic
        # We create a mini-batch with the window results to reuse calculate_risk_and_alerts
        temp_df = pd.DataFrame([{
            'timestamp': df['timestamp'].iloc[-1],
            'is_anomaly': False, # Placeholder
            'confidence': feats['confidence_mean'],
            'reasons': "", # Placeholder
            **feats
        }])
        
        # Re-run rule-based anomaly check for the reasons string
        reasons = []
        if feats['heart_rate_slope'] > 0.05 and feats['heart_rate_mean'] > 100:
            reasons.append("Rising HR trend")
        if feats['spo2_slope'] < -0.01 and feats['spo2_mean'] < 95:
            reasons.append("Declining SpO2 trend")
        if feats['bp_systolic_slope'] > 0.1 and feats['bp_systolic_mean'] > 140:
            reasons.append("Rising Systolic BP")
        
        temp_df['reasons'] = "; ".join(reasons)
        
        # Risk calculation
        risk_df = calculate_risk_and_alerts(temp_df)
        result = risk_df.iloc[0]
        
        # Handle NaNs for JSON serialization
        risk_score = float(result['risk_score'])
        if np.isnan(risk_score): risk_score = 0.0
        
        confidence = float(result['final_confidence'])
        if np.isnan(confidence): confidence = 0.0
        
        return PredictionResponse(
            anomaly=bool(result['alert_triggered']),
            risk_score=risk_score,
            confidence=confidence,
            details=str(result['alert_comment']) if not pd.isna(result['alert_comment']) else "Processing error"
        )
    except Exception as e:
        with open("server_error.log", "w") as f:
            f.write(traceback.format_exc())
            f.write(str(e))
        # Log to stderr too for docker/cloud logs
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
