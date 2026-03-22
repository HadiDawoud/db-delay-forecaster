from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "model.joblib"

app = FastAPI(
    title="DB Delay Forecaster API",
    description="Production-ready inference for train departure delays.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    # these match FEATURE_COLS in features.py
    hour: int
    weekday: int
    month: int
    is_weekend: int
    is_rush_hour: int
    lag_1: float
    lag_3: float
    lag_7: float
    rolling_mean_5: float
    rolling_std_5: float

@app.get("/")
def read_root():
    return {"status": "online", "model_available": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model file not found. Please train the model first.")
    
    # Load model and predict
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([request.model_dump()])
    prediction = model.predict(input_df)[0]
    
    return {
        "predicted_delay_min": round(float(prediction), 2),
        "unit": "minutes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
