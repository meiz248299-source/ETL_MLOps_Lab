"""
FastAPI Application
Purpose: REST API to serve the model with version control
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model API",
    description="REST API for machine learning model serving",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = os.getenv('MODEL_PATH', 'models/model_latest.joblib')
try:
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Request/Response schemas
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }

class PredictionResponse(BaseModel):
    prediction: float
    timestamp: str
    model_version: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "healthy", "service": "MLOps Prediction API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    
    Purpose:
    - Receive feature vector
    - Return prediction with metadata
    - Track metrics with Prometheus
    """
    PREDICTION_COUNT.inc()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        
        logger.info(f"Prediction made", extra={
            'features': request.features,
            'prediction': float(prediction)
        })
        
        return PredictionResponse(
            prediction=float(prediction),
            timestamp=datetime.now().isoformat(),
            model_version=os.getenv('MODEL_VERSION', '1.0.0')
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make batch predictions
    
    Purpose:
    - Efficient batch processing
    - Return multiple predictions
    """
    try:
        features = [req.features for req in requests]
        X = np.array(features)
        predictions = model.predict(X).tolist()
        
        return {
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)