# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import os
from datetime import datetime
import structlog
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus监控指标
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(title="MLOps Model API", version="1.0.0")

# 加载模型
model_path = os.getenv('MODEL_PATH', 'models/model_latest.joblib')
model = joblib.load(model_path)

# 请求模型
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus指标端点"""
    return generate_latest()

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    预测端点
    
    作用:
    1. 接收特征向量
    2. 调用模型进行预测
    3. 返回预测结果
    4. 记录指标和日志
    
    监控指标:
    - 请求计数
    - 响应延迟
    - 错误率
    """
    PREDICTION_COUNT.inc()S
    
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        
        # 结构化日志
        logger.info("预测完成", 
                   features=request.features, 
                   prediction=float(prediction))
        
        return {
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat(),
            "model_version": os.getenv('MODEL_VERSION', '1.0.0')
        }
    except Exception as e:
        logger.error("预测失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))