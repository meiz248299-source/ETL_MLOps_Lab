# src/model/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
import logging
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str = 'regression', 
                 tracking_uri: str = 'http://localhost:5000',
                 experiment_name: str = 'mlops_experiment'):
        """
        初始化模型训练器
        
        参数:
        - model_type: 模型类型 ('regression' 或 'classification')
        - tracking_uri: MLflow追踪服务器URI
        - experiment_name: MLflow实验名称
        
        MLflow的作用:
        1. 追踪实验参数
        2. 记录模型性能指标
        3. 版本化管理模型
        4. 模型注册和部署
        """
        self.model_type = model_type
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        if model_type == 'regression':
            # 随机森林回归器
            # - 集成学习方法
            # - 对异常值鲁棒
            # - 能处理非线性关系
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train_with_tracking(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           hyperparams: Dict[str, Any] = None) -> Tuple[Any, Dict[str, float]]:
        """
        训练模型并记录到MLflow
        
        作用:
        1. 启动MLflow运行
        2. 记录超参数
        3. 训练模型
        4. 计算并记录指标
        5. 保存模型到MLflow
        
        优势:
        - 所有实验可追溯
        - 便于比较不同模型
        - 自动记录代码版本
        """
        
        with mlflow.start_run() as run:
            # 记录参数
            if hyperparams:
                self.model.set_params(**hyperparams)
                mlflow.log_params(hyperparams)
            
            # 训练模型
            logger.info("开始训练模型...")
            self.model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # 计算指标
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # 记录指标到MLflow
            mlflow.log_metric('train_mse', train_mse)
            mlflow.log_metric('test_mse', test_mse)
            
            # 保存模型
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"模型训练完成，运行ID: {run.info.run_id}")
            
            metrics = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'run_id': run.info.run_id
            }
            
            return self.model, metrics
    
    def save_model(self, model, path: str = 'models/model.joblib'):
        """保存模型到本地"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"模型保存到 {path}")