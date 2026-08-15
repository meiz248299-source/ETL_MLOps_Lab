"""
Model Training Module
Purpose: Train machine learning model with MLflow tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple, Optional
import logging
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains machine learning models with experiment tracking
    
    Purpose:
    - Train models using scikit-learn
    - Track experiments with MLflow
    - Save models for deployment
    """
    
    def __init__(self, model_type: str = 'regression',
                 tracking_uri: str = 'http://localhost:5000',
                 experiment_name: str = 'mlops_experiment'):
        """
        Initialize model trainer
        
        Args:
            model_type: 'regression' or 'classification'
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        self.model_type = model_type
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        if model_type == 'regression':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str,
                     feature_cols: Optional[list] = None,
                     test_size: float = 0.2) -> Tuple:
        """
        Prepare train-test split with features
        
        Purpose:
        - Separate features and target
        - Split data for training and evaluation
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def train_with_tracking(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           hyperparams: Optional[Dict] = None) -> Tuple:
        """
        Train model with MLflow tracking
        
        Purpose:
        - Track experiment parameters and metrics in MLflow
        - Log model for versioning
        - Enable model comparison
        
        MLflow Benefits:
        - Experiment reproducibility
        - Model versioning
        - Parameter and metric tracking
        - Model registry
        """
        with mlflow.start_run() as run:
            # Log parameters
            if hyperparams:
                self.model.set_params(**hyperparams)
                mlflow.log_params(hyperparams)
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            if self.model_type == 'regression':
                train_metric = mean_squared_error(y_train, y_pred_train)
                test_metric = mean_squared_error(y_test, y_pred_test)
                metric_name = 'mse'
            else:
                train_metric = accuracy_score(y_train, y_pred_train)
                test_metric = accuracy_score(y_test, y_pred_test)
                metric_name = 'accuracy'
            
            # Log metrics to MLflow
            mlflow.log_metric(f'train_{metric_name}', train_metric)
            mlflow.log_metric(f'test_{metric_name}', test_metric)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Model trained with run_id: {run.info.run_id}")
            
            metrics = {
                f'train_{metric_name}': train_metric,
                f'test_{metric_name}': test_metric,
                'run_id': run.info.run_id
            }
            
            return self.model, metrics
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             param_grid: Dict[str, list],
                             cv: int = 5) -> Tuple:
        """
        Perform hyperparameter tuning with GridSearchCV
        
        Purpose:
        - Find optimal hyperparameters
        - Use cross-validation for robust evaluation
        
        Why GridSearch?
        - Exhaustive search over parameter space
        - Guarantees finding optimal combination
        - Time-consuming but thorough
        """
        logger.info(f"Starting hyperparameter tuning with {len(param_grid)} parameters")
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, 
                                   scoring='accuracy' if self.model_type == 'classification' else 'r2',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}, Best score: {best_score}")
        
        return grid_search.best_estimator_, {'best_params': best_params, 'best_score': best_score}
    
    def save_model(self, model, path: str = 'models/model.joblib'):
        """
        Save trained model locally
        
        Purpose:
        - Persist model for later use
        - Load for prediction
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")