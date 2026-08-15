"""
Feature Engineering Module
Purpose: Create new features from existing data
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Performs feature engineering operations:
    - Temporal feature extraction
    - Lag feature creation
    - Rolling statistics
    - Categorical encoding
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_temporal_features(self, datetime_col: str) -> pd.DataFrame:
        """
        Extract temporal features from datetime column
        
        Purpose:
        - Capture time-based patterns and seasonality
        
        Features created:
        - year: Year component
        - month: Month (1-12)
        - day: Day of month (1-31)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - hour: Hour (0-23)
        - is_weekend: Binary flag for weekend
        
        Why these features?
        - Capture seasonal patterns
        - Identify weekday/weekend differences
        - Analyze hourly trends
        """
        df = self.df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['hour'] = df[datetime_col].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Created temporal features from {datetime_col}")
        return df
    
    def create_lag_features(self, target_col: str, lags: List[int], 
                            group_by: str = None) -> pd.DataFrame:
        """
        Create lag features for time series data
        
        Purpose:
        - Use past values to predict future
        - Capture autocorrelation in time series
        
        Example: lag_1 = value at t-1, lag_7 = value 7 days ago
        
        Applications:
        - Sales forecasting
        - Stock price prediction
        - Weather prediction
        """
        df = self.df.copy()
        if group_by:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_by)[target_col].shift(lag)
        else:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created lag features: {lags}")
        return df
    
    def create_rolling_features(self, target_col: str, windows: List[int], 
                               group_by: str = None) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Purpose:
        - Smooth short-term fluctuations
        - Capture trends and volatility
        
        Rolling Mean: Trend indicator
        Rolling Std: Volatility indicator
        
        Applications:
        - Moving averages in trading
        - Volatility estimation
        - Smoothing noisy data
        """
        df = self.df.copy()
        for window in windows:
            if group_by:
                df[f'{target_col}_rolling_mean_{window}'] = (
                    df.groupby(group_by)[target_col].rolling(window).mean()
                    .reset_index(level=0, drop=True)
                )
                df[f'{target_col}_rolling_std_{window}'] = (
                    df.groupby(group_by)[target_col].rolling(window).std()
                    .reset_index(level=0, drop=True)
                )
            else:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        
        logger.info(f"Created rolling features with windows: {windows}")
        return df
    
    def encode_categorical(self, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Purpose:
        - Convert categorical data to numeric format for ML models
        
        One-Hot Encoding:
        - Creates binary columns for each category
        - Suitable for nominal categories
        - Avoids implied ordering
        
        Label Encoding:
        - Maps categories to numeric values (0, 1, 2, ...)
        - Suitable for ordinal categories
        - May imply ordering that doesn't exist
        """
        df = self.df.copy()
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        
        logger.info(f"Encoded categorical columns: {columns} using {method}")
        return df