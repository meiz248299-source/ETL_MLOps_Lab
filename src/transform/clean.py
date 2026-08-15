"""
Data Cleaning Module
Purpose: Handle missing values, normalization, and outlier removal
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Performs data cleaning operations including:
    - Missing value handling
    - Data normalization
    - Outlier removal
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def handle_missing_values(self, strategy: str = 'median', 
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values with various strategies
        
        Purpose:
        - Clean data by handling NaN values
        - Strategies: drop, fill, median, mean, mode
        
        Why median?
        - Median is robust to outliers
        - Suitable for skewed distributions
        """
        df = self.df.copy()
        if columns is None:
            columns = df.columns
        
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        
        if strategy == 'drop':
            df = df.dropna(subset=columns)
        elif strategy == 'fill':
            df = df.fillna(0)
        elif strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'mode':
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        logger.info(f"Handled missing values using strategy: {strategy}")
        return df
    
    def normalize_data(self, method: str = 'standardize', 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numeric columns
        
        Purpose:
        - Scale features to similar ranges
        - Standardization: (x - mean) / std, results in mean=0, std=1
        - MinMax scaling: (x - min) / (max - min), results in [0,1] range
        
        Why normalize?
        - Prevents features with larger scales from dominating
        - Required for algorithms like SVM, neural networks
        - Improves convergence speed
        """
        df = self.df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if method == 'standardize':
                df[col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == 'minmax':
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df
    
    def remove_outliers(self, columns: Optional[List[str]] = None, 
                        method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method
        
        Purpose:
        - Identify and remove anomalous data points
        
        IQR Method:
        - Q1 = 25th percentile, Q3 = 75th percentile
        - IQR = Q3 - Q1
        - Outliers: < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
        - Robust to non-normal distributions
        
        Z-score Method:
        - z = (x - mean) / std
        - Outliers: |z| > 3
        - Assumes normal distribution
        """
        df = self.df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < 3]
        
        logger.info(f"Removed outliers using {method} method")
        return df