# src/transform/clean.py
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def handle_missing_values(self, strategy: str = 'median', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理缺失值
        
        作用:
        - 识别并处理数据中的缺失值
        - 提供多种填充策略:
          * drop: 删除包含缺失值的行
          * fill: 用0填充
          * median: 用中位数填充(对数值列)
          * mean: 用均值填充
          * mode: 用众数填充
        
        为什么选择中位数?
        - 中位数对异常值不敏感
        - 适合偏态分布的数据
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
        
        logger.info(f"使用策略 {strategy} 处理了缺失值")
        return df
    
    def normalize_data(self, method: str = 'standardize', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        数据归一化
        
        作用:
        - 将数值特征缩放到统一范围
        - 标准化 (standardize): 均值为0，标准差为1
          公式: (x - mean) / std
          适用于需要正态分布假设的算法
        
        - 最小-最大缩放 (minmax): 缩放到[0,1]范围
          公式: (x - min) / (max - min)
          适用于需要固定范围的算法(如神经网络)
        """
        df = self.df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if method == 'standardize':
                df[col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == 'minmax':
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        logger.info(f"使用{method}方法归一化了{len(columns)}列")
        return df
    
    def remove_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr') -> pd.DataFrame:
        """
        移除异常值
        
        作用:
        - 识别并移除数据中的异常值
        
        - IQR方法 (四分位距):
          定义: Q1 = 25%分位数, Q3 = 75%分位数
          异常值: < Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR
          适用于任何分布的数据
        
        - Z-score方法:
          定义: z = (x - mean) / std
          异常值: |z| > 3
          适用于近似正态分布的数据
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
        
        logger.info(f"使用{method}方法移除了异常值")
        return df