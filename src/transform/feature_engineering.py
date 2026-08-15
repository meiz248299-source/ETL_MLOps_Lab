# src/transform/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_temporal_features(self, datetime_col: str) -> pd.DataFrame:
        """
        创建时间特征
        
        作用:
        - 从时间戳中提取有用信息
        - 创建循环特征捕捉时间模式
        
        提取的特征:
        - year: 年份
        - month: 月份 (1-12)
        - day: 日期 (1-31)
        - day_of_week: 星期几 (0=周一, 6=周日)
        - hour: 小时 (0-23)
        - is_weekend: 是否周末 (0/1)
        
        为什么需要这些特征?
        - 捕捉季节性模式
        - 识别工作日/周末差异
        - 分析小时级别的波动
        """
        df = self.df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['hour'] = df[datetime_col].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"从{datetime_col}创建了时间特征")
        return df
    
    def create_lag_features(self, target_col: str, lags: List[int], group_by: str = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        作用:
        - 为时间序列数据创建过去值
        - 捕捉历史趋势和自相关性
        
        例如: lag_1 = t-1时刻的值
              lag_7 = 7天前的值
        
        应用场景:
        - 销售预测 (使用过去销量)
        - 股票价格预测
        - 天气预测
        """
        df = self.df.copy()
        if group_by:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_by)[target_col].shift(lag)
        else:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"创建了滞后特征: {lags}")
        return df
    
    def create_rolling_features(self, target_col: str, windows: List[int], group_by: str = None) -> pd.DataFrame:
        """
        创建滚动窗口特征
        
        作用:
        - 计算滑动窗口内的统计量
        - 平滑短期波动，捕捉趋势
        
        常见统计量:
        - 滚动均值: 反映趋势
        - 滚动标准差: 反映波动性
        
        应用场景:
        - 移动平均线 (交易策略)
        - 波动率估计 (风险管理)
        """
        df = self.df.copy()
        for window in windows:
            if group_by:
                df[f'{target_col}_rolling_mean_{window}'] = (
                    df.groupby(group_by)[target_col].rolling(window).mean().reset_index(level=0, drop=True)
                )
                df[f'{target_col}_rolling_std_{window}'] = (
                    df.groupby(group_by)[target_col].rolling(window).std().reset_index(level=0, drop=True)
                )
            else:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        
        logger.info(f"使用窗口 {windows} 创建了滚动特征")
        return df
    
    def encode_categorical(self, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        编码分类变量
        
        作用:
        - 将类别型数据转换为数值型
        
        One-Hot编码:
        - 为每个类别创建二进制列
        - 适用于类别数量较少的情况
        - 避免类别间的顺序关系
        
        Label编码:
        - 将类别映射为数字 (0, 1, 2, ...)
        - 适用于有序类别或树模型
        - 可能引入不存在的顺序关系
        """
        df = self.df.copy()
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        
        logger.info(f"使用{method}方法编码了分类列: {columns}")
        return df