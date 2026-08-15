# src/extract/db_extractor.py
from sqlalchemy import create_engine, text
import pandas as pd
from typing import Dict, Any, Iterator
import logging

logger = logging.getLogger(__name__)

class DatabaseExtractor:
    def __init__(self, connection_string: str):
        """
        初始化数据库提取器
        
        参数:
        - connection_string: 数据库连接字符串
          格式: postgresql://user:password@host:port/database
        """
        self.engine = create_engine(connection_string)
    
    def extract_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        执行SQL查询并提取数据
        
        作用:
        - 将SQL查询结果直接转为DataFrame
        - 支持参数化查询，防止SQL注入
        - 适用于数据量适中的查询
        """
        logger.info(f"执行查询: {query[:100]}...")
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def extract_table_batch(self, table_name: str, batch_size: int = 5000) -> Iterator[pd.DataFrame]:
        """
        批量提取表数据
        
        作用:
        - 使用OFFSET/LIMIT分页提取
        - 适用于大表的批量导出
        - 每次返回一个批次的数据
        """
        offset = 0
        while True:
            query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
            df = self.extract_query(query)
            if df.empty:
                break
            yield df
            offset += batch_size
            logger.info(f"已提取批次，偏移量: {offset}")