# src/load/db_loader.py
from sqlalchemy import create_engine
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class DatabaseLoader:
    def __init__(self, connection_string: str):
        """
        初始化数据库加载器
        
        参数:
        - connection_string: 数据库连接字符串
        """
        self.engine = create_engine(connection_string)
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, 
                       if_exists: str = 'replace', 
                       index: bool = False) -> None:
        """
        加载DataFrame到数据库
        
        作用:
        - 将数据写入数据库表
        - if_exists参数控制写入方式:
          * replace: 删除并重新创建表
          * append: 追加到现有表
          * fail: 表存在时报错
        
        注意事项:
        - 大量数据时可能较慢
        - 考虑分批次加载
        """
        logger.info(f"加载{len(df)}行数据到表{table_name}")
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
        logger.info(f"成功加载数据到{table_name}")
    
    def load_batch(self, df_iterator, table_name: str, batch_size: int = 5000) -> None:
        """
        批量加载数据
        
        作用:
        - 分批写入，避免内存不足
        - 第1批使用replace，后续使用append
        - 适合非常大的数据集
        
        优势:
        - 内存友好
        - 可恢复性 (如果中途失败，已写入的数据不会丢失)
        """
        logger.info(f"以{batch_size}为批次加载数据到{table_name}")
        
        for i, df_batch in enumerate(df_iterator):
            df_batch.to_sql(table_name, self.engine, 
                          if_exists='append' if i > 0 else 'replace', 
                          index=False)
            logger.info(f"已加载批次 {i+1}: {len(df_batch)} 行")