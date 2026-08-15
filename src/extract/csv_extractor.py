# src/extract/csv_extractor.py
import pandas as pd
import dask.dataframe as dd
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)

class CSVExtractor:
    def __init__(self, file_path: str, chunk_size: int = 10000):
        """
        初始化CSV提取器
        
        参数:
        - file_path: CSV文件路径
        - chunk_size: 每批处理的行数
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def extract_chunks(self) -> Iterator[pd.DataFrame]:
        """
        分块提取数据
        
        作用: 
        - 将大文件分成小块(chunk)逐批读取
        - 避免一次性加载整个文件导致内存不足
        - 返回迭代器，每次yield一个DataFrame块
        """
        logger.info(f"从{self.file_path}分块提取数据，每块{self.chunk_size}行")
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield chunk
    
    def extract_with_dask(self) -> dd.DataFrame:
        """
        使用Dask进行大规模数据提取
        
        作用:
        - Dask支持超出内存的数据处理
        - 自动进行分区和并行计算
        - 适用于GB级别以上的大文件
        """
        logger.info(f"使用Dask从{self.file_path}提取大型数据集")
        return dd.read_csv(self.file_path, assume_missing=True)