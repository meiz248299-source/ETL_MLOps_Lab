"""
CSV Data Extractor
Purpose: Extract data from CSV files with batch processing for large datasets
"""

import pandas as pd
import dask.dataframe as dd
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)

class CSVExtractor:
    """
    Extracts data from CSV files with chunking for memory efficiency
    and Dask support for large-scale data processing
    """
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        """
        Initialize CSV extractor
        
        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk for batch processing
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def extract_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Extract data in chunks for memory efficiency
        
        Purpose: 
        - Read large CSV files in manageable batches
        - Prevents memory overflow when processing large datasets
        - Returns an iterator that yields DataFrame chunks
        """
        logger.info(f"Extracting data from {self.file_path} in chunks of {self.chunk_size}")
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield chunk
    
    def extract_with_dask(self) -> dd.DataFrame:
        """
        Use Dask for large-scale data extraction
        
        Purpose:
        - Handles datasets larger than available memory
        - Automatically partitions data for parallel processing
        - Suitable for GB+ size files
        """
        logger.info(f"Extracting large dataset using Dask from {self.file_path}")
        return dd.read_csv(self.file_path, assume_missing=True)