"""
Database Extractor
Purpose: Extract data from databases using SQLAlchemy
"""

from sqlalchemy import create_engine, text
import pandas as pd
from typing import Dict, Any, Iterator
import logging

logger = logging.getLogger(__name__)

class DatabaseExtractor:
    """
    Extracts data from relational databases with batch processing support
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database extractor
        
        Args:
            connection_string: Database connection URL
            Example: postgresql://user:password@localhost:5432/database
        """
        self.engine = create_engine(connection_string)
    
    def extract_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute SQL query and extract data
        
        Purpose:
        - Run parameterized SQL queries safely
        - Convert results directly to DataFrame
        - Prevents SQL injection through parameter binding
        """
        logger.info(f"Executing query: {query[:100]}...")
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def extract_table_batch(self, table_name: str, batch_size: int = 5000) -> Iterator[pd.DataFrame]:
        """
        Extract table data in batches using OFFSET/LIMIT
        
        Purpose:
        - Handle large tables without memory issues
        - Process data in manageable batches
        - Supports incremental extraction
        """
        offset = 0
        while True:
            query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
            df = self.extract_query(query)
            if df.empty:
                break
            yield df
            offset += batch_size
            logger.info(f"Extracted batch at offset: {offset}")