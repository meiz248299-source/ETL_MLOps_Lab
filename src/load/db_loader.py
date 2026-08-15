"""
Database Loader
Purpose: Load transformed data into relational databases using SQLAlchemy
"""

from sqlalchemy import create_engine
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class DatabaseLoader:
    """
    Loads data into databases with batch support
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database loader
        
        Args:
            connection_string: Database connection URL
            Example: postgresql://user:password@localhost:5432/database
        """
        self.engine = create_engine(connection_string)
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, 
                       if_exists: str = 'replace', 
                       index: bool = False,
                       dtype: Optional[Dict] = None) -> None:
        """
        Load DataFrame to database table
        
        Purpose:
        - Save processed data to database
        
        if_exists options:
        - replace: Drop and recreate table
        - append: Add to existing table
        - fail: Raise error if table exists
        
        Why SQLAlchemy?
        - Works with multiple database backends
        - Handles connection pooling
        - Provides ORM capabilities
        """
        logger.info(f"Loading {len(df)} rows to table {table_name}")
        df.to_sql(table_name, self.engine, if_exists=if_exists, 
                  index=index, dtype=dtype)
        logger.info(f"Successfully loaded data to {table_name}")
    
    def load_batch(self, df_iterator, table_name: str, 
                   batch_size: int = 5000) -> None:
        """
        Load data in batches for large datasets
        
        Purpose:
        - Handle large datasets without memory issues
        - First batch creates table, subsequent batches append
        - Resumable if interrupted
        """
        logger.info(f"Loading data to {table_name} in batches of {batch_size}")
        
        for i, df_batch in enumerate(df_iterator):
            df_batch.to_sql(table_name, self.engine, 
                          if_exists='append' if i > 0 else 'replace', 
                          index=False)
            logger.info(f"Loaded batch {i+1}: {len(df_batch)} rows")