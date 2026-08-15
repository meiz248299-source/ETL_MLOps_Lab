"""
API Data Extractor
Purpose: Extract data from web APIs using requests library
"""

import requests
import pandas as pd
import time
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class APIExtractor:
    """
    Extracts data from REST APIs with retry logic and error handling
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API extractor
        
        Args:
            base_url: Base URL of the API
            api_key: Authentication key (optional)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def extract_endpoint(self, endpoint: str, params: Dict = None, retries: int = 3) -> Dict:
        """
        Extract data from API endpoint with retry logic
        
        Purpose:
        - Handle API failures gracefully with exponential backoff
        - Retry on network errors or timeout
        - Implements robust error handling
        """
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == retries - 1:
                    raise
    
    def extract_to_dataframe(self, endpoint: str, params: Dict = None) -> pd.DataFrame:
        """
        Extract API data and convert to DataFrame
        
        Purpose:
        - Convert JSON API response to pandas DataFrame
        - Handle different response formats automatically
        """
        data = self.extract_endpoint(endpoint, params)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            return pd.DataFrame([data])