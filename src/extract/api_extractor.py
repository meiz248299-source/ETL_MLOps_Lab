# src/extract/api_extractor.py
import requests
import pandas as pd
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class APIExtractor:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        初始化API提取器
        
        参数:
        - base_url: API基础URL
        - api_key: API认证密钥(可选)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def extract_endpoint(self, endpoint: str, params: Dict = None, retries: int = 3) -> Dict:
        """
        从API端点提取数据
        
        作用:
        - 发送HTTP请求获取数据
        - 实现重试机制处理网络波动
        - 指数退避策略控制重试间隔
        
        重试机制:
        - 最多重试3次
        - 每次重试等待时间翻倍 (2, 4, 8秒)
        - 处理超时和网络错误
        """
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"第{attempt+1}次尝试失败: {e}")
                time.sleep(2 ** attempt)  # 指数退避
                if attempt == retries - 1:
                    raise
    
    def extract_to_dataframe(self, endpoint: str, params: Dict = None) -> pd.DataFrame:
        """
        提取API数据并转换为DataFrame
        
        作用:
        - 调用API获取JSON数据
        - 自动处理不同格式的响应
        - 转换为pandas DataFrame便于后续处理
        """
        data = self.extract_endpoint(endpoint, params)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            return pd.DataFrame([data])