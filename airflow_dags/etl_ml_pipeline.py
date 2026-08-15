# airflow_dags/etl_ml_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

sys.path.append('/opt/airflow')

# DAG默认参数
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['mlops@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4),
}

# 创建DAG
dag = DAG(
    'etl_ml_pipeline',
    default_args=default_args,
    description='ETL和ML训练流水线',
    schedule_interval='@daily',  # 每天运行
    catchup=False,
    tags=['etl', 'ml', 'mlops'],
)

def extract_data(**context):
    """
    数据提取任务
    
    作用:
    - 从多个源提取数据
    - 保存原始数据备份
    - 传递数据路径给下游任务
    """
    from src.extract.csv_extractor import CSVExtractor
    
    extractor = CSVExtractor('data/raw/source_data.csv')
    df = extractor.extract_with_dask().compute()
    
    backup_path = f'data/raw/backup_{context["ds"]}.parquet'
    df.to_parquet(backup_path)
    
    context['ti'].xcom_push(key='raw_data_path', value=backup_path)
    return f"提取了 {len(df)} 条记录"

def transform_data(**context):
    """
    数据转换任务
    
    作用:
    - 数据清洗
    - 特征工程
    - 保存处理后的数据
    """
    from src.transform.clean import DataCleaner
    from src.transform.feature_engineering import FeatureEngineer
    import pandas as pd
    
    raw_path = context['ti'].xcom_pull(key='raw_data_path')
    df = pd.read_parquet(raw_path)
    
    # 清洗数据
    cleaner = DataCleaner(df)
    df = cleaner.handle_missing_values(strategy='median')
    
    # 特征工程
    engineer = FeatureEngineer(df)
    if 'timestamp' in df.columns:
        df = engineer.create_temporal_features('timestamp')
    
    processed_path = f'data/processed/processed_{context["ds"]}.parquet'
    df.to_parquet(processed_path)
    
    context['ti'].xcom_push(key='processed_data_path', value=processed_path)
    return f"转换了 {len(df)} 条记录"

def train_model(**context):
    """
    模型训练任务
    
    作用:
    - 加载处理后的数据
    - 训练模型
    - 跟踪实验
    - 保存最佳模型
    """
    from src.model.train import ModelTrainer
    import pandas as pd
    
    processed_path = context['ti'].xcom_pull(key='processed_data_path')
    df = pd.read_parquet(processed_path)
    
    trainer = ModelTrainer(
        model_type='regression',
        tracking_uri='http://mlflow:5000',
        experiment_name='airflow_pipeline'
    )
    
    # 准备数据
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, target_col='target'
    )
    
    # 训练和追踪
    model, metrics = trainer.train_with_tracking(
        X_train, y_train, X_test, y_test,
        hyperparams={'n_estimators': 100}
    )
    
    # 保存最新模型
    trainer.save_model(model, 'models/model_latest.joblib')
    
    return "模型训练完成"

# 定义任务
start = DummyOperator(task_id='start', dag=dag)
extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)
transform = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)
train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)
end = DummyOperator(task_id='end', dag=dag)

# 定义依赖关系
start >> extract >> transform >> train >> end