"""
Airflow DAG
Purpose: Schedule and automate ETL and MLOps pipelines
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
import sys
import os

sys.path.append('/opt/airflow')

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['mlops@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4),
}

dag = DAG(
    'etl_ml_pipeline',
    default_args=default_args,
    description='Complete ETL and ML pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['etl', 'ml', 'mlops'],
)

def extract_data(**context):
    """Extract data from multiple sources"""
    from src.extract.csv_extractor import CSVExtractor
    from src.extract.db_extractor import DatabaseExtractor
    
    # Extract from CSV
    csv_extractor = CSVExtractor('data/raw/source_data.csv')
    df = csv_extractor.extract_with_dask().compute()
    
    # Save raw data backup
    backup_path = f'data/raw/backup_{context["ds"]}.parquet'
    df.to_parquet(backup_path)
    
    context['ti'].xcom_push(key='raw_data_path', value=backup_path)
    return f"Extracted {len(df)} records"

def transform_data(**context):
    """Apply transformations"""
    from src.transform.clean import DataCleaner
    from src.transform.feature_engineering import FeatureEngineer
    import pandas as pd
    
    raw_path = context['ti'].xcom_pull(key='raw_data_path')
    df = pd.read_parquet(raw_path)
    
    # Clean data
    cleaner = DataCleaner(df)
    df = cleaner.handle_missing_values(strategy='median')
    df = cleaner.remove_outliers(method='iqr')
    
    # Feature engineering
    engineer = FeatureEngineer(df)
    if 'timestamp' in df.columns:
        df = engineer.create_temporal_features('timestamp')
    
    # Save transformed data
    processed_path = f'data/processed/processed_{context["ds"]}.parquet'
    df.to_parquet(processed_path)
    
    context['ti'].xcom_push(key='processed_data_path', value=processed_path)
    return f"Transformed {len(df)} records"

def train_model(**context):
    """Train model and track with MLflow"""
    from src.model.train import ModelTrainer
    import pandas as pd
    
    processed_path = context['ti'].xcom_pull(key='processed_data_path')
    df = pd.read_parquet(processed_path)
    
    trainer = ModelTrainer(
        model_type='regression',
        tracking_uri='http://mlflow:5000',
        experiment_name='airflow_pipeline'
    )
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, target_col='target')
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    best_model, tuning_results = trainer.hyperparameter_tuning(X_train, y_train, param_grid)
    
    # Train and track
    model, metrics = trainer.train_with_tracking(
        X_train, y_train, X_test, y_test,
        hyperparams=tuning_results['best_params'],
        log_params={'data_date': context['ds']}
    )
    
    # Save model
    trainer.save_model(model, 'models/model_latest.joblib')
    
    context['ti'].xcom_push(key='model_metrics', value=metrics)
    return "Model training completed"

def deploy_model(**context):
    """Deploy model (update API)"""
    import shutil
    
    # Copy latest model to API location
    shutil.copy('models/model_latest.joblib', 'api/models/model.joblib')
    
    # Restart API service (if using Docker)
    # This would typically trigger a deployment pipeline
    return "Model deployed"

# Define tasks
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

deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

notify_success = EmailOperator(
    task_id='notify_success',
    to=['mlops@example.com'],
    subject='MLOps Pipeline - Success',
    html_content='<h2>Pipeline completed successfully!</h2>',
    dag=dag
)

notify_failure = EmailOperator(
    task_id='notify_failure',
    to=['mlops@example.com', 'oncall@example.com'],
    subject='MLOps Pipeline - Failed',
    html_content='<h2>Pipeline failed!</h2><p>Please check logs.</p>',
    dag=dag,
    trigger_rule=TriggerRule.ONE_FAILED
)

# Define dependencies
start >> extract >> transform >> train >> deploy
deploy >> [notify_success, notify_failure]