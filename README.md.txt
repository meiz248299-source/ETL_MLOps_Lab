# ETL and MLOps Pipeline

## Overview
This project implements a scalable ETL (Extract, Transform, Load) pipeline integrated with MLOps workflows for efficient model deployment and monitoring.

## Features
- **Extract**: CSV, Database (SQLAlchemy), and Web API extraction
- **Transform**: Data cleaning, normalization, feature engineering with Dask
- **Load**: PostgreSQL/SQLite database and AWS S3 cloud storage
- **ML Model**: Scikit-learn training with MLflow tracking
- **API**: FastAPI REST API for model serving
- **Orchestration**: Apache Airflow for pipeline automation
- **CI/CD**: GitHub Actions for automated deployment
- **Monitoring**: Prometheus and Grafana for real-time monitoring

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/meiz248299-source/ETL_MLOps_Lab.git
cd ETL_MLOps_Lab