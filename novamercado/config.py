"""Configuración leída de variables de entorno (ver .env.example)."""
import os

DATA_DIR = os.getenv("DATA_DIR", "/data")

DB_HOST = os.getenv("PRODUCT_DB_HOST", "product-db")
DB_PORT = int(os.getenv("PRODUCT_DB_PORT", "5432"))
DB_USER = os.getenv("PRODUCT_DB_USER", "novamercado")
DB_PASSWORD = os.getenv("PRODUCT_DB_PASSWORD", "novamercado")
DB_NAME = os.getenv("PRODUCT_DB_NAME", "novamercado")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "novamercado_ventas")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "novamercado_pronostico")

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-apiserver:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_ADMIN_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")
DAG_ID = os.getenv("RETRAIN_DAG_ID", "reentrenamiento_novamercado")

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/model_cache")
POLL_SECONDS = int(os.getenv("MODEL_POLL_SECONDS", "20"))

# Criterio de reemplazo: el candidato debe bajar el costo de validación al menos este %.
MEJORA_MINIMA = float(os.getenv("MEJORA_MINIMA", "0.01"))
DIAS_VALIDACION = int(os.getenv("DIAS_VALIDACION", "28"))
DIAS_CALIBRACION = int(os.getenv("DIAS_CALIBRACION", "28"))
