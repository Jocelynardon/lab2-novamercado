# Contexto del proyecto (para agentes de programación)

Servicio de pronóstico diario de ventas por tienda y familia para **NovaMercado** (Laboratorio 2,
Product Development). Stack: FastAPI (puerto 8000), PostgreSQL propio del producto, MLflow
(servicio `mlflow`, alias `champion`), Apache Airflow 3 (reentrenamiento). Todo con
`docker compose up -d --build` (ver README.md).

## Mapa del código
- `novamercado/normalizacion.py`: política reparar/advertir/rechazar. Es **puro**; no meter BD ni HTTP.
- `novamercado/esquemas.py`: modelos Pydantic del contrato. No renombrar ni quitar campos.
- `novamercado/features.py`: una sola función de features para entrenar **y** predecir. Regla: para la fecha D solo ventas < D.
- `novamercado/modelo.py`: baseline, GBM Poisson, costo de negocio, multiplicadores (newsvendor).
- `novamercado/entrenamiento.py`: ventanas train/calibración/validación y criterio de reemplazo.
- `novamercado/registro.py`: MLflow (versiones, alias). `novamercado/db.py`: PostgreSQL.
- `novamercado/ciclo.py`: bootstrap idempotente y reentrenamiento (lo llama el DAG).
- `api/main.py`: endpoints (`crear_app` acepta dobles para pruebas). `api/gestor_modelo.py`: recarga en caliente y salud.
- `dags/reentrenamiento.py`: DAG `reentrenamiento_novamercado` (sin schedule; lo dispara la API).

## Reglas
- Respetar exactamente el contrato del Anexo (respuestas pueden tener campos extra, nunca menos).
- No usar el reloj del sistema para validar fechas.
- Toda política nueva de datos: documentarla en DECISIONES.md y agregar prueba en `tests/`.
- Pruebas: `docker compose run --rm --no-deps api pytest -q` (no deben requerir MLflow/Airflow/BD).
- Versiones fijas en `requirements/*.txt` y Dockerfiles; API y Airflow deben compartir versiones
  de scikit-learn/pandas/numpy (el modelo se serializa con joblib).
- Datos (`data/`) y credenciales (`.env`) nunca van al repositorio.
