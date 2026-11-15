"""Integración con MLflow: experimentos, Model Registry y alias del modelo vigente."""
from __future__ import annotations

import json
import logging
import os
import tempfile

import joblib

from novamercado import config
from novamercado.entrenamiento import ResultadoEntrenamiento
from novamercado.modelo import PaqueteModelo

log = logging.getLogger(__name__)
ARCHIVO_PAQUETE = "paquete.joblib"


def _cliente():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    return mlflow, MlflowClient(tracking_uri=config.MLFLOW_URI)


def version_vigente() -> str | None:
    """Versión con el alias vigente, o None si todavía no hay modelo. Lanza si MLflow no responde."""
    from mlflow.exceptions import MlflowException

    _, client = _cliente()
    try:
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        return str(mv.version)
    except MlflowException as e:
        # Modelo o alias inexistente -> todavía no hay vigente. Cualquier otro error (conexión) se propaga.
        if getattr(e, "error_code", "") in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            return None
        raise


def mlflow_disponible(timeout: float = 3.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{config.MLFLOW_URI}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def descargar_paquete(version: str, destino: str) -> PaqueteModelo:
    import mlflow

    _, client = _cliente()
    mv = client.get_model_version(config.MODEL_NAME, version)
    ruta = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=destino)
    return joblib.load(os.path.join(ruta, ARCHIVO_PAQUETE))


def registrar_resultado(res: ResultadoEntrenamiento, origen: str, version_anterior: str | None) -> tuple[str, bool]:
    """Registra el experimento y la nueva versión; mueve el alias si `res.promover`.

    Devuelve (versión vigente al terminar, promovido).
    """
    mlflow, client = _cliente()
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"reentrenamiento_{origen}") as run:
        mlflow.set_tags({
            "origen": origen, "promovido": str(res.promover), "motivo": res.motivo,
            "version_vigente_previa": str(version_anterior),
        })
        mlflow.log_params({
            "modelo": res.candidato.modelo.nombre, **{f"ventana_{k}": v for k, v in res.ventanas.items()},
            "filas_entrenamiento": res.n_filas, "mejora_minima": config.MEJORA_MINIMA,
            **{f"gbm_{k}": v for k, v in res.candidato.modelo.params.items()},
        })
        mlflow.log_metrics({f"candidato_{k}": v for k, v in res.metricas_candidato.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in res.metricas_baseline.items()})
        if res.metricas_vigente:
            mlflow.log_metrics({f"vigente_{k}": v for k, v in res.metricas_vigente.items()})
        with mlflow.start_run(run_name="baseline_dow4", nested=True):
            mlflow.log_param("modelo", "baseline_dow4")
            mlflow.log_metrics(res.metricas_baseline)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, "paquete")
            os.makedirs(ruta)
            joblib.dump(res.candidato, os.path.join(ruta, ARCHIVO_PAQUETE))
            with open(os.path.join(ruta, "multiplicadores.json"), "w") as f:
                json.dump(res.candidato.multiplicadores, f, indent=2)
            mlflow.log_artifacts(ruta, artifact_path="paquete")
        run_id = run.info.run_id
        fuente = f"{run.info.artifact_uri}/paquete"

    try:
        client.create_registered_model(config.MODEL_NAME, description="Pronóstico diario de ventas NovaMercado")
    except Exception:
        pass  # ya existe
    mv = client.create_model_version(config.MODEL_NAME, source=fuente, run_id=run_id,
                                     tags={"promovido": str(res.promover), "motivo": res.motivo[:250]})
    for k, v in res.metricas_candidato.items():
        client.set_model_version_tag(config.MODEL_NAME, mv.version, f"val_{k}", f"{v:.4f}")
    if res.promover:
        client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, mv.version)
        log.info("versión %s promovida a @%s", mv.version, config.MODEL_ALIAS)
        return str(mv.version), True
    return str(version_anterior) if version_anterior else str(mv.version), False
