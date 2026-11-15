"""DAG de reentrenamiento de NovaMercado.

Se dispara desde la API (POST /admin/reentrenar) con un dag_run_id propio; no tiene
programación periódica. Usa el histórico + las observaciones guardadas en la BD del
producto, compara el candidato con el modelo vigente y solo mueve el alias si mejora.
Si cualquier paso falla, el alias no se toca y la ejecución queda como 'fallido'.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task

log = logging.getLogger(__name__)


def _al_fallar(context):
    from novamercado.ciclo import marcar_fallido

    run_id = context["dag_run"].run_id
    exc = context.get("exception")
    marcar_fallido(run_id, f"falló la tarea {context['task_instance'].task_id}: "
                           f"{type(exc).__name__ if exc else 'error'}; el modelo vigente no cambió")


@dag(
    dag_id="reentrenamiento_novamercado",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=["novamercado", "ml"],
    default_args={"retries": 0, "on_failure_callback": _al_fallar},
    doc_md=__doc__,
)
def reentrenamiento_novamercado():

    @task(execution_timeout=timedelta(minutes=2))
    def verificar_datos() -> dict:
        from novamercado.db import Repositorio

        repo = Repositorio()
        repo.crear_reentrenamiento(get_current_context()["dag_run"].run_id)
        ventas = repo.ventas_entrenamiento()
        if len(ventas) < 1000:
            raise ValueError(f"datos insuficientes para entrenar: {len(ventas)} filas")
        resumen = {"filas": len(ventas), "desde": str(ventas["fecha"].min().date()),
                   "hasta": str(ventas["fecha"].max().date())}
        log.info("datos disponibles: %s", resumen)
        return resumen

    @task(execution_timeout=timedelta(minutes=8))
    def entrenar_evaluar_promover(resumen: dict) -> dict:
        from novamercado.ciclo import reentrenar

        resultado = reentrenar(get_current_context()["dag_run"].run_id)
        log.info("resultado: %s", resultado)
        return resultado

    entrenar_evaluar_promover(verificar_datos())


reentrenamiento_novamercado()
