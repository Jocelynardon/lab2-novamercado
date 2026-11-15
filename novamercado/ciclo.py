"""Orquestación: arranque desde cero (bootstrap) y reentrenamiento (lo llama el DAG)."""
from __future__ import annotations

import logging
import tempfile
import time

from novamercado import config
from novamercado.datos import cargar_catalogo, leer_historico
from novamercado.db import Repositorio

log = logging.getLogger(__name__)


def _esperar(nombre: str, prueba, intentos: int = 90, pausa: float = 2.0):
    for i in range(intentos):
        try:
            if prueba():
                return
        except Exception as e:  # noqa: BLE001
            log.info("esperando %s (%s)", nombre, type(e).__name__)
        time.sleep(pausa)
    raise RuntimeError(f"{nombre} no respondió a tiempo")


def preparar_base_de_datos(repo: Repositorio | None = None):
    """Crea el esquema y carga el histórico una sola vez (idempotente)."""
    repo = repo or Repositorio()
    _esperar("base de datos", repo.ping)
    repo.crear_esquema()
    if repo.meta("carga_inicial") is None:
        n = repo.cargar_historico(leer_historico())
        repo.set_meta("carga_inicial", f"{n} filas")
        log.info("histórico cargado: %d filas", n)
    else:
        log.info("histórico ya cargado (%s)", repo.meta("carga_inicial"))


def _entrenar(repo: Repositorio, origen: str):
    from novamercado import registro
    from novamercado.entrenamiento import entrenar_y_decidir

    cat = cargar_catalogo()
    version_prev = registro.version_vigente()
    vigente = None
    if version_prev:
        with tempfile.TemporaryDirectory() as tmp:
            vigente = registro.descargar_paquete(version_prev, tmp)
    ventas = repo.ventas_entrenamiento()
    t0 = time.time()
    res = entrenar_y_decidir(ventas, cat, vigente)
    log.info("entrenamiento en %.1fs: %s", time.time() - t0, res.motivo)
    version, promovido = registro.registrar_resultado(res, origen, version_prev)
    return res, version, promovido


def bootstrap():
    """Primer arranque: BD + datos + primer modelo. Si ya existe un modelo vigente, no hace nada."""
    from novamercado import registro

    repo = Repositorio()
    preparar_base_de_datos(repo)
    _esperar("MLflow", registro.mlflow_disponible)
    vigente = registro.version_vigente()
    if vigente is not None:
        log.info("ya existe un modelo vigente (v%s); no se reentrena", vigente)
        return
    res, version, _ = _entrenar(repo, "bootstrap")
    log.info("primer modelo registrado: v%s", version)


def reentrenar(run_id: str):
    """Lo ejecuta el DAG. Registra el resultado en la tabla `reentrenamientos`."""
    repo = Repositorio()
    repo.crear_reentrenamiento(run_id)
    res, version, promovido = _entrenar(repo, f"airflow:{run_id}")
    repo.actualizar_reentrenamiento(
        run_id, "exitoso", promovido, version, res.motivo,
        {"candidato": res.metricas_candidato, "baseline": res.metricas_baseline,
         "vigente": res.metricas_vigente, "ventanas": res.ventanas})
    return {"promovido": promovido, "version": version, "motivo": res.motivo}


def marcar_fallido(run_id: str, motivo: str):
    from novamercado import registro

    try:
        version = registro.version_vigente()
    except Exception:  # noqa: BLE001
        version = None
    Repositorio().actualizar_reentrenamiento(run_id, "fallido", False, version, motivo[:500])


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
        bootstrap()
