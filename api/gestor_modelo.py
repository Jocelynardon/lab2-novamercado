"""Carga y recarga en caliente del modelo vigente + monitoreo de dependencias.

- Al arrancar intenta cargar la copia local (volumen) para responder aunque MLflow no esté.
- Un hilo revisa cada N segundos el alias en MLflow; si cambió de versión, descarga y
  reemplaza el modelo en memoria sin reiniciar el servicio.
- El mismo hilo verifica BD, MLflow y Airflow para que /health responda al instante.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import urllib.request

import joblib

from novamercado import config

log = logging.getLogger(__name__)


class GestorModelo:
    def __init__(self, repo=None, cache_dir: str | None = None, poll: int | None = None):
        self.repo = repo
        self.cache_dir = cache_dir or config.MODEL_CACHE_DIR
        self.poll = poll or config.POLL_SECONDS
        self._lock = threading.Lock()
        self.paquete = None
        self.version: str | None = None
        self.estado = {"base_de_datos": False, "mlflow": False, "airflow": False}
        self._hilo: threading.Thread | None = None
        self._parar = threading.Event()

    # ---------------------------------------------------------------- lectura
    def actual(self):
        with self._lock:
            return self.paquete, self.version

    # ----------------------------------------------------------------- caché
    def cargar_cache(self):
        meta = os.path.join(self.cache_dir, "meta.json")
        archivo = os.path.join(self.cache_dir, "paquete.joblib")
        if os.path.exists(meta) and os.path.exists(archivo):
            try:
                with open(meta) as f:
                    version = json.load(f)["version"]
                paquete = joblib.load(archivo)
                with self._lock:
                    self.paquete, self.version = paquete, str(version)
                log.info("modelo v%s cargado desde la caché local", version)
            except Exception:  # noqa: BLE001
                log.exception("no se pudo leer la caché del modelo")

    def _guardar_cache(self, paquete, version: str):
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = os.path.join(self.cache_dir, "paquete.joblib.tmp")
        joblib.dump(paquete, tmp)
        os.replace(tmp, os.path.join(self.cache_dir, "paquete.joblib"))
        with open(os.path.join(self.cache_dir, "meta.json"), "w") as f:
            json.dump({"version": version}, f)

    # --------------------------------------------------------- sincronización
    def refrescar_modelo(self):
        from novamercado import registro

        if not registro.mlflow_disponible():
            self.estado["mlflow"] = False
            return
        try:
            version = registro.version_vigente()
            self.estado["mlflow"] = True
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow no respondió: %s", type(e).__name__)
            self.estado["mlflow"] = False
            return
        if version is None or version == self.version:
            return
        log.info("nueva versión vigente detectada: v%s (actual v%s)", version, self.version)
        with tempfile.TemporaryDirectory() as tmp:
            paquete = registro.descargar_paquete(version, tmp)
        with self._lock:
            self.paquete, self.version = paquete, version
        try:
            self._guardar_cache(paquete, version)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo guardar la caché del modelo")

    def revisar_dependencias(self):
        if self.repo is not None:
            self.estado["base_de_datos"] = self.repo.ping()
        try:
            with urllib.request.urlopen(f"{config.AIRFLOW_URL}/api/v2/monitor/health", timeout=3) as r:
                datos = json.loads(r.read() or b"{}")
            sched = (datos.get("scheduler") or {}).get("status", "healthy")
            self.estado["airflow"] = r.status == 200 and sched == "healthy"
        except Exception:  # noqa: BLE001
            self.estado["airflow"] = False

    def ciclo(self):
        while not self._parar.is_set():
            for paso in (self.refrescar_modelo, self.revisar_dependencias):
                try:
                    paso()
                except Exception:  # noqa: BLE001
                    log.exception("error en %s", paso.__name__)
            self._parar.wait(self.poll)

    def iniciar(self):
        self.cargar_cache()
        self._hilo = threading.Thread(target=self.ciclo, name="gestor-modelo", daemon=True)
        self._hilo.start()

    def detener(self):
        self._parar.set()
