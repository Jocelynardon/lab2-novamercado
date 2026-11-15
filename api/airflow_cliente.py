"""Cliente mínimo de la API REST v2 de Airflow 3 (autenticación con JWT)."""
from __future__ import annotations

import logging

import requests

from novamercado import config

log = logging.getLogger(__name__)


class ErrorAirflow(Exception):
    pass


class ClienteAirflow:
    def __init__(self, url: str | None = None, usuario: str | None = None,
                 clave: str | None = None, dag_id: str | None = None):
        self.url = (url or config.AIRFLOW_URL).rstrip("/")
        self.usuario = usuario or config.AIRFLOW_USER
        self.clave = clave or config.AIRFLOW_PASSWORD
        self.dag_id = dag_id or config.DAG_ID
        self._token: str | None = None

    def _obtener_token(self) -> str:
        r = requests.post(f"{self.url}/auth/token",
                          json={"username": self.usuario, "password": self.clave}, timeout=10)
        if r.status_code not in (200, 201):
            # Con SIMPLE_AUTH_MANAGER_ALL_ADMINS=true, Airflow entrega el token sin credenciales.
            r2 = requests.get(f"{self.url}/auth/token", timeout=10)
            if r2.status_code not in (200, 201):
                raise ErrorAirflow(f"no se pudo autenticar en Airflow ({r.status_code}/{r2.status_code})")
            r = r2
        self._token = r.json()["access_token"]
        return self._token

    def _pedir(self, metodo: str, ruta: str, **kw) -> requests.Response:
        try:
            for intento in range(2):
                token = self._token or self._obtener_token()
                r = requests.request(metodo, f"{self.url}{ruta}", timeout=10,
                                     headers={"Authorization": f"Bearer {token}"}, **kw)
                if r.status_code == 401 and intento == 0:
                    self._token = None
                    continue
                return r
        except requests.RequestException as e:
            raise ErrorAirflow(f"Airflow no disponible: {type(e).__name__}") from e
        raise ErrorAirflow("Airflow rechazó la autenticación")

    def disparar(self, run_id: str) -> str:
        # Asegura que el DAG no esté en pausa (por si se creó pausado).
        self._pedir("PATCH", f"/api/v2/dags/{self.dag_id}", json={"is_paused": False})
        r = self._pedir("POST", f"/api/v2/dags/{self.dag_id}/dagRuns",
                        json={"dag_run_id": run_id, "logical_date": None, "conf": {"origen": "api"}})
        if r.status_code not in (200, 201):
            raise ErrorAirflow(f"Airflow no aceptó la ejecución ({r.status_code}): {r.text[:200]}")
        return r.json().get("dag_run_id", run_id)

    def estado(self, run_id: str) -> str | None:
        """queued | running | success | failed, o None si Airflow no conoce la ejecución."""
        r = self._pedir("GET", f"/api/v2/dags/{self.dag_id}/dagRuns/{run_id}")
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise ErrorAirflow(f"respuesta inesperada de Airflow ({r.status_code})")
        return r.json().get("state")
