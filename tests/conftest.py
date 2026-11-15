"""Fixtures: catálogo pequeño en memoria, repositorio y modelo falsos (sin BD ni MLflow)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novamercado.datos import Catalogo  # noqa: E402
from novamercado.esquemas import Calidad  # noqa: E402
from novamercado.normalizacion import ResolutorFamilias  # noqa: E402

FAMILIAS = [("abarrotes", "Abarrotes"), ("bebidas", "Bebidas"), ("lacteos", "Lácteos"),
            ("carnes", "Carnes"), ("panaderia", "Panadería"), ("frutas_verduras", "Frutas y verduras"),
            ("limpieza", "Limpieza"), ("cuidado_personal", "Cuidado personal"), ("licores", "Licores"),
            ("snacks", "Snacks y dulces"), ("bebe", "Bebé"), ("mascotas", "Mascotas")]


@pytest.fixture(scope="session")
def catalogo() -> Catalogo:
    tiendas = pd.DataFrame({
        "tienda_id": list(range(1, 11)),
        "nombre": [f"T{i}" for i in range(1, 11)],
        "ciudad": ["Guatemala"] * 10,
        "departamento": ["Guatemala"] * 10,
        "tipo": ["hipermercado"] * 10,
        "fecha_apertura": pd.to_datetime(["2019-01-01"] * 9 + ["2025-03-15"]),
    })
    familias = pd.DataFrame(FAMILIAS, columns=["familia_id", "nombre"])
    feriados = pd.DataFrame({"fecha": pd.to_datetime(["2025-12-25", "2026-12-31"]),
                             "nombre": ["Navidad", "Fin de año"], "tipo": ["feriado", "medio_dia"],
                             "ambito": ["nacional", "nacional"]})
    costos = pd.DataFrame({"familia_id": [f for f, _ in FAMILIAS],
                           "costo_faltante": [1.0] * 12, "costo_sobrante": [0.5] * 12})
    return Catalogo(tiendas=tiendas, familias=familias, feriados=feriados, costos=costos)


@pytest.fixture(scope="session")
def reglas(catalogo):
    return dict(
        tiendas_validas=catalogo.tiendas_ids,
        familias=ResolutorFamilias(catalogo.familias_ids,
                                   dict(zip(catalogo.familias.familia_id, catalogo.familias.nombre))),
        apertura={int(r.tienda_id): r.fecha_apertura.date() for r in catalogo.tiendas.itertuples()},
    )


def payload_prediccion(**cambios):
    p = {"tienda_id": 5, "fecha": "2025-11-17",
         "familias": [{"familia_id": f, "en_promocion": 0} for f, _ in FAMILIAS]}
    p.update(cambios)
    return p


def payload_observacion(**cambios):
    p = {"tienda_id": 5, "fecha": "2025-11-17",
         "ventas": [{"familia_id": f, "ventas": 100 + i} for i, (f, _) in enumerate(FAMILIAS)]}
    p.update(cambios)
    return p


class RepoFalso:
    """Imita a novamercado.db.Repositorio en memoria."""

    def __init__(self):
        self.ventas: dict[tuple, int] = {}
        self.solicitudes: list[dict] = []
        self.reentrenos: dict[str, dict] = {}

    def ping(self):
        return True

    def registrar_rechazo(self, tipo, payload, detalle):
        self.solicitudes.append({"tipo": tipo, "estado": "rechazado", "detalle": detalle})

    def registrar_prediccion(self, payload, req, calidad, preds, version):
        self.solicitudes.append({"tipo": "prediccion", "estado": calidad.estado, "version": version,
                                 "n": len(preds)})
        return len(self.solicitudes)

    def guardar_observacion(self, payload, req, observaciones):
        from novamercado.esquemas import ObservacionCalidad
        n, obs = 0, list(observaciones)
        for it in req.ventas:
            k = (req.fecha, req.tienda_id, it.familia_id)
            antes = self.ventas.get(k)
            if antes != it.ventas:
                n += 1
                if antes is not None:
                    obs.append(ObservacionCalidad(campo=f"ventas.{it.familia_id}", accion="advertir",
                                                  detalle="reemplazo"))
                self.ventas[k] = it.ventas
        self.solicitudes.append({"tipo": "observacion", "estado": Calidad.desde(obs).estado})
        return n, obs

    def historia_tienda(self, tienda_id, fecha, dias=35):
        filas = [{"fecha": pd.Timestamp(f), "tienda_id": t, "familia_id": fam, "ventas": v}
                 for (f, t, fam), v in self.ventas.items() if t == tienda_id and f < fecha]
        return pd.DataFrame(filas, columns=["fecha", "tienda_id", "familia_id", "ventas"])

    def crear_reentrenamiento(self, rid):
        self.reentrenos.setdefault(rid, {"id": rid, "estado": "en_curso", "promovido": None,
                                         "version": None, "motivo": None})

    def actualizar_reentrenamiento(self, rid, estado, promovido=None, version=None, motivo=None,
                                   metricas=None):
        self.reentrenos[rid] = {"id": rid, "estado": estado, "promovido": promovido,
                                "version": version, "motivo": motivo}

    def obtener_reentrenamiento(self, rid):
        return self.reentrenos.get(rid)


class PaqueteFalso:
    def predecir(self, X):
        e = 100.0 + X["en_promocion"].to_numpy(float) * 10
        return e, np.round(e * 1.1).astype(int)


class GestorFalso:
    def __init__(self, con_modelo=True):
        self.paquete = PaqueteFalso() if con_modelo else None
        self.version = "7" if con_modelo else None
        self.estado = {"base_de_datos": True, "mlflow": True, "airflow": True}

    def actual(self):
        return self.paquete, self.version

    def iniciar(self):
        pass

    def detener(self):
        pass


class AirflowFalso:
    def __init__(self, disponible=True):
        self.disponible = disponible
        self.runs: dict[str, str] = {}

    def disparar(self, rid):
        if not self.disponible:
            raise RuntimeError("Airflow caído")
        self.runs[rid] = "running"
        return rid

    def estado(self, rid):
        return self.runs.get(rid)
