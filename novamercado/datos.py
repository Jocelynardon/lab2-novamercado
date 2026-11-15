"""Lectura de los archivos publicados (catálogos, feriados, costos)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from novamercado import config


@dataclass
class Catalogo:
    tiendas: pd.DataFrame          # tienda_id, nombre, ciudad, departamento, tipo, fecha_apertura
    familias: pd.DataFrame         # familia_id, nombre
    feriados: pd.DataFrame         # fecha, nombre, tipo, ambito
    costos: pd.DataFrame           # familia_id, costo_faltante, costo_sobrante
    familias_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.familias_ids = sorted(self.familias["familia_id"].tolist())

    @property
    def tiendas_ids(self) -> set[int]:
        return set(int(t) for t in self.tiendas["tienda_id"])

    def fecha_apertura(self, tienda_id: int) -> date:
        fila = self.tiendas.loc[self.tiendas["tienda_id"] == tienda_id].iloc[0]
        return pd.Timestamp(fila["fecha_apertura"]).date()

    @property
    def ultima_fecha_feriados(self) -> date:
        return pd.Timestamp(self.feriados["fecha"].max()).date()

    def costos_dict(self) -> dict[str, tuple[float, float]]:
        return {
            r.familia_id: (float(r.costo_faltante), float(r.costo_sobrante))
            for r in self.costos.itertuples()
        }


def cargar_catalogo(data_dir: str | None = None) -> Catalogo:
    d = data_dir or config.DATA_DIR
    leer = lambda n: pd.read_csv(os.path.join(d, n), encoding="utf-8")
    tiendas = leer("tiendas.csv")
    tiendas["tienda_id"] = tiendas["tienda_id"].astype(int)
    tiendas["fecha_apertura"] = pd.to_datetime(tiendas["fecha_apertura"])
    feriados = leer("feriados.csv")
    feriados["fecha"] = pd.to_datetime(feriados["fecha"])
    return Catalogo(
        tiendas=tiendas,
        familias=leer("familias.csv"),
        feriados=feriados,
        costos=leer("costos.csv"),
    )


def leer_historico(data_dir: str | None = None) -> pd.DataFrame:
    d = data_dir or config.DATA_DIR
    h = pd.read_csv(os.path.join(d, "historico_ventas.csv"), parse_dates=["fecha"])
    h["tienda_id"] = h["tienda_id"].astype(int)
    return h
