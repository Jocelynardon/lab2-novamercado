"""Entrenamiento, evaluación y criterio de reemplazo (parte pura, sin BD ni MLflow)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from novamercado import config
from novamercado.datos import Catalogo
from novamercado.features import construir_features
from novamercado.modelo import (
    Baseline, ModeloGBM, PaqueteModelo, ajustar_multiplicadores, metricas, sugerida_desde,
)

log = logging.getLogger(__name__)


@dataclass
class ResultadoEntrenamiento:
    candidato: PaqueteModelo
    metricas_candidato: dict
    metricas_baseline: dict
    metricas_vigente: dict | None
    promover: bool
    motivo: str
    ventanas: dict
    n_filas: int


def particionar(fechas: pd.Series, dias_val: int, dias_cal: int) -> dict:
    fin = pd.Timestamp(fechas.max())
    ini_val = fin - pd.Timedelta(days=dias_val - 1)
    ini_cal = ini_val - pd.Timedelta(days=dias_cal)
    return {"inicio_calibracion": ini_cal, "inicio_validacion": ini_val, "fin": fin}


def _evaluar(modelo, X_cal, X_val, costos) -> tuple[dict, dict]:
    e_cal = modelo.predict(X_cal)
    mult = ajustar_multiplicadores(e_cal, X_cal["ventas"], X_cal["familia_id"], costos)
    e_val = modelo.predict(X_val)
    s_val = sugerida_desde(e_val, X_val["familia_id"].tolist(), mult)
    return mult, metricas(X_val["ventas"], e_val, s_val, X_val["familia_id"].tolist(), costos)


def entrenar_y_decidir(ventas: pd.DataFrame, cat: Catalogo,
                       vigente: PaqueteModelo | None = None,
                       dias_val: int | None = None, dias_cal: int | None = None,
                       mejora_minima: float | None = None) -> ResultadoEntrenamiento:
    """
    ventas: fecha, tienda_id, familia_id, ventas, en_promocion (histórico + observaciones).

    Ventanas temporales (sin mirar al futuro):
      entrenamiento  < inicio_calibracion
      calibración    [inicio_calibracion, inicio_validacion)  -> multiplicadores de la cantidad sugerida
      validación     [inicio_validacion, fin]                 -> costo con el que se decide el reemplazo
    """
    dias_val = dias_val or config.DIAS_VALIDACION
    dias_cal = dias_cal or config.DIAS_CALIBRACION
    mejora_minima = config.MEJORA_MINIMA if mejora_minima is None else mejora_minima
    costos = cat.costos_dict()

    ventas = ventas.copy()
    ventas["fecha"] = pd.to_datetime(ventas["fecha"])
    X = construir_features(ventas[["fecha", "tienda_id", "familia_id", "ventas"]],
                           ventas[["fecha", "tienda_id", "familia_id", "en_promocion", "ventas"]], cat)
    v = particionar(X["fecha"], dias_val, dias_cal)
    tr = X[X["fecha"] < v["inicio_calibracion"]]
    tr = tr[tr["dias_con_historia_28"] >= 7]          # descarta el arranque sin historia
    cal = X[(X["fecha"] >= v["inicio_calibracion"]) & (X["fecha"] < v["inicio_validacion"])]
    val = X[X["fecha"] >= v["inicio_validacion"]]
    log.info("filas: entrenamiento=%d calibracion=%d validacion=%d", len(tr), len(cal), len(val))

    base = Baseline()
    _, m_base = _evaluar(base, cal, val, costos)

    gbm = ModeloGBM().fit(tr, tr["ventas"])
    mult, m_cand = _evaluar(gbm, cal, val, costos)
    candidato = PaqueteModelo(
        modelo=gbm, multiplicadores=mult, familias=list(cat.familias_ids),
        fecha_corte_entrenamiento=v["inicio_calibracion"].date().isoformat(),
        metricas_validacion=m_cand,
    )

    m_vig = None
    motivo_extra = ""
    if vigente is not None:
        try:
            e = vigente.modelo.predict(val)
            s = sugerida_desde(e, val["familia_id"].tolist(), vigente.multiplicadores)
            m_vig = metricas(val["ventas"], e, s, val["familia_id"].tolist(), costos)
        except Exception as exc:  # modelo vigente incompatible -> se reemplaza
            log.exception("no se pudo evaluar el modelo vigente")
            m_vig = None
            vigente = None
            motivo_extra = f" (el vigente no pudo evaluarse: {type(exc).__name__})"

    c_new = m_cand["costo"]
    if not np.isfinite(c_new):
        promover, motivo = False, "El candidato produjo un costo no válido"
    elif vigente is None or m_vig is None:
        # Sin modelo vigente el servicio no puede operar: se promueve el candidato.
        promover = True
        motivo = (f"No hay modelo vigente evaluable; candidato con costo {c_new:,.0f} "
                  f"(baseline {m_base['costo']:,.0f})" + motivo_extra)
    elif c_new >= m_base["costo"]:
        promover = False
        motivo = f"Candidato no supera al baseline: costo {c_new:,.0f} vs {m_base['costo']:,.0f}"
    else:
        c_old = m_vig["costo"]
        umbral = c_old * (1 - mejora_minima)
        promover = c_new < umbral
        motivo = (f"Costo en validación {c_new:,.0f} vs {c_old:,.0f} del vigente "
                  f"({'mejora' if promover else 'no mejora'} al menos {mejora_minima:.0%})")
    motivo = motivo.replace(",", " ")
    return ResultadoEntrenamiento(
        candidato=candidato, metricas_candidato=m_cand, metricas_baseline=m_base,
        metricas_vigente=m_vig, promover=promover, motivo=motivo,
        ventanas={k: t.date().isoformat() for k, t in v.items()}, n_filas=len(tr),
    )
