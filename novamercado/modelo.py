"""Modelos (baseline y ML), métrica de negocio y cantidad sugerida."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from novamercado.features import CATEGORICAS, FEATURES

MULTIPLICADORES = np.round(np.arange(0.50, 2.501, 0.01), 2)


# ------------------------------------------------------------ métrica de negocio
def costo(real, sugerido, familias, costos: dict[str, tuple[float, float]]) -> np.ndarray:
    """costo = c_falt * max(real - sug, 0) + c_sobr * max(sug - real, 0), fila a fila."""
    real = np.asarray(real, float)
    sug = np.asarray(sugerido, float)
    cf = np.array([costos[f][0] for f in familias])
    cs = np.array([costos[f][1] for f in familias])
    return cf * np.maximum(real - sug, 0) + cs * np.maximum(sug - real, 0)


def metricas(real, esperada, sugerida, familias, costos) -> dict[str, float]:
    real = np.asarray(real, float)
    esperada = np.asarray(esperada, float)
    err = esperada - real
    return {
        "costo": float(costo(real, sugerida, familias, costos).sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / max(real.sum(), 1.0)),
        "sesgo": float(err.sum() / max(real.sum(), 1.0)),
    }


def sugerida_desde(esperada, familias, multiplicadores: dict[str, float]) -> np.ndarray:
    m = np.array([multiplicadores.get(f, 1.0) for f in familias])
    return np.maximum(np.round(np.asarray(esperada, float) * m), 0).astype(int)


def ajustar_multiplicadores(esperada, real, familias, costos) -> dict[str, float]:
    """Para cada familia, el factor m que minimiza el costo de sugerir round(m * esperada).

    Es la solución empírica del problema del vendedor de periódicos: con costos
    asimétricos conviene pedir por encima (o por debajo) de la media.
    """
    df = pd.DataFrame({"e": np.asarray(esperada, float), "r": np.asarray(real, float),
                       "f": list(familias)})
    out = {}
    for fam, g in df.groupby("f"):
        cf, cs = costos[fam]
        sug = np.round(np.outer(g["e"].to_numpy(), MULTIPLICADORES))
        r = g["r"].to_numpy()[:, None]
        c = (cf * np.maximum(r - sug, 0) + cs * np.maximum(sug - r, 0)).sum(axis=0)
        out[fam] = float(MULTIPLICADORES[int(np.argmin(c))])
    return out


# ---------------------------------------------------------------- modelos
@dataclass
class Baseline:
    """Promedio del mismo día de la semana en las 4 semanas previas."""

    nombre: str = "baseline_dow4"

    def fit(self, X: pd.DataFrame, y) -> "Baseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = X["media_dow_4"].fillna(X["media_28"]).fillna(X["lag_1"]).fillna(0.0)
        return np.maximum(p.to_numpy(float), 0.0)


@dataclass
class ModeloGBM:
    """Gradient boosting con pérdida Poisson (ventas son conteos no negativos)."""

    nombre: str = "hgb_poisson"
    params: dict = field(default_factory=lambda: dict(
        loss="poisson", learning_rate=0.06, max_iter=500, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=42,
    ))
    estimador: HistGradientBoostingRegressor | None = None

    def fit(self, X: pd.DataFrame, y) -> "ModeloGBM":
        self.estimador = HistGradientBoostingRegressor(
            categorical_features=[FEATURES.index(c) for c in CATEGORICAS], **self.params)
        self.estimador.fit(X[FEATURES].to_numpy(float), np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.estimador.predict(X[FEATURES].to_numpy(float)), 0.0)


@dataclass
class PaqueteModelo:
    """Lo que se registra en MLflow y carga la API: modelo + multiplicadores + metadatos."""

    modelo: object
    multiplicadores: dict[str, float]
    familias: list[str]
    fecha_corte_entrenamiento: str
    metricas_validacion: dict[str, float]
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def predecir(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        esperada = self.modelo.predict(X)
        sugerida = sugerida_desde(esperada, X["familia_id"].tolist(), self.multiplicadores)
        return esperada, sugerida
