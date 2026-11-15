"""Construcción de variables para el modelo.

Una sola función (`construir_features`) se usa al entrenar y al predecir, para
que el modelo vea exactamente lo mismo en ambos casos.

Regla de oro: para la fecha objetivo D solo se usan ventas con fecha < D
(las ventas de D se conocen hasta D+1). Las promociones de D sí se conocen
porque vienen en la solicitud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from novamercado.datos import Catalogo

LAGS = [1, 7, 14, 21, 28]
FEATURES = [
    "tienda_id", "familia_idx", "tipo_idx",
    "dow", "dia_mes", "mes", "dias_fin_mes", "quincena",
    "en_promocion",
    "feriado", "comercial", "pago", "medio_dia",
    "feriado_manana", "comercial_manana", "feriado_ayer",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "media_7", "media_28", "media_dow_4", "tendencia", "dias_con_historia_28",
]
CATEGORICAS = ["tienda_id", "familia_idx", "tipo_idx"]
TIPOS_TIENDA = ["express", "hipermercado", "supermercado"]


def _calendario_feriados(cat: Catalogo, fechas: pd.Series, ciudades: pd.Series) -> pd.DataFrame:
    """Banderas de feriado para cada (fecha, ciudad), considerando ámbito nacional o local."""
    fer = cat.feriados
    out = pd.DataFrame({"fecha": fechas.values, "ciudad": ciudades.values})
    for col, tipo, desfase in [
        ("feriado", "feriado", 0), ("comercial", "comercial", 0), ("pago", "pago", 0),
        ("medio_dia", "medio_dia", 0), ("feriado_manana", "feriado", -1),
        ("comercial_manana", "comercial", -1), ("feriado_ayer", "feriado", 1),
    ]:
        f = fer[fer["tipo"] == tipo]
        # desfase -1: el feriado es mañana -> fecha_feriado - 1 día == fecha
        fechas_f = f["fecha"] + pd.Timedelta(days=desfase)
        nac = set(fechas_f[f["ambito"] == "nacional"])
        loc = set(zip(fechas_f[f["ambito"] != "nacional"], f.loc[f["ambito"] != "nacional", "ambito"]))
        out[col] = [
            int(d in nac or (d, c) in loc) for d, c in zip(out["fecha"], out["ciudad"])
        ]
    return out.drop(columns=["fecha", "ciudad"])


def construir_features(hist: pd.DataFrame, objetivos: pd.DataFrame, cat: Catalogo) -> pd.DataFrame:
    """
    hist:      fecha, tienda_id, familia_id, ventas   (ventas observadas)
    objetivos: fecha, tienda_id, familia_id, en_promocion  (filas a predecir)
    Devuelve `objetivos` + columnas FEATURES, en el mismo orden de filas.
    """
    obj = objetivos.copy()
    obj["fecha"] = pd.to_datetime(obj["fecha"])
    obj["tienda_id"] = obj["tienda_id"].astype(int)
    obj = obj.reset_index(drop=True)
    obj["_fila"] = np.arange(len(obj))

    # --- rezagos y medias móviles a partir de una matriz diaria completa
    if len(hist):
        h = hist[["fecha", "tienda_id", "familia_id", "ventas"]].copy()
        h["fecha"] = pd.to_datetime(h["fecha"])
        h["tienda_id"] = h["tienda_id"].astype(int)
        h["ventas"] = h["ventas"].astype(float)
        ancho = h.pivot_table(index="fecha", columns=["tienda_id", "familia_id"],
                              values="ventas", aggfunc="last")
        inicio = min(ancho.index.min(), obj["fecha"].min() - pd.Timedelta(days=29))
        fin = max(ancho.index.max(), obj["fecha"].max())
        ancho = ancho.reindex(pd.date_range(inicio, fin, freq="D"))
        piezas = {}
        for k in LAGS:
            piezas[f"lag_{k}"] = ancho.shift(k)
        piezas["media_7"] = ancho.rolling(7, min_periods=1).mean().shift(1)
        piezas["media_28"] = ancho.rolling(28, min_periods=1).mean().shift(1)
        piezas["dias_con_historia_28"] = ancho.notna().astype(float).rolling(28, min_periods=1).sum().shift(1)
        largo = []
        for nombre, df in piezas.items():
            s = df.stack(["tienda_id", "familia_id"], future_stack=True).rename(nombre)
            largo.append(s)
        largo = pd.concat(largo, axis=1)
        largo.index.names = ["fecha", "tienda_id", "familia_id"]
        largo = largo.reset_index()
        obj = obj.merge(largo, on=["fecha", "tienda_id", "familia_id"], how="left")
    for col in [f"lag_{k}" for k in LAGS] + ["media_7", "media_28", "dias_con_historia_28"]:
        if col not in obj:
            obj[col] = np.nan
    obj["media_dow_4"] = obj[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1, skipna=True)
    obj["tendencia"] = obj["media_7"] / obj["media_28"].replace(0, np.nan)
    obj["dias_con_historia_28"] = obj["dias_con_historia_28"].fillna(0)

    # --- calendario
    f = obj["fecha"]
    obj["dow"] = f.dt.dayofweek
    obj["dia_mes"] = f.dt.day
    obj["mes"] = f.dt.month
    obj["dias_fin_mes"] = (f.dt.days_in_month - f.dt.day).astype(int)
    obj["quincena"] = ((f.dt.day == 15) | (f.dt.day == f.dt.days_in_month)).astype(int)

    # --- tienda y familia
    tiendas = cat.tiendas.set_index("tienda_id")
    obj["tipo_idx"] = obj["tienda_id"].map(tiendas["tipo"]).map(
        {t: i for i, t in enumerate(TIPOS_TIENDA)}).fillna(-1).astype(int)
    obj["familia_idx"] = obj["familia_id"].map(
        {fam: i for i, fam in enumerate(cat.familias_ids)}).fillna(-1).astype(int)
    ciudades = obj["tienda_id"].map(tiendas["ciudad"]).fillna("")
    fer = _calendario_feriados(cat, obj["fecha"], ciudades)
    obj = pd.concat([obj.reset_index(drop=True), fer.reset_index(drop=True)], axis=1)

    obj["en_promocion"] = pd.to_numeric(obj["en_promocion"], errors="coerce")
    obj = obj.sort_values("_fila").drop(columns="_fila").reset_index(drop=True)
    return obj
