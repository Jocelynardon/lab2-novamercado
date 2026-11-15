"""Pruebas de features (sin fuga de información) y de la métrica de costos."""
import numpy as np
import pandas as pd

from novamercado.features import construir_features
from novamercado.modelo import ajustar_multiplicadores, costo, sugerida_desde


def test_features_no_usan_el_dia_objetivo(catalogo):
    fechas = pd.date_range("2025-10-01", "2025-11-17")
    hist = pd.DataFrame({"fecha": fechas, "tienda_id": 5, "familia_id": "abarrotes",
                         "ventas": np.arange(len(fechas), dtype=float)})
    obj = pd.DataFrame({"fecha": [pd.Timestamp("2025-11-17")], "tienda_id": [5],
                        "familia_id": ["abarrotes"], "en_promocion": [0]})
    X = construir_features(hist, obj, catalogo)
    ultimo = len(fechas) - 1                       # ventas del 17/11 (no debe verse)
    assert X.loc[0, "lag_1"] == ultimo - 1
    assert X.loc[0, "lag_7"] == ultimo - 7
    assert X.loc[0, "media_7"] == np.mean(np.arange(ultimo - 7, ultimo))


def test_costo_asimetrico():
    costos = {"pan": (1.0, 2.5)}
    assert costo([10], [12], ["pan"], costos)[0] == 5.0      # 2 sobrantes * 2.5
    assert costo([10], [7], ["pan"], costos)[0] == 3.0       # 3 faltantes * 1.0


def test_multiplicador_sigue_a_los_costos():
    rng = np.random.default_rng(0)
    real = rng.poisson(100, 2000)
    esperada = np.full(2000, 100.0)
    fams = ["caro_faltar"] * 1000 + ["caro_sobrar"] * 1000
    costos = {"caro_faltar": (2.0, 0.1), "caro_sobrar": (0.1, 2.0)}
    m = ajustar_multiplicadores(esperada, real, fams, costos)
    assert m["caro_faltar"] > 1.0 > m["caro_sobrar"]
    s = sugerida_desde([100.4], ["x"], {"x": 1.0})
    assert s.dtype.kind == "i" and s[0] == 100
