"""Pruebas de la política ante datos defectuosos (catálogo E01-E10 y casos generales)."""
import pytest

from novamercado.normalizacion import (
    ErrorContrato, normalizar_observacion, normalizar_prediccion, parse_fecha, parse_numero, Bitacora,
)
from tests.conftest import payload_observacion, payload_prediccion


def acciones(obs):
    return [o.accion for o in obs]


def test_prediccion_valida_sin_observaciones(reglas):
    req, obs = normalizar_prediccion(payload_prediccion(), **reglas)
    assert req.tienda_id == 5 and len(req.familias) == 12 and obs == []


@pytest.mark.parametrize("valor", ["T05", "5", "tienda 5", 5.0, "T-005"])
def test_e01_tienda_texto_se_repara(reglas, valor):
    req, obs = normalizar_prediccion(payload_prediccion(tienda_id=valor), **reglas)
    assert req.tienda_id == 5 and acciones(obs) == ["reparar"]


@pytest.mark.parametrize("valor", ["T99", 0, "abc", None, True, 5.5])
def test_e01_tienda_invalida_se_rechaza(reglas, valor):
    with pytest.raises(ErrorContrato):
        normalizar_prediccion(payload_prediccion(tienda_id=valor), **reglas)


def test_e02_familia_otra_forma(reglas):
    p = payload_prediccion()
    p["familias"][2]["familia_id"] = "LÁCTEOS"
    p["familias"][5]["familia_id"] = "Frutas y Verduras"
    p["familias"][0]["familia_id"] = "abarrotes "
    req, obs = normalizar_prediccion(p, **reglas)
    ids = {f.familia_id for f in req.familias}
    assert {"lacteos", "frutas_verduras", "abarrotes"} <= ids and len(ids) == 12
    assert acciones(obs) == ["reparar"] * 3
    assert "'LÁCTEOS' se interpretó como 'lacteos'" in obs[1].detalle or \
        any("LÁCTEOS" in o.detalle for o in obs)


def test_familia_parecida_se_advierte_y_desconocida_se_ignora(reglas):
    p = payload_prediccion()
    p["familias"][2]["familia_id"] = "lacteo"
    p["familias"].append({"familia_id": "ferreteria", "en_promocion": 0})
    req, obs = normalizar_prediccion(p, **reglas)
    assert len(req.familias) == 12
    assert "advertir" in acciones(obs)


def test_e03_promocion_texto_o_decimal(reglas):
    p = payload_prediccion()
    p["familias"][1]["en_promocion"] = "3"
    p["familias"][2]["en_promocion"] = 2.0
    req, obs = normalizar_prediccion(p, **reglas)
    promos = {f.familia_id: f.en_promocion for f in req.familias}
    assert promos["bebidas"] == 3 and promos["lacteos"] == 2
    assert acciones(obs) == ["reparar", "reparar"]


@pytest.mark.parametrize("valor,esperado", [(-2, 0), ("muchas", 0), (2.6, 3), (None, 0)])
def test_promocion_dudosa_se_advierte(reglas, valor, esperado):
    p = payload_prediccion()
    p["familias"][0]["en_promocion"] = valor
    req, obs = normalizar_prediccion(p, **reglas)
    assert req.familias[0].en_promocion == esperado and acciones(obs) == ["advertir"]


@pytest.mark.parametrize("texto,esperado,accion", [
    ("2025-11-17T00:00:00", "2025-11-17", "reparar"),
    ("17/11/2025", "2025-11-17", "reparar"),
    ("2025/11/17", "2025-11-17", "reparar"),
    ("20251117", "2025-11-17", "reparar"),
    ("11/17/2025", "2025-11-17", "reparar"),
    ("05/11/2025", "2025-11-05", "advertir"),
])
def test_e04_fecha_otro_formato(texto, esperado, accion):
    b = Bitacora()
    assert parse_fecha(texto, b).isoformat() == esperado
    assert acciones(b.items) == [accion]


@pytest.mark.parametrize("texto", ["2025-02-30", "mañana", "", 20251117, "17/13/2025"])
def test_fecha_invalida_se_rechaza(texto):
    with pytest.raises(ErrorContrato):
        parse_fecha(texto, Bitacora())


def test_e05_faltan_familias_se_completan_con_advertencia(reglas):
    p = payload_prediccion()
    p["familias"] = p["familias"][:10]
    req, obs = normalizar_prediccion(p, **reglas)
    assert len(req.familias) == 12 and acciones(obs) == ["advertir"]


def test_e06_familia_repetida(reglas):
    p = payload_prediccion()
    p["familias"].append({"familia_id": "abarrotes", "en_promocion": 3})
    req, obs = normalizar_prediccion(p, **reglas)
    assert len(req.familias) == 12
    assert {f.familia_id: f.en_promocion for f in req.familias}["abarrotes"] == 3
    assert acciones(obs) == ["advertir"]
    p = payload_prediccion()
    p["familias"].append({"familia_id": "abarrotes", "en_promocion": 0})
    _, obs = normalizar_prediccion(p, **reglas)
    assert acciones(obs) == ["reparar"]


def test_e07_campo_renombrado(reglas):
    p = payload_prediccion()
    p["tienda"] = p.pop("tienda_id")
    req, obs = normalizar_prediccion(p, **reglas)
    assert req.tienda_id == 5 and acciones(obs) == ["reparar"]


def test_falta_campo_obligatorio_se_rechaza(reglas):
    p = payload_prediccion()
    del p["fecha"]
    with pytest.raises(ErrorContrato):
        normalizar_prediccion(p, **reglas)


def test_tienda_antes_de_apertura_se_rechaza(reglas):
    with pytest.raises(ErrorContrato):
        normalizar_prediccion(payload_prediccion(tienda_id=10, fecha="2024-06-01"), **reglas)


def test_e08_ventas_negativas_no_se_guardan(reglas):
    p = payload_observacion()
    p["ventas"][3]["ventas"] = -4
    req, obs = normalizar_observacion(p, **reglas)
    assert len(req.ventas) == 11 and acciones(obs) == ["advertir"]


@pytest.mark.parametrize("texto,valor", [("1,234", 1234), ("812,0", 812), ("1.234,5", 1234.5),
                                         ("1,234.0", 1234), (" 50 ", 50)])
def test_e10_numeros_con_formato(texto, valor):
    assert parse_numero(texto).valor == valor


def test_e10_observacion_con_formato(reglas):
    p = payload_observacion()
    p["ventas"][0]["ventas"] = "1,234"
    p["ventas"][1]["ventas"] = "812,0"
    req, obs = normalizar_observacion(p, **reglas)
    v = {x.familia_id: x.ventas for x in req.ventas}
    assert v["abarrotes"] == 1234 and v["bebidas"] == 812 and acciones(obs) == ["reparar", "reparar"]


def test_cuerpo_no_objeto_se_rechaza(reglas):
    with pytest.raises(ErrorContrato):
        normalizar_prediccion([1, 2], **reglas)
    with pytest.raises(ErrorContrato):
        normalizar_prediccion(payload_prediccion(familias="abarrotes"), **reglas)
