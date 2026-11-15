"""Pruebas del contrato HTTP con dobles de prueba (sin MLflow, sin Airflow, sin PostgreSQL)."""
import pytest
from fastapi.testclient import TestClient

from api.main import crear_app
from tests.conftest import (
    AirflowFalso, GestorFalso, RepoFalso, payload_observacion, payload_prediccion,
)


@pytest.fixture
def ctx(catalogo):
    repo, gestor, airflow = RepoFalso(), GestorFalso(), AirflowFalso()
    app = crear_app(repo=repo, gestor=gestor, airflow=airflow, catalogo=catalogo, iniciar_hilos=False)
    return TestClient(app), repo, gestor, airflow


def test_prediccion_valida_cumple_contrato(ctx):
    c, repo, *_ = ctx
    r = c.post("/v1/prediccion", json=payload_prediccion())
    assert r.status_code == 200
    d = r.json()
    assert set(d) >= {"tienda_id", "fecha", "version_modelo", "predicciones", "calidad"}
    assert d["tienda_id"] == 5 and d["fecha"] == "2025-11-17"
    assert isinstance(d["version_modelo"], str) and d["version_modelo"] == "7"
    assert len(d["predicciones"]) == 12
    for p in d["predicciones"]:
        assert set(p) >= {"familia_id", "venta_esperada", "cantidad_sugerida"}
        assert isinstance(p["cantidad_sugerida"], int) and p["cantidad_sugerida"] >= 0
    assert d["calidad"] == {"estado": "ok", "observaciones": []}
    assert repo.solicitudes[-1]["version"] == "7"


def test_prediccion_corregida(ctx):
    c, *_ = ctx
    p = payload_prediccion(tienda_id="T05")
    p["familias"][2]["familia_id"] = "LÁCTEOS"
    d = c.post("/v1/prediccion", json=p).json()
    assert d["calidad"]["estado"] == "corregido"
    obs = d["calidad"]["observaciones"]
    assert all(o["accion"] == "reparar" for o in obs)
    assert {"campo", "accion", "detalle"} <= set(obs[0])
    assert any(o["campo"] == "familias[2].familia_id" for o in obs)


def test_prediccion_con_advertencia(ctx):
    c, *_ = ctx
    p = payload_prediccion()
    p["familias"] = p["familias"][:10]
    d = c.post("/v1/prediccion", json=p).json()
    assert d["calidad"]["estado"] == "advertencia" and len(d["predicciones"]) == 12


@pytest.mark.parametrize("cuerpo", [
    payload_prediccion(tienda_id="T99"),
    payload_prediccion(fecha="no-es-fecha"),
    {"fecha": "2025-11-17", "familias": []},
])
def test_prediccion_rechazada_422_con_detail(ctx, cuerpo):
    c, repo, *_ = ctx
    r = c.post("/v1/prediccion", json=cuerpo)
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    assert repo.solicitudes[-1]["estado"] == "rechazado"


def test_json_invalido_422(ctx):
    c, *_ = ctx
    r = c.post("/v1/prediccion", content=b"{no json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "detail" in r.json()


def test_observacion_idempotente(ctx):
    c, *_ = ctx
    r1 = c.post("/v1/observaciones", json=payload_observacion())
    assert r1.status_code == 200 and r1.json()["registros_guardados"] == 12
    assert r1.json()["calidad"] == {"estado": "ok", "observaciones": []}
    r2 = c.post("/v1/observaciones", json=payload_observacion())
    assert r2.json()["registros_guardados"] == 0


def test_observacion_negativa_y_formato(ctx):
    c, *_ = ctx
    p = payload_observacion()
    p["ventas"][0]["ventas"] = -4
    p["ventas"][1]["ventas"] = "1,292"
    d = c.post("/v1/observaciones", json=p).json()
    assert d["registros_guardados"] == 11 and d["calidad"]["estado"] == "advertencia"


def test_reentrenar_202_y_estado(ctx):
    c, repo, _, airflow = ctx
    r = c.post("/admin/reentrenar")
    assert r.status_code == 202
    rid = r.json()["id"]
    assert isinstance(rid, str)
    assert c.get(f"/admin/reentrenar/{rid}").json() == {"estado": "en_curso"}
    repo.actualizar_reentrenamiento(rid, "exitoso", True, "8", "Costo 1 vs 2")
    d = c.get(f"/admin/reentrenar/{rid}").json()
    assert d == {"estado": "exitoso", "promovido": True, "version": "8", "motivo": "Costo 1 vs 2"}


def test_reentrenar_fallido_en_airflow(ctx):
    c, repo, gestor, airflow = ctx
    rid = c.post("/admin/reentrenar").json()["id"]
    airflow.runs[rid] = "failed"
    d = c.get(f"/admin/reentrenar/{rid}").json()
    assert d["estado"] == "fallido" and d["promovido"] is False and d["version"] == "7"


def test_reentrenar_503_si_airflow_no_esta(catalogo):
    app = crear_app(repo=RepoFalso(), gestor=GestorFalso(), airflow=AirflowFalso(disponible=False),
                    catalogo=catalogo, iniciar_hilos=False)
    assert TestClient(app).post("/admin/reentrenar").status_code == 503


def test_reentrenar_id_desconocido_404(ctx):
    c, *_ = ctx
    assert c.get("/admin/reentrenar/no-existe").status_code == 404


def test_modelo_y_health(ctx):
    c, _, gestor, _ = ctx
    assert c.get("/modelo").json() == {"nombre": "novamercado_ventas", "alias": "champion", "version": "7"}
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["estado"] == "ok"
    gestor.estado["mlflow"] = False
    h = c.get("/health").json()
    assert h["estado"] == "degradado" and "mlflow" in h["motivo"]


def test_health_503_sin_modelo(catalogo):
    app = crear_app(repo=RepoFalso(), gestor=GestorFalso(con_modelo=False), airflow=AirflowFalso(),
                    catalogo=catalogo, iniciar_hilos=False)
    c = TestClient(app)
    assert c.get("/health").status_code == 503
    assert c.post("/v1/prediccion", json=payload_prediccion()).status_code == 503
