"""Servicio FastAPI de pronóstico de ventas (contrato del Anexo)."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from novamercado import config
from novamercado.datos import Catalogo, cargar_catalogo
from novamercado.esquemas import (
    Calidad, EstadoReentrenamiento, HealthResponse, ModeloResponse, ObservacionCalidad,
    ObservacionResponse, PrediccionFamilia, PrediccionResponse, ReentrenarResponse,
)
from novamercado.features import construir_features
from novamercado.normalizacion import (
    ErrorContrato, ResolutorFamilias, normalizar_observacion, normalizar_prediccion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


def _error(status: int, detalle: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detalle})


def crear_app(repo=None, gestor=None, airflow=None, catalogo: Catalogo | None = None,
              iniciar_hilos: bool = True) -> FastAPI:
    if repo is None:
        from novamercado.db import Repositorio
        repo = Repositorio()
    if gestor is None:
        from api.gestor_modelo import GestorModelo
        gestor = GestorModelo(repo)
    if airflow is None:
        from api.airflow_cliente import ClienteAirflow
        airflow = ClienteAirflow()
    cat = catalogo or cargar_catalogo()
    resolutor = ResolutorFamilias(cat.familias_ids, dict(zip(cat.familias.familia_id, cat.familias.nombre)))
    apertura = {int(r.tienda_id): pd.Timestamp(r.fecha_apertura).date() for r in cat.tiendas.itertuples()}
    reglas = dict(tiendas_validas=cat.tiendas_ids, familias=resolutor, apertura=apertura)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if iniciar_hilos:
            gestor.iniciar()
        yield
        if iniciar_hilos:
            gestor.detener()

    app = FastAPI(title="NovaMercado - Pronóstico diario de ventas", version="1.0", lifespan=lifespan)
    app.state.repo, app.state.gestor, app.state.airflow = repo, gestor, airflow

    async def _leer_json(request: Request):
        try:
            return await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise ErrorContrato("el cuerpo no es un JSON válido")

    def _registrar_rechazo(tipo, payload, detalle):
        try:
            repo.registrar_rechazo(tipo, payload, detalle)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo registrar el rechazo")

    # ------------------------------------------------------------ predicción
    @app.post("/v1/prediccion", response_model=PrediccionResponse)
    async def prediccion(request: Request):
        payload = None
        try:
            payload = await _leer_json(request)
            req, obs = normalizar_prediccion(payload, fecha_max_calendario=cat.ultima_fecha_feriados,
                                             **reglas)
        except ErrorContrato as e:
            _registrar_rechazo("prediccion", payload, e.detail)
            return _error(422, e.detail)

        paquete, version = gestor.actual()
        if paquete is None:
            return _error(503, "todavía no hay un modelo cargado")

        try:
            hist = repo.historia_tienda(req.tienda_id, req.fecha)
        except Exception:  # noqa: BLE001
            log.exception("sin acceso al historial")
            hist = pd.DataFrame(columns=["fecha", "tienda_id", "familia_id", "ventas"])
            obs.append(ObservacionCalidad(campo="historial", accion="advertir",
                                          detalle="no se pudo leer el historial reciente; "
                                                  "pronóstico sin ventas previas"))
        objetivos = pd.DataFrame([{"fecha": pd.Timestamp(req.fecha), "tienda_id": req.tienda_id,
                                   "familia_id": f.familia_id, "en_promocion": f.en_promocion}
                                  for f in req.familias])
        X = construir_features(hist, objetivos, cat)
        esperada, sugerida = paquete.predecir(X)
        preds = [PrediccionFamilia(familia_id=f, venta_esperada=round(float(e), 1),
                                   cantidad_sugerida=int(s))
                 for f, e, s in zip(X["familia_id"], esperada, sugerida)]
        calidad = Calidad.desde(obs)
        try:
            repo.registrar_prediccion(payload, req, calidad, preds, version)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo registrar la predicción")
        return PrediccionResponse(tienda_id=req.tienda_id, fecha=req.fecha, version_modelo=str(version),
                                  predicciones=preds, calidad=calidad)

    # --------------------------------------------------------- observaciones
    @app.post("/v1/observaciones", response_model=ObservacionResponse)
    async def observaciones(request: Request):
        payload = None
        try:
            payload = await _leer_json(request)
            req, obs = normalizar_observacion(payload, **reglas)
        except ErrorContrato as e:
            _registrar_rechazo("observacion", payload, e.detail)
            return _error(422, e.detail)
        try:
            n, obs_final = repo.guardar_observacion(payload, req, obs)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo guardar la observación")
            return _error(503, "la base de datos no está disponible; reintente")
        return ObservacionResponse(registros_guardados=n, calidad=Calidad.desde(obs_final))

    # ------------------------------------------------------- administración
    @app.post("/admin/reentrenar", status_code=202, response_model=ReentrenarResponse)
    async def reentrenar():
        rid = f"reentreno_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        try:
            airflow.disparar(rid)
        except Exception as e:  # noqa: BLE001
            log.warning("no se pudo disparar el reentrenamiento: %s", e)
            return _error(503, f"no se pudo iniciar el reentrenamiento: {e}")
        try:
            repo.crear_reentrenamiento(rid)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo registrar el reentrenamiento")
        return ReentrenarResponse(id=rid)

    @app.get("/admin/reentrenar/{rid}", response_model=EstadoReentrenamiento,
             response_model_exclude_none=True)
    async def estado_reentrenamiento(rid: str):
        fila = repo.obtener_reentrenamiento(rid)
        if fila is None:
            return _error(404, f"no existe la ejecución '{rid}'")
        if fila["estado"] == "en_curso":
            try:
                estado_af = airflow.estado(rid)
            except Exception:  # noqa: BLE001
                estado_af = None
            if estado_af == "failed" or (estado_af == "success"):
                fila = repo.obtener_reentrenamiento(rid)   # el DAG pudo terminar entre lecturas
                if fila["estado"] == "en_curso":
                    motivo = ("la ejecución falló en Airflow; el modelo vigente no cambió"
                              if estado_af == "failed" else "la ejecución terminó sin registrar resultado")
                    repo.actualizar_reentrenamiento(rid, "fallido", False, gestor.version, motivo)
                    fila = repo.obtener_reentrenamiento(rid)
        if fila["estado"] == "en_curso":
            return EstadoReentrenamiento(estado="en_curso")
        return EstadoReentrenamiento(
            estado=fila["estado"], promovido=bool(fila["promovido"]),
            version=str(fila["version"] or gestor.version or ""), motivo=fila["motivo"] or "")

    # ------------------------------------------------------------ consulta
    @app.get("/modelo", response_model=ModeloResponse)
    async def modelo():
        _, version = gestor.actual()
        if version is None:
            return _error(503, "todavía no hay un modelo cargado")
        return ModeloResponse(nombre=config.MODEL_NAME, alias=config.MODEL_ALIAS, version=str(version))

    @app.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
    async def health():
        paquete, version = gestor.actual()
        deps = dict(gestor.estado)
        if paquete is None:
            return JSONResponse(status_code=503, content={
                "estado": "degradado", "motivo": "todavía no hay un modelo cargado", "dependencias": deps})
        caidas = [k for k, ok in deps.items() if not ok]
        if caidas:
            return HealthResponse(estado="degradado", motivo=f"no disponible: {', '.join(caidas)}",
                                  dependencias=deps, version_modelo=version)
        return HealthResponse(estado="ok", dependencias=deps, version_modelo=version)

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = crear_app()
    return app
