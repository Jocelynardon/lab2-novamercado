"""Modelos Pydantic del contrato (Anexo del enunciado).

Las solicitudes llegan "sucias"; el módulo `normalizacion` las repara y el
resultado se valida con los modelos *estrictos* de abajo antes de usarse.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, StrictInt, StrictStr


class _Estricto(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# ---------- Calidad ----------
class ObservacionCalidad(BaseModel):
    campo: str
    accion: Literal["reparar", "advertir"]
    detalle: str


class Calidad(BaseModel):
    estado: Literal["ok", "corregido", "advertencia"]
    observaciones: list[ObservacionCalidad] = Field(default_factory=list)

    @classmethod
    def desde(cls, observaciones: list[ObservacionCalidad]) -> "Calidad":
        if any(o.accion == "advertir" for o in observaciones):
            estado = "advertencia"
        elif observaciones:
            estado = "corregido"
        else:
            estado = "ok"
        return cls(estado=estado, observaciones=observaciones)


# ---------- Solicitudes normalizadas (contrato estricto) ----------
class ItemPrediccion(_Estricto):
    familia_id: StrictStr
    en_promocion: NonNegativeInt


class PrediccionRequest(_Estricto):
    tienda_id: StrictInt
    fecha: date
    familias: list[ItemPrediccion]


class ItemObservacion(_Estricto):
    familia_id: StrictStr
    ventas: NonNegativeInt


class ObservacionRequest(_Estricto):
    tienda_id: StrictInt
    fecha: date
    ventas: list[ItemObservacion]


# ---------- Respuestas ----------
class PrediccionFamilia(BaseModel):
    familia_id: str
    venta_esperada: float
    cantidad_sugerida: NonNegativeInt


class PrediccionResponse(BaseModel):
    tienda_id: int
    fecha: date
    version_modelo: str
    predicciones: list[PrediccionFamilia]
    calidad: Calidad


class ObservacionResponse(BaseModel):
    registros_guardados: NonNegativeInt
    calidad: Calidad


class ReentrenarResponse(BaseModel):
    id: str


class EstadoReentrenamiento(BaseModel):
    estado: Literal["en_curso", "exitoso", "fallido"]
    promovido: Optional[bool] = None
    version: Optional[str] = None
    motivo: Optional[str] = None


class ModeloResponse(BaseModel):
    nombre: str
    alias: str
    version: str


class HealthResponse(BaseModel):
    estado: Literal["ok", "degradado"]
    motivo: Optional[str] = None
    dependencias: dict[str, bool] = Field(default_factory=dict)
    version_modelo: Optional[str] = None
