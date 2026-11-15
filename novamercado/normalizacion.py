"""Defensa del contrato: reparar, advertir o rechazar datos de entrada.

Reglas generales (no por caso del catálogo):
  * REPARAR  -> la corrección es inequívoca (un solo significado posible).
  * ADVERTIR -> se puede responder, pero hubo que suponer algo; se dice qué.
  * RECHAZAR -> no se puede identificar *qué* se pide (tienda, fecha, lista);
                se lanza ErrorContrato y el endpoint responde 422.

Este módulo es puro (sin BD, sin MLflow, sin FastAPI) para poder probarlo.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any

from novamercado.esquemas import (
    ObservacionCalidad,
    ObservacionRequest,
    PrediccionRequest,
)

REPARAR, ADVERTIR = "reparar", "advertir"


class ErrorContrato(Exception):
    """Solicitud que no se puede atender: se responde 422 con `detail`."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class Bitacora:
    """Acumula las observaciones de calidad de una solicitud."""

    items: list[ObservacionCalidad]

    def __init__(self):
        self.items = []

    def reparar(self, campo: str, detalle: str):
        self.items.append(ObservacionCalidad(campo=campo, accion=REPARAR, detalle=detalle))

    def advertir(self, campo: str, detalle: str):
        self.items.append(ObservacionCalidad(campo=campo, accion=ADVERTIR, detalle=detalle))


# ---------------------------------------------------------------- utilidades
def clave(texto: Any) -> str:
    """'LÁCTEOS ' -> 'lacteos';  'Tienda_ID' -> 'tiendaid'."""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


_CONECTORES = {"y", "e", "de", "del", "la", "las", "el", "los"}


def clave_sin_conectores(texto: Any) -> str:
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in _CONECTORES]
    return "".join(tokens)


def _repr(v: Any) -> str:
    return f"'{v}'" if isinstance(v, str) else repr(v)


# ------------------------------------------------------------ nombres de campos
ALIAS_CAMPOS = {
    "tienda_id": ["tiendaid", "tienda", "idtienda", "store", "storeid", "sucursal",
                  "sucursalid", "codigotienda", "tiendacodigo", "numtienda", "notienda"],
    "fecha": ["fecha", "date", "dia", "fechapronostico", "fechaventa", "fechaventas", "day"],
    "familias": ["familias", "familia", "items", "productos", "lineas", "detalle", "categorias"],
    "familia_id": ["familiaid", "familia", "idfamilia", "family", "familyid", "categoria",
                   "categoriaid", "nombrefamilia"],
    "en_promocion": ["enpromocion", "promocion", "promociones", "promo", "promos",
                     "onpromotion", "productosenpromocion", "cantidadpromocion", "enpromo"],
    "ventas_lista": ["ventas", "observaciones", "items", "detalle", "lineas", "familias"],
    "ventas": ["ventas", "venta", "unidades", "cantidad", "sales", "ventasreales", "unidadesvendidas"],
}


def _extraer_campo(obj: dict, nombre: str, alias_key: str, ruta: str, bit: Bitacora,
                   obligatorio: bool = True) -> tuple[bool, Any]:
    """Busca `nombre` en `obj`; si no está, intenta con nombres equivalentes."""
    if nombre in obj:
        return True, obj[nombre]
    alias = set(ALIAS_CAMPOS[alias_key])
    candidatos = [k for k in obj if clave(k) in alias]
    if len(candidatos) == 1:
        k = candidatos[0]
        bit.reparar(f"{ruta}{nombre}", f"el campo '{k}' se interpretó como '{nombre}'")
        return True, obj[k]
    if len(candidatos) > 1:
        raise ErrorContrato(
            f"{ruta}{nombre}: varios campos podrían ser '{nombre}' ({', '.join(candidatos)}); "
            "no se puede decidir cuál usar")
    if obligatorio:
        raise ErrorContrato(f"falta el campo obligatorio '{ruta}{nombre}'")
    return False, None


# ----------------------------------------------------------------- tienda
def parse_tienda(valor: Any, tiendas_validas: set[int], bit: Bitacora) -> int:
    campo = "tienda_id"
    if isinstance(valor, bool) or valor is None:
        raise ErrorContrato(f"tienda_id inválido: {_repr(valor)}")
    if isinstance(valor, int):
        tid = valor
    elif isinstance(valor, float):
        if not math.isfinite(valor) or valor != int(valor):
            raise ErrorContrato(f"tienda_id inválido: {valor!r}")
        tid = int(valor)
        bit.reparar(campo, f"{valor!r} se interpretó como {tid}")
    elif isinstance(valor, str):
        m = re.fullmatch(r"\s*(?:[A-Za-z]+[\s_\-#]*)?0*(\d+)(?:\.0+)?\s*", valor)
        if not m:
            raise ErrorContrato(f"tienda_id inválido: '{valor}' no contiene un identificador numérico")
        tid = int(m.group(1))
        bit.reparar(campo, f"'{valor}' se interpretó como {tid}")
    else:
        raise ErrorContrato(f"tienda_id inválido: {_repr(valor)}")
    if tid not in tiendas_validas:
        raise ErrorContrato(f"la tienda {tid} no existe en el catálogo")
    return tid


# ------------------------------------------------------------------ fecha
def _fecha_o_error(y: int, m: int, d: int, original: Any) -> date:
    try:
        if not 2000 <= y <= 2100:
            raise ValueError
        return date(y, m, d)
    except ValueError:
        raise ErrorContrato(f"fecha inválida: {_repr(original)}")


def parse_fecha(valor: Any, bit: Bitacora, campo: str = "fecha") -> date:
    if not isinstance(valor, str):
        raise ErrorContrato(f"fecha inválida: {_repr(valor)} (se espera texto AAAA-MM-DD)")
    s = valor.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        f = _fecha_o_error(int(s[:4]), int(s[5:7]), int(s[8:10]), valor)
        if s != valor:
            bit.reparar(campo, f"'{valor}' se interpretó como '{f.isoformat()}'")
        return f
    # AAAA-MM-DD con hora / zona horaria, o AAAA-M-D, o AAAA/MM/DD
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?", s)
    if m:
        f = _fecha_o_error(int(m[1]), int(m[2]), int(m[3]), valor)
        bit.reparar(campo, f"'{valor}' se interpretó como '{f.isoformat()}'")
        return f
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        f = _fecha_o_error(int(m[1]), int(m[2]), int(m[3]), valor)
        bit.reparar(campo, f"'{valor}' se interpretó como '{f.isoformat()}'")
        return f
    # DD/MM/AAAA (convención local); MM/DD/AAAA solo si es la única lectura posible
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:[T\s].*)?", s)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a > 12 or a == b:
            f = _fecha_o_error(y, b, a, valor)
            bit.reparar(campo, f"'{valor}' se interpretó como '{f.isoformat()}' (DD/MM/AAAA)")
        elif b > 12:
            f = _fecha_o_error(y, a, b, valor)
            bit.reparar(campo, f"'{valor}' se interpretó como '{f.isoformat()}' (MM/DD/AAAA, única lectura válida)")
        else:
            f = _fecha_o_error(y, b, a, valor)
            bit.advertir(campo, f"'{valor}' es ambigua; se supuso DD/MM/AAAA -> '{f.isoformat()}'")
        return f
    raise ErrorContrato(f"fecha inválida: '{valor}' (se espera AAAA-MM-DD)")


# ---------------------------------------------------------------- números
@dataclass
class Numero:
    valor: float | None
    reparado: bool = False      # venía como texto/decimal pero sin ambigüedad
    ambiguo: bool = False       # hubo que suponer el separador
    nota: str = ""


def parse_numero(valor: Any) -> Numero:
    if isinstance(valor, bool) or valor is None:
        return Numero(None, nota="vacío o no numérico")
    if isinstance(valor, int):
        return Numero(float(valor))
    if isinstance(valor, float):
        if not math.isfinite(valor):
            return Numero(None, nota="no finito")
        return Numero(valor, reparado=True, nota="decimal")
    if not isinstance(valor, str):
        return Numero(None, nota="tipo no numérico")
    s = valor.strip().replace(" ", "").replace(" ", "")
    if s == "":
        return Numero(None, nota="texto vacío")
    if re.fullmatch(r"[+-]?\d+", s):
        return Numero(float(s), reparado=True, nota="texto")
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", s):
            # '1.234': punto como separador de miles o decimal -> ambiguo
            return Numero(float(s.replace(".", "")), ambiguo=True,
                          nota="se supuso que '.' separa miles")
        return Numero(float(s), reparado=True, nota="texto decimal")
    if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3}){2,}", s):
        return Numero(float(s.replace(".", "")), reparado=True, nota="'.' como separador de miles")
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", s):
        return Numero(float(s.replace(",", "")), reparado=True, nota="',' como separador de miles")
    if re.fullmatch(r"[+-]?\d+,\d{1,2}", s):
        return Numero(float(s.replace(",", ".")), reparado=True, nota="',' como separador decimal")
    if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+,\d+", s):
        return Numero(float(s.replace(".", "").replace(",", ".")), reparado=True,
                      nota="'.' miles y ',' decimal")
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+\.\d+", s):
        return Numero(float(s.replace(",", "")), reparado=True, nota="',' miles y '.' decimal")
    return Numero(None, nota=f"'{valor}' no es un número")


def _entero_no_negativo(valor: Any, campo: str, nombre: str, bit: Bitacora,
                        si_invalido: str) -> int | None:
    """Convierte a entero >= 0. Devuelve None si no se puede (el llamador decide)."""
    n = parse_numero(valor)
    if n.valor is None:
        bit.advertir(campo, f"{nombre} {_repr(valor)} no es válido ({n.nota}); {si_invalido}")
        return None
    v = n.valor
    if v < 0:
        bit.advertir(campo, f"{nombre} negativo ({_repr(valor)}); {si_invalido}")
        return None
    entero = int(round(v))
    if abs(v - entero) > 1e-9:
        bit.advertir(campo, f"{nombre} {_repr(valor)} no es entero; se redondeó a {entero}")
    elif n.ambiguo:
        bit.advertir(campo, f"{nombre} {_repr(valor)}: {n.nota} -> {entero}")
    elif n.reparado:
        bit.reparar(campo, f"{nombre} {_repr(valor)} se interpretó como {entero}")
    return entero


# ----------------------------------------------------------------- familias
class ResolutorFamilias:
    def __init__(self, familias_ids: list[str], nombres: dict[str, str] | None = None):
        self.ids = list(familias_ids)
        self.mapa: dict[str, str] = {}
        for fid in self.ids:
            for k in (clave(fid), clave_sin_conectores(fid)):
                self.mapa[k] = fid
        for fid, nombre in (nombres or {}).items():
            for k in (clave(nombre), clave_sin_conectores(nombre)):
                self.mapa.setdefault(k, fid)

    def resolver(self, valor: Any) -> tuple[str | None, str | None]:
        """Devuelve (familia_id, accion). accion: None=exacta, reparar, advertir."""
        if valor is None or isinstance(valor, (bool, dict, list)):
            return None, None
        texto = str(valor)
        if texto in self.ids:
            return texto, None
        for k in (clave(texto), clave_sin_conectores(texto)):
            if k in self.mapa:
                return self.mapa[k], REPARAR
        cand = difflib.get_close_matches(clave_sin_conectores(texto), list(self.mapa), n=3, cutoff=0.8)
        destinos = {self.mapa[c] for c in cand}
        if len(destinos) == 1:
            return destinos.pop(), ADVERTIR
        return None, None


def _lista(valor: Any, ruta: str) -> list:
    if not isinstance(valor, list):
        raise ErrorContrato(f"'{ruta}' debe ser una lista")
    return valor


# --------------------------------------------------------------- predicción
def normalizar_prediccion(payload: Any, *, tiendas_validas: set[int],
                          familias: ResolutorFamilias,
                          familias_por_tienda: dict[int, list[str]] | None = None,
                          apertura: dict[int, date] | None = None,
                          fecha_max_calendario: date | None = None,
                          ) -> tuple[PrediccionRequest, list[ObservacionCalidad]]:
    if not isinstance(payload, dict):
        raise ErrorContrato("el cuerpo debe ser un objeto JSON")
    bit = Bitacora()
    _, v_tienda = _extraer_campo(payload, "tienda_id", "tienda_id", "", bit)
    _, v_fecha = _extraer_campo(payload, "fecha", "fecha", "", bit)
    _, v_fams = _extraer_campo(payload, "familias", "familias", "", bit)
    tienda = parse_tienda(v_tienda, tiendas_validas, bit)
    fecha = parse_fecha(v_fecha, bit)
    _validar_fecha_tienda(tienda, fecha, apertura, fecha_max_calendario, bit)
    items = _lista(v_fams, "familias")

    vistos: dict[str, tuple[int, int]] = {}  # familia -> (en_promocion, índice)
    for i, item in enumerate(items):
        ruta = f"familias[{i}]."
        if not isinstance(item, dict):
            bit.advertir(f"familias[{i}]", f"elemento {_repr(item)} no es un objeto; se ignoró")
            continue
        try:
            _, v_fid = _extraer_campo(item, "familia_id", "familia_id", ruta, bit)
        except ErrorContrato as e:
            bit.advertir(f"familias[{i}]", f"{e.detail}; se ignoró el elemento")
            continue
        fid, acc = familias.resolver(v_fid)
        if fid is None:
            bit.advertir(f"{ruta}familia_id", f"familia {_repr(v_fid)} no reconocida; se ignoró")
            continue
        if acc == REPARAR:
            bit.reparar(f"{ruta}familia_id", f"{_repr(v_fid)} se interpretó como '{fid}'")
        elif acc == ADVERTIR:
            bit.advertir(f"{ruta}familia_id", f"{_repr(v_fid)} se asoció por similitud con '{fid}'")

        hay, v_promo = _extraer_campo(item, "en_promocion", "en_promocion", ruta, bit, obligatorio=False)
        if not hay:
            bit.advertir(f"{ruta}en_promocion", "no se envió; se supuso 0")
            promo = 0
        else:
            promo = _entero_no_negativo(v_promo, f"{ruta}en_promocion", "en_promocion", bit,
                                        "se supuso 0")
            promo = 0 if promo is None else promo

        if fid in vistos:
            previo, j = vistos[fid]
            if previo == promo:
                bit.reparar(f"{ruta}familia_id",
                            f"'{fid}' repetida con los mismos valores (familias[{j}]); se conservó una")
            else:
                bit.advertir(f"{ruta}familia_id",
                             f"'{fid}' repetida con valores distintos (en_promocion {previo} y {promo}); "
                             "se usó el último recibido")
                vistos[fid] = (promo, i)
            continue
        vistos[fid] = (promo, i)

    esperadas = (familias_por_tienda or {}).get(tienda, familias.ids)
    faltantes = [f for f in esperadas if f not in vistos]
    if faltantes:
        bit.advertir("familias", f"faltan familias que la tienda vende ({', '.join(faltantes)}); "
                                 "se pronosticaron suponiendo en_promocion = 0")
        for f in faltantes:
            vistos[f] = (0, -1)

    req = PrediccionRequest(
        tienda_id=tienda, fecha=fecha,
        familias=[{"familia_id": f, "en_promocion": p} for f, (p, _) in vistos.items()],
    )
    return req, bit.items


def _validar_fecha_tienda(tienda, fecha, apertura, fecha_max_calendario, bit):
    if apertura and tienda in apertura and fecha < apertura[tienda]:
        raise ErrorContrato(
            f"la tienda {tienda} abrió el {apertura[tienda].isoformat()}; no hay ventas para {fecha.isoformat()}")
    if fecha_max_calendario and fecha > fecha_max_calendario:
        bit.advertir("fecha", f"{fecha.isoformat()} está fuera del calendario de feriados "
                              f"(hasta {fecha_max_calendario.isoformat()}); se supuso día sin feriado")


# ------------------------------------------------------------- observaciones
def normalizar_observacion(payload: Any, *, tiendas_validas: set[int],
                           familias: ResolutorFamilias,
                           apertura: dict[int, date] | None = None,
                           ) -> tuple[ObservacionRequest, list[ObservacionCalidad]]:
    if not isinstance(payload, dict):
        raise ErrorContrato("el cuerpo debe ser un objeto JSON")
    bit = Bitacora()
    _, v_tienda = _extraer_campo(payload, "tienda_id", "tienda_id", "", bit)
    _, v_fecha = _extraer_campo(payload, "fecha", "fecha", "", bit)
    _, v_ventas = _extraer_campo(payload, "ventas", "ventas_lista", "", bit)
    tienda = parse_tienda(v_tienda, tiendas_validas, bit)
    fecha = parse_fecha(v_fecha, bit)
    _validar_fecha_tienda(tienda, fecha, apertura, None, bit)
    items = _lista(v_ventas, "ventas")

    vistos: dict[str, tuple[int, int]] = {}
    for i, item in enumerate(items):
        ruta = f"ventas[{i}]."
        if not isinstance(item, dict):
            bit.advertir(f"ventas[{i}]", f"elemento {_repr(item)} no es un objeto; no se guardó")
            continue
        try:
            _, v_fid = _extraer_campo(item, "familia_id", "familia_id", ruta, bit)
            _, v_val = _extraer_campo(item, "ventas", "ventas", ruta, bit)
        except ErrorContrato as e:
            bit.advertir(f"ventas[{i}]", f"{e.detail}; no se guardó")
            continue
        fid, acc = familias.resolver(v_fid)
        if fid is None:
            bit.advertir(f"{ruta}familia_id", f"familia {_repr(v_fid)} no reconocida; no se guardó")
            continue
        if acc == REPARAR:
            bit.reparar(f"{ruta}familia_id", f"{_repr(v_fid)} se interpretó como '{fid}'")
        elif acc == ADVERTIR:
            bit.advertir(f"{ruta}familia_id", f"{_repr(v_fid)} se asoció por similitud con '{fid}'")
        ventas = _entero_no_negativo(v_val, f"{ruta}ventas", "ventas", bit,
                                     "no se guardó (no se inventan ventas)")
        if ventas is None:
            continue
        if fid in vistos:
            previo, j = vistos[fid]
            if previo == ventas:
                bit.reparar(f"{ruta}familia_id",
                            f"'{fid}' repetida con el mismo valor (ventas[{j}]); se guardó una vez")
            else:
                bit.advertir(f"{ruta}familia_id",
                             f"'{fid}' repetida con valores distintos ({previo} y {ventas}); "
                             "se guardó el último recibido")
                vistos[fid] = (ventas, i)
            continue
        vistos[fid] = (ventas, i)

    req = ObservacionRequest(
        tienda_id=tienda, fecha=fecha,
        ventas=[{"familia_id": f, "ventas": v} for f, (v, _) in vistos.items()],
    )
    return req, bit.items
