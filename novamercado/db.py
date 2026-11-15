"""Acceso a la base de datos PostgreSQL del producto."""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import date, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras

from novamercado import config
from novamercado.esquemas import Calidad, ObservacionCalidad

log = logging.getLogger(__name__)
SCHEMA_PATH = os.getenv(
    "SCHEMA_PATH", os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql"))


class Repositorio:
    def __init__(self, dsn: dict | None = None):
        self.dsn = dsn or dict(host=config.DB_HOST, port=config.DB_PORT, user=config.DB_USER,
                               password=config.DB_PASSWORD, dbname=config.DB_NAME)

    @contextmanager
    def conexion(self):
        con = psycopg2.connect(connect_timeout=3, **self.dsn)
        try:
            with con:              # commit / rollback automático
                yield con
        finally:
            con.close()

    # ------------------------------------------------------------ utilidades
    def ping(self) -> bool:
        try:
            with self.conexion() as con, con.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def crear_esquema(self):
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            sql = f.read()
        with self.conexion() as con, con.cursor() as cur:
            cur.execute(sql)

    def meta(self, clave: str) -> str | None:
        with self.conexion() as con, con.cursor() as cur:
            cur.execute("SELECT valor FROM meta WHERE clave = %s", (clave,))
            r = cur.fetchone()
            return r[0] if r else None

    def set_meta(self, clave: str, valor: str):
        with self.conexion() as con, con.cursor() as cur:
            cur.execute("""INSERT INTO meta (clave, valor) VALUES (%s, %s)
                           ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, actualizado = now()""",
                        (clave, valor))

    # ---------------------------------------------------------- carga inicial
    def cargar_historico(self, hist: pd.DataFrame) -> int:
        filas = [(r.fecha.date(), int(r.tienda_id), str(r.familia_id), int(r.ventas),
                  int(r.en_promocion), "historico") for r in hist.itertuples()]
        with self.conexion() as con, con.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO ventas (fecha, tienda_id, familia_id, ventas, en_promocion, origen)
                   VALUES %s ON CONFLICT (fecha, tienda_id, familia_id) DO NOTHING""",
                filas, page_size=5000)
            cur.execute("SELECT count(*) FROM ventas")
            return cur.fetchone()[0]

    # ------------------------------------------------------------- solicitudes
    def _insertar_solicitud(self, cur, tipo, payload, tienda, fecha, estado, http_status,
                            detalle=None, version=None, observaciones=()) -> int:
        cur.execute(
            """INSERT INTO solicitudes (tipo, payload, tienda_id, fecha, estado, http_status, detalle, version_modelo)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (tipo, json.dumps(payload, ensure_ascii=False, default=str), tienda, fecha, estado,
             http_status, detalle, version))
        sid = cur.fetchone()[0]
        if observaciones:
            psycopg2.extras.execute_values(
                cur, "INSERT INTO correcciones (solicitud_id, campo, accion, detalle) VALUES %s",
                [(sid, o.campo, o.accion, o.detalle) for o in observaciones])
        return sid

    def registrar_rechazo(self, tipo: str, payload, detalle: str):
        with self.conexion() as con, con.cursor() as cur:
            self._insertar_solicitud(cur, tipo, payload, None, None, "rechazado", 422, detalle)

    def registrar_prediccion(self, payload, req, calidad, predicciones, version: str) -> int:
        promos = {f.familia_id: f.en_promocion for f in req.familias}
        with self.conexion() as con, con.cursor() as cur:
            sid = self._insertar_solicitud(cur, "prediccion", payload, req.tienda_id, req.fecha,
                                           calidad.estado, 200, None, version, calidad.observaciones)
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO predicciones (solicitud_id, tienda_id, fecha, familia_id, en_promocion,
                                             venta_esperada, cantidad_sugerida, version_modelo) VALUES %s""",
                [(sid, req.tienda_id, req.fecha, p.familia_id, promos.get(p.familia_id, 0),
                  p.venta_esperada, p.cantidad_sugerida, version) for p in predicciones])
            return sid

    def guardar_observacion(self, payload, req, observaciones) -> tuple[int, list]:
        """Upsert idempotente. Devuelve (registros nuevos o modificados, observaciones de calidad
        completas: las recibidas + una advertencia por cada valor que reemplazó a otro distinto)."""
        with self.conexion() as con, con.cursor() as cur:
            cur.execute("SELECT familia_id, ventas FROM ventas WHERE fecha = %s AND tienda_id = %s",
                        (req.fecha, req.tienda_id))
            previos = dict(cur.fetchall())
            n = 0
            cambios = []
            for item in req.ventas:
                cur.execute(
                    """INSERT INTO ventas (fecha, tienda_id, familia_id, ventas, origen)
                       VALUES (%s, %s, %s, %s, 'observacion')
                       ON CONFLICT (fecha, tienda_id, familia_id) DO UPDATE
                         SET ventas = EXCLUDED.ventas, origen = 'observacion', actualizado = now()
                         WHERE ventas.ventas IS DISTINCT FROM EXCLUDED.ventas""",
                    (req.fecha, req.tienda_id, item.familia_id, item.ventas))
                n += cur.rowcount
                antes = previos.get(item.familia_id)
                if antes is not None and antes != item.ventas:
                    cambios.append((item.familia_id, antes, item.ventas))
            observaciones = list(observaciones) + [
                ObservacionCalidad(campo=f"ventas.{fam}", accion="advertir",
                                   detalle=f"ya existía {antes} para {req.fecha.isoformat()}; "
                                           f"se reemplazó por {ahora} (último envío)")
                for fam, antes, ahora in cambios]
            estado = Calidad.desde(observaciones).estado
            self._insertar_solicitud(cur, "observacion", payload, req.tienda_id, req.fecha,
                                     estado, 200, f"registros_guardados={n}", None, observaciones)
            return n, observaciones

    # ---------------------------------------------------------------- lectura
    def historia_tienda(self, tienda_id: int, fecha: date, dias: int = 35) -> pd.DataFrame:
        """Ventas de la tienda en los `dias` previos a `fecha` (sin incluir `fecha`)."""
        with self.conexion() as con, con.cursor() as cur:
            cur.execute(
                """SELECT fecha, tienda_id, familia_id, ventas FROM ventas
                   WHERE tienda_id = %s AND fecha >= %s AND fecha < %s""",
                (tienda_id, fecha - timedelta(days=dias), fecha))
            filas = cur.fetchall()
        return pd.DataFrame(filas, columns=["fecha", "tienda_id", "familia_id", "ventas"])

    def ventas_entrenamiento(self) -> pd.DataFrame:
        """Histórico + observaciones. La promoción de días observados se toma de la última
        solicitud de pronóstico de ese día (las observaciones no la traen)."""
        with self.conexion() as con, con.cursor() as cur:
            cur.execute(
                """SELECT v.fecha, v.tienda_id, v.familia_id, v.ventas,
                          COALESCE(v.en_promocion, p.en_promocion) AS en_promocion
                   FROM ventas v
                   LEFT JOIN LATERAL (
                       SELECT en_promocion FROM predicciones p
                       WHERE p.fecha = v.fecha AND p.tienda_id = v.tienda_id AND p.familia_id = v.familia_id
                       ORDER BY p.id DESC LIMIT 1) p ON TRUE""")
            filas = cur.fetchall()
        df = pd.DataFrame(filas, columns=["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"])
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["en_promocion"] = pd.to_numeric(df["en_promocion"], errors="coerce")
        return df

    # -------------------------------------------------------- reentrenamientos
    def crear_reentrenamiento(self, rid: str):
        with self.conexion() as con, con.cursor() as cur:
            cur.execute("INSERT INTO reentrenamientos (id, estado) VALUES (%s, 'en_curso') "
                        "ON CONFLICT (id) DO NOTHING", (rid,))

    def actualizar_reentrenamiento(self, rid: str, estado: str, promovido=None, version=None,
                                   motivo=None, metricas=None):
        with self.conexion() as con, con.cursor() as cur:
            cur.execute(
                """INSERT INTO reentrenamientos (id, estado, promovido, version, motivo, metricas)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET estado = EXCLUDED.estado, promovido = EXCLUDED.promovido,
                       version = EXCLUDED.version, motivo = EXCLUDED.motivo,
                       metricas = EXCLUDED.metricas, actualizado = now()""",
                (rid, estado, promovido, version, motivo,
                 json.dumps(metricas, default=str) if metricas is not None else None))

    def obtener_reentrenamiento(self, rid: str) -> dict | None:
        with self.conexion() as con, con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, estado, promovido, version, motivo FROM reentrenamientos WHERE id = %s",
                        (rid,))
            r = cur.fetchone()
            return dict(r) if r else None
