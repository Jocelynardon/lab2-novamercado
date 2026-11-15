-- Esquema de la base de datos del producto (idempotente: se puede correr muchas veces)

-- Ventas diarias por tienda y familia: histórico publicado + observaciones recibidas.
CREATE TABLE IF NOT EXISTS ventas (
    fecha         DATE        NOT NULL,
    tienda_id     INTEGER     NOT NULL,
    familia_id    TEXT        NOT NULL,
    ventas        INTEGER     NOT NULL CHECK (ventas >= 0),
    en_promocion  INTEGER,                       -- solo se conoce en el histórico
    origen        TEXT        NOT NULL,          -- 'historico' | 'observacion'
    actualizado   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fecha, tienda_id, familia_id)   -- garantiza idempotencia
);

-- Cada solicitud recibida (incluidas las rechazadas), con su payload original.
CREATE TABLE IF NOT EXISTS solicitudes (
    id             BIGSERIAL PRIMARY KEY,
    tipo           TEXT        NOT NULL,         -- 'prediccion' | 'observacion'
    recibido       TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload        JSONB,
    tienda_id      INTEGER,
    fecha          DATE,
    estado         TEXT        NOT NULL,         -- ok | corregido | advertencia | rechazado
    http_status    INTEGER     NOT NULL,
    detalle        TEXT,
    version_modelo TEXT
);

-- Correcciones / advertencias aplicadas a cada solicitud (calidad.observaciones).
CREATE TABLE IF NOT EXISTS correcciones (
    id           BIGSERIAL PRIMARY KEY,
    solicitud_id BIGINT NOT NULL REFERENCES solicitudes(id),
    campo        TEXT   NOT NULL,
    accion       TEXT   NOT NULL,
    detalle      TEXT   NOT NULL
);

-- Predicciones emitidas, con la versión del modelo que las produjo.
CREATE TABLE IF NOT EXISTS predicciones (
    id                BIGSERIAL PRIMARY KEY,
    solicitud_id      BIGINT      NOT NULL REFERENCES solicitudes(id),
    tienda_id         INTEGER     NOT NULL,
    fecha             DATE        NOT NULL,
    familia_id        TEXT        NOT NULL,
    en_promocion      INTEGER     NOT NULL,
    venta_esperada    DOUBLE PRECISION NOT NULL,
    cantidad_sugerida INTEGER     NOT NULL,
    version_modelo    TEXT        NOT NULL,
    creado            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_predicciones_clave ON predicciones (fecha, tienda_id, familia_id);

-- Ejecuciones de reentrenamiento disparadas por /admin/reentrenar.
CREATE TABLE IF NOT EXISTS reentrenamientos (
    id          TEXT PRIMARY KEY,                -- igual al dag_run_id de Airflow
    creado      TIMESTAMPTZ NOT NULL DEFAULT now(),
    actualizado TIMESTAMPTZ NOT NULL DEFAULT now(),
    estado      TEXT NOT NULL,                   -- en_curso | exitoso | fallido
    promovido   BOOLEAN,
    version     TEXT,
    motivo      TEXT,
    metricas    JSONB
);

-- Marcas de control (p. ej. si la carga inicial ya se hizo).
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT NOT NULL,
    actualizado TIMESTAMPTZ NOT NULL DEFAULT now()
);
