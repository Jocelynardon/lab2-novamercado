# NovaMercado: servicio de pronóstico diario de ventas

Laboratorio 2: Product Development, Maestría en Data Science (Universidad Galileo).

Servicio de ML en producción que pronostica, por tienda y familia, la **venta esperada** y la
**cantidad sugerida** para el día. Se defiende de datos defectuosos, registra todo lo que predice
y observa en PostgreSQL y se reentrena con Airflow. El modelo vigente se gestiona en MLflow.

```
                 ┌──────────────┐   alias @champion    ┌─────────────┐
  tiendas ──────▶│  API FastAPI │◀─────(cada 15 s)─────│   MLflow    │
  :8000          │  (api/)      │                      │  :5000      │
                 └──┬────────┬──┘                      └─────▲───────┘
           registra │        │ POST /admin/reentrenar        │ registra versiones
                    ▼        ▼                               │ y mueve el alias
            ┌────────────┐  ┌──────────────────────────┐     │
            │ PostgreSQL │◀─│ Airflow 3 (DAG           │─────┘
            │ producto   │  │ reentrenamiento_novamercado) :8080
            └────────────┘  └──────────────────────────┘
```

## Requisitos

- Docker Desktop (o Docker Engine) con Docker Compose v2.
- Unos 8 GB de RAM para Docker y 10 GB de disco libres.
- Puertos libres: 8000 (API), 5000 (MLflow) y 8080 (Airflow).

## Levantar el producto desde cero

1. Clonar el repositorio (tag de entrega):
   ```bash
   git clone --branch entrega-lab2 <url-del-repo> lab2-novamercado
   cd lab2-novamercado
   ```
2. **Colocar los archivos publicados en la carpeta `data/`** (no se versionan):
   ```
   data/
   ├── tiendas.csv
   ├── familias.csv
   ├── historico_ventas.csv
   ├── feriados.csv
   ├── costos.csv
   ├── catalogo_errores.md        (opcional, solo referencia)
   └── ejemplos/                  (opcional, lo usa scripts/generar_evidencia.sh)
   ```
3. Crear el archivo de configuración:
   ```bash
   cp .env.example .env
   ```
4. Levantar todo:
   ```bash
   docker compose up -d --build
   ```
5. Esperar a que `GET /health` responda `"estado": "ok"`. Toma unos 2–4 minutos sin contar la
   construcción de imágenes:
   ```bash
   curl -s localhost:8000/health
   ```

En el primer arranque, el servicio `bootstrap` crea el esquema, carga el histórico y entrena y
registra el primer modelo (v1, alias `champion`). En los arranques siguientes detecta que todo
existe y no repite el trabajo. Los datos viven en volúmenes de Docker, así que
`docker compose down` seguido de `docker compose up -d` conserva todo. Para empezar de cero,
usar `docker compose down -v`.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `api` | 8000 | FastAPI: contrato del Anexo. Documentación interactiva en `/docs` |
| `product-db` | interno | PostgreSQL 16 del producto (separada de la de Airflow) |
| `mlflow` | 5000 | Tracking + Model Registry (SQLite + artefactos en volumen) |
| `bootstrap` | – | Carga inicial y primer modelo (idempotente; termina y sale) |
| `airflow-apiserver` | 8080 | UI y API REST de Airflow 3 (usuario y clave en `.env`) |
| `airflow-scheduler`, `airflow-dag-processor`, `airflow-db`, `airflow-init` | – | Resto de Airflow |

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/v1/prediccion` | Pronóstico de un día para una tienda |
| POST | `/v1/observaciones` | Ventas reales de un día (idempotente) |
| POST | `/admin/reentrenar` | Dispara el DAG de reentrenamiento → 202 `{"id": ...}` (503 si Airflow no responde) |
| GET | `/admin/reentrenar/{id}` | `en_curso` / `exitoso` / `fallido` + `promovido`, `version`, `motivo` |
| GET | `/modelo` | Modelo vigente (`nombre`, `alias`, `version`) |
| GET | `/health` | `ok`, `degradado` (+`motivo`) o 503 si no hay modelo cargado |

Ejemplo:
```bash
curl -s -X POST localhost:8000/v1/prediccion -H "Content-Type: application/json" \
     -d @data/ejemplos/prediccion_valida.json
```

## Pruebas

Las pruebas no dependen de MLflow, Airflow ni PostgreSQL: usan dobles de prueba.
```bash
docker compose run --rm --no-deps api pytest -q
```

## Evidencia de funcionamiento

```bash
bash scripts/generar_evidencia.sh      # con el producto levantado
```
Guarda solicitudes y respuestas (válidas, corregidas, con advertencia y rechazadas), la
idempotencia de observaciones y un reentrenamiento completo en `evidencia/respuestas_api/`.
Las capturas de MLflow y Airflow están en `evidencia/`.

## Operación

- **Recarga del modelo sin reinicio:** la API consulta el alias `champion` cada 15 s y, si cambió,
  descarga la nueva versión. Así responde con ella en menos de 2 minutos.
- **MLflow caído:** la API sigue respondiendo con el modelo en memoria. Además guarda una copia en
  el volumen `model_cache`, que le permite arrancar aunque MLflow no esté. `/health` pasa a
  `degradado` en menos de 30 s y vuelve a `ok` en menos de 30 s cuando MLflow regresa.
- **Logs:** `docker compose logs -f api`, `docker compose logs -f airflow-scheduler`.

## Estructura

```
api/            FastAPI (endpoints, gestor del modelo, cliente de Airflow)
novamercado/    Paquete compartido: limpieza, features, modelo, entrenamiento, BD, MLflow
dags/           DAG de reentrenamiento (Airflow 3)
docker/         Dockerfiles (versiones fijas)
requirements/   Dependencias fijadas de la API y de Airflow
sql/            Esquema de la BD del producto
tests/          pytest
scripts/        Generación de evidencia
evidencia/      Capturas y respuestas de ejemplo
DECISIONES.md   Decisiones técnicas y de producto
CLAUDE.md       Contexto para el agente de programación
```
