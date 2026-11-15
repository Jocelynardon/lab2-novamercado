# Decisiones técnicas y de producto

## 1. Política ante problemas en los datos

### Principio general

Cada problema se clasifica con una regla general, no caso por caso, porque el servicio recibirá
problemas que no están en el catálogo:

| Acción | Cuándo | Efecto |
|---|---|---|
| **Reparar** | Existe **una sola** interpretación posible del dato | Se corrige y se informa en `calidad` (estado `corregido`) |
| **Advertir** | Se puede responder, pero **hubo que suponer algo** | Se responde y se dice qué se supuso (estado `advertencia`) |
| **Rechazar (422)** | No se puede saber **qué** se está pidiendo: tienda, fecha o la lista misma | `detail` explica el motivo |

Criterio de fondo: **nunca inventar datos sin decirlo**. En una predicción, una suposición
razonable con advertencia es más útil para la tienda que un rechazo, porque igual necesita
preparar inventario. En una observación, un valor dudoso **no se guarda**, porque contaminaría el
entrenamiento.

La lógica vive en `novamercado/normalizacion.py`, que es puro y está probado en
`tests/test_normalizacion.py`. El resultado se valida con los modelos estrictos de Pydantic de
`novamercado/esquemas.py`.

### Aplicación a cada tipo de problema

| Código | Problema | Política | Justificación |
|---|---|---|---|
| E01 | `tienda_id` como texto o con prefijo (`"T05"`, `"5"`, `5.0`) | **Reparar** si contiene un único número entero que existe en el catálogo | Solo hay una lectura posible |
| – | Tienda inexistente, `true`, `5.5`, texto sin número | **Rechazar** | No se sabe para quién es el pronóstico |
| – | Fecha anterior a la apertura de la tienda | **Rechazar** | No hay operación que pronosticar |
| E02 | Familia escrita de otra forma (mayúsculas, tildes, espacios, guiones, nombre largo "Frutas y Verduras") | **Reparar**: se compara la forma normalizada (sin tildes, minúsculas, sin conectores "y/de") contra el id y el nombre del catálogo | Equivalencia exacta tras normalizar |
| – | Familia parecida pero no idéntica (`"lacteo"`) | **Advertir**: se asocia por similitud (≥ 0.8) solo si hay un único candidato | Es probable pero no seguro |
| – | Familia desconocida | **Advertir** y se ignora ese elemento; luego aplica E05 | No se puede pronosticar algo que no existe |
| E03 | `en_promocion` como texto o decimal entero (`"3"`, `2.0`) | **Reparar** | Valor inequívoco |
| – | `en_promocion` con decimales (`2.6`), negativo, vacío o no numérico | **Advertir**: se redondea, o se supone 0 | La promoción solo mueve la demanda; el pronóstico sigue siendo útil |
| E04 | Fecha ISO con hora o zona, `AAAA/MM/DD`, `AAAAMMDD`, `DD/MM/AAAA` con día > 12 | **Reparar** | Una sola lectura posible |
| – | `DD/MM/AAAA` con día y mes ≤ 12 (`05/11/2025`) | **Advertir**: se supone `DD/MM` (convención en Guatemala) | Ambigua |
| – | Fecha imposible (`2025-02-30`), texto libre, número | **Rechazar** | No se sabe qué día se pide |
| – | Fecha posterior al calendario de `feriados.csv` | **Advertir**: se supone día sin feriado | El calendario no cubre esa fecha |
| E05 | Faltan familias que la tienda vende | **Advertir**: se pronostican **todas**, con `en_promocion = 0` para las faltantes | La tienda necesita inventario de todo; omitirlas aumentaría el costo por faltantes |
| E06 | Familia repetida con **los mismos** valores | **Reparar** (se deja una) | No hay conflicto |
| E06 | Familia repetida con **valores distintos** | **Advertir**: se usa el **último** recibido | Criterio "último gana", consistente con E09 |
| E07 | Campo con otro nombre (`"tienda"` en vez de `"tienda_id"`) | **Reparar** si coincide con un sinónimo conocido y es el único candidato | Sin ambigüedad |
| – | Varios campos candidatos o campo obligatorio ausente | **Rechazar** | No se puede decidir |
| E08 | Ventas negativas (devoluciones) | **Advertir** y **no se guarda** ese registro (el resto sí) | No se inventa una venta (ni 0 ni valor absoluto) |
| E09 | Envío repetido idéntico | Idempotente: `registros_guardados = 0` | Clave primaria (fecha, tienda, familia) + `ON CONFLICT ... WHERE distinto` |
| E09 | Reenvío del mismo día con **valores distintos** | Se **actualiza** y se **advierte** mostrando el valor anterior y el nuevo | Se asume que es una corrección de la tienda; queda trazado |
| E10 | `"1,234"` (coma de miles), `"812,0"` (coma decimal), `"1.234,5"` | **Reparar** | El patrón determina el separador |
| – | `"1.234"` (punto que puede ser de miles o decimal) | **Advertir**: se toma como miles (las ventas son unidades enteras) | Ambiguo |
| – | Ventas con decimales (`812.5`) | **Advertir** y se redondea | Las ventas son unidades |
| – | Cuerpo que no es JSON u objeto; `familias`/`ventas` que no es lista | **Rechazar** | No hay solicitud interpretable |

Los campos desconocidos adicionales se ignoran.

Todas las solicitudes se registran con su payload original en la tabla `solicitudes`, incluidas
las rechazadas (`estado = rechazado`, `http_status = 422`). Cada acción de calidad se guarda en
`correcciones`.

## 2. Información que usa el modelo al predecir

Para la fecha objetivo **D** solo se usan ventas con fecha **< D**, porque las ventas de un día
se conocen al día siguiente. La misma función (`novamercado/features.py`) construye las
variables al entrenar y al predecir, para evitar diferencias entre ambos casos. Una prueba
verifica que el día D no se filtra.

| Grupo | Variables |
|---|---|
| Identidad | tienda (categórica), familia (categórica), formato de tienda |
| Calendario | día de la semana, día del mes, mes, días al fin de mes, quincena |
| Calendario comercial | feriado, fecha comercial, fecha de pago, medio día (nacional **o** local según la ciudad de la tienda); feriado o fecha comercial **mañana**; feriado **ayer** |
| Promoción | `en_promocion` del día, que viene en la solicitud |
| Historia reciente | ventas de hace 1, 7, 14, 21 y 28 días; media de 7 y de 28 días; media del mismo día de la semana (4 semanas); tendencia (media 7 / media 28); días con datos |

**Por qué historia reciente y no solo calendario:** el periodo de evaluación puede tener cambios
de demanda. Los rezagos se calculan en cada solicitud a partir de las **observaciones guardadas
en la BD**, así que el pronóstico se ajusta al nivel actual aunque el modelo no se haya
reentrenado. Si faltan observaciones, el modelo (gradient boosting) trata los valores ausentes de
forma nativa.

**Modelos:**
- **Baseline:** promedio del mismo día de la semana en las últimas 4 semanas.
- **ML:** `HistGradientBoostingRegressor` con pérdida **Poisson**, adecuada para conteos no
  negativos.

En la validación inicial (febrero de 2026), el ML redujo el costo de negocio de 134 756 a 88 387
(−34%) y el WAPE de 21.5% a 17.4%.

La promoción de los días observados se toma de la **última solicitud de pronóstico** de ese día,
porque las observaciones no la incluyen.

## 3. Elección de la cantidad sugerida

Es el problema del vendedor de periódicos: con costos asimétricos no conviene pedir la media.
Para cada familia se busca un factor `m` (entre 0.50 y 2.50) que **minimiza el costo real de
`costos.csv`** en una ventana de calibración no vista por el modelo:

```
cantidad_sugerida = round(m_familia × venta_esperada)      (entero ≥ 0)
costo = c_faltante·max(real − sugerido, 0) + c_sobrante·max(sugerido − real, 0)
```

El resultado es coherente con los costos. Ejemplos del primer modelo:

- Licores: faltar cuesta 2.0 y sobrar 0.1, así que `m = 1.48` (pedir de más).
- Panadería: faltar cuesta 1.0 y sobrar 2.5, así que `m = 0.84` (pedir de menos).

Se eligió un factor multiplicativo por familia, y no un cuantil absoluto, porque escala con el
tamaño de la tienda. Además se optimiza directamente la métrica de negocio, discreta y con
redondeo incluido.

## 4. Criterio de reemplazo del modelo

Ventanas temporales sobre histórico + observaciones (sin mirar al futuro):

```
[ ...... entrenamiento ...... ][ calibración 28 d ][ validación 28 d ]
```

1. El candidato se entrena con la ventana de entrenamiento y sus multiplicadores se ajustan en
   calibración.
2. En la **misma ventana de validación** se calcula el **costo de negocio** del candidato, del
   baseline y del modelo vigente (con sus propios multiplicadores).
3. **Se promueve** (se mueve el alias `champion`) solo si:
   - el costo del candidato es menor que el del baseline, **y**
   - es al menos **1% menor** que el del vigente (`MEJORA_MINIMA`, configurable).

   El margen evita cambiar de modelo por ruido.
4. Si no existe modelo vigente (primer arranque), se promueve el candidato.

Garantías:
- Todo candidato se registra como versión en MLflow (con métricas, motivo y linaje del run). Solo
  el alias decide cuál se usa.
- Si el reentrenamiento falla en cualquier paso, el alias no se toca: la ejecución queda como
  `fallido` y el vigente sigue sirviendo.
- La API detecta el cambio de alias en ≤ 15 s y carga la nueva versión sin reiniciarse.

Limitación conocida: si la ventana de validación se solapa con la de calibración del vigente, la
comparación favorece levemente al vigente (sus multiplicadores vieron esos días). Es un sesgo
conservador: en la duda, no se cambia de modelo.

## 5. Otras decisiones

| Tema | Decisión | Alternativa descartada y por qué |
|---|---|---|
| Fechas | Nunca se usa el reloj del sistema para validar fechas; se valida contra el catálogo (apertura, calendario) | Rechazar fechas "futuras": el servicio opera con fechas simuladas |
| Disparo del reentrenamiento | API → REST v2 de Airflow (JWT), con `dag_run_id` propio que sirve como `id` | DAG programado que revisa una cola: más latencia y más ejecuciones vacías |
| Estado del reentrenamiento | Tabla `reentrenamientos` (la escribe el DAG) + estado de Airflow como respaldo si la tarea muere | Solo XCom: más acoplado a la API de Airflow |
| Recarga del modelo | Hilo que consulta el alias cada 15 s y cambia el modelo en memoria de forma atómica | Reiniciar el contenedor: viola el requisito |
| MLflow caído | Modelo en memoria + copia en volumen `model_cache`; `/health` degradado | Fallar: el servicio debe seguir respondiendo |
| Backend de MLflow | SQLite + artefactos en volumen, servidos por el propio MLflow (`--serve-artifacts`) | Otra base Postgres: más piezas para un solo usuario |
| BD del producto | PostgreSQL propia, separada de la de Airflow | Compartir: lo prohíbe el enunciado |
| Primer arranque | Servicio `bootstrap` idempotente (marca `carga_inicial` en tabla `meta`; no entrena si existe `@champion`) | Entrenar en el arranque de la API: lo haría lento y frágil |
| Paquete compartido | `novamercado/` lo usan la API y Airflow (mismas versiones de scikit-learn y pandas) para cargar el mismo `joblib` | Duplicar código: riesgo de diferencias |
| Autenticación de Airflow | SimpleAuthManager con usuario y clave en `.env` | Credenciales en la imagen: prohibido |
| Versiones | Imágenes y dependencias fijadas (`python:3.12.7-slim`, `apache/airflow:3.0.6`, `mlflow:v2.17.2`, `postgres:16.4`) | `latest`: no reproducible |
