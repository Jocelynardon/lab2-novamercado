#!/usr/bin/env bash
# Genera evidencia de funcionamiento en evidencia/ usando los ejemplos publicados.
# Uso (con el producto levantado):  bash scripts/generar_evidencia.sh
set -u
API=${API:-http://localhost:8000}
EJ=${EJEMPLOS:-data/ejemplos}
OUT=evidencia/respuestas_api
mkdir -p "$OUT"

llamar () {  # metodo ruta archivo_salida [archivo_json]
  local metodo=$1 ruta=$2 salida=$3 cuerpo=${4:-}
  {
    echo "### $metodo $ruta ${cuerpo:+(cuerpo: $cuerpo)}"
    if [ -n "$cuerpo" ]; then
      echo "--- solicitud"; cat "$cuerpo"; echo
      echo "--- respuesta"
      curl -s -w '\nHTTP %{http_code}\n' -X "$metodo" "$API$ruta" -H 'Content-Type: application/json' -d @"$cuerpo"
    else
      echo "--- respuesta"
      curl -s -w '\nHTTP %{http_code}\n' -X "$metodo" "$API$ruta"
    fi
  } > "$OUT/$salida.txt"
  echo "  -> $OUT/$salida.txt"
}

echo "Estado y modelo"
llamar GET /health health
llamar GET /modelo modelo

echo "Predicciones"
llamar POST /v1/prediccion prediccion_valida "$EJ/prediccion_valida.json"
for f in "$EJ"/errores/E0[1-7]*.json; do
  llamar POST /v1/prediccion "prediccion_$(basename "$f" .json)" "$f"
done

echo "Casos de rechazo (422)"
TMP=$(mktemp -d)
sed 's/"tienda_id": 5/"tienda_id": 99/' "$EJ/prediccion_valida.json" > "$TMP/tienda_inexistente.json"
sed 's/"2025-11-17"/"2025-02-30"/' "$EJ/prediccion_valida.json" > "$TMP/fecha_invalida.json"
llamar POST /v1/prediccion rechazo_tienda_inexistente "$TMP/tienda_inexistente.json"
llamar POST /v1/prediccion rechazo_fecha_invalida "$TMP/fecha_invalida.json"

echo "Observaciones"
sed 's/2025-11-17/2026-03-01/' "$EJ/observacion_valida.json" > "$TMP/obs_nueva.json"
llamar POST /v1/observaciones observacion_nueva "$TMP/obs_nueva.json"
llamar POST /v1/observaciones observacion_repetida_idempotente "$TMP/obs_nueva.json"
for f in "$EJ"/errores/E0[8-9]*.json "$EJ"/errores/E10*.json; do
  sed 's/2025-11-17/2026-03-02/' "$f" > "$TMP/$(basename "$f")"
  llamar POST /v1/observaciones "observacion_$(basename "$f" .json)" "$TMP/$(basename "$f")"
done

echo "Reentrenamiento"
RESP=$(curl -s -X POST "$API/admin/reentrenar")
echo "$RESP" > "$OUT/reentrenar_disparo.txt"
ID=$(echo "$RESP" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
echo "  id: $ID (esperando a que termine...)"
for i in $(seq 1 60); do
  EST=$(curl -s "$API/admin/reentrenar/$ID")
  case "$EST" in *en_curso*) sleep 10 ;; *) break ;; esac
done
echo "$EST" > "$OUT/reentrenar_estado.txt"
echo "  $EST"
llamar GET /admin/reentrenar/id-que-no-existe reentrenar_404
echo "Listo. Revise la carpeta $OUT"
