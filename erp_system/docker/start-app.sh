#!/bin/bash
set -euo pipefail

APP_MODE="${APP_MODE:-backend}"

case "$APP_MODE" in
  hf-space)
    export PORT="${PORT:-7860}"
    export API_URL="${API_URL:-http://127.0.0.1:8000}"
    export ERP_RUNTIME_MODE="api"
    mkdir -p /tmp/nginx
    sed "s/__PORT__/${PORT}/g" /app/docker/nginx.space.template.conf > /tmp/nginx/nginx.conf
    exec supervisord -c /app/docker/supervisord.space.conf
    ;;
  frontend)
    exec bash /app/frontend/start_streamlit.sh
    ;;
  backend|*)
    exec python /app/backend/api.py
    ;;
esac
