#!/usr/bin/env bash
# Vaani Sahayak — safe startup script
#
# Runs both native model servers sequentially to avoid simultaneous MPS memory
# pressure on 16GB Apple Silicon Macs.
#
# Usage: ./start.sh [--no-tts] [--no-docker]
#
# Layout:
#   8001 — Param-1 LLM (native, MPS)
#   8003 — Indic Parler-TTS (native, MPS) — started ONLY if --no-tts is not set
#   8000 — FastAPI backend (Docker)
#   3002 — React frontend (Docker)

set -euo pipefail

NO_TTS=false
NO_DOCKER=false
for arg in "$@"; do
  case $arg in
    --no-tts)    NO_TTS=true ;;
    --no-docker) NO_DOCKER=true ;;
  esac
done

# ── helpers ─────────────────────────────────────────────────────────────────

kill_servers() {
  echo "Stopping any existing model servers..."
  pkill -f "servers/server_param1.py" 2>/dev/null || true
  pkill -f "servers/server_tts.py" 2>/dev/null || true
  sleep 1
}

free_gb() {
  python3 -c "import psutil; print(f'{psutil.virtual_memory().available / 1e9:.1f}')" 2>/dev/null || echo "?"
}

wait_for_url() {
  local url=$1 label=$2
  for i in $(seq 1 24); do
    if curl -s "$url" >/dev/null 2>&1; then
      echo "  ✓ $label ready"
      return 0
    fi
    sleep 5
    echo "  Waiting for $label... (${i}/24)"
  done
  echo "  ✗ $label did not start in time. Check logs."
  return 1
}

# ── preflight ────────────────────────────────────────────────────────────────

echo ""
echo "🇮🇳  Vaani Sahayak — sovereign AI stack"
echo "System RAM free: $(free_gb) GB"
echo ""

kill_servers

# ── Param-1 LLM server ───────────────────────────────────────────────────────

echo "[1/3] Starting Param-1 LLM server (port 8001)..."
python servers/server_param1.py --preload --port 8001 > /tmp/param1.log 2>&1 &
PARAM1_PID=$!
echo "      PID: $PARAM1_PID  |  tail -f /tmp/param1.log"
wait_for_url "http://localhost:8001/v1/models" "Param-1"
echo "      RAM free after Param-1: $(free_gb) GB"
echo ""

# ── TTS server ───────────────────────────────────────────────────────────────

if [ "$NO_TTS" = false ]; then
  FREE=$(python3 -c "import psutil; print(psutil.virtual_memory().available / 1e9)" 2>/dev/null || echo 0)
  MIN_FOR_TTS=5

  if python3 -c "exit(0 if float('$FREE') >= $MIN_FOR_TTS else 1)" 2>/dev/null; then
    echo "[2/3] Starting Indic Parler-TTS server (port 8003)..."
    echo "      (Waiting for TTS model to load — this takes ~20s)"
    python servers/server_tts.py --preload --port 8003 > /tmp/tts.log 2>&1 &
    TTS_PID=$!
    echo "      PID: $TTS_PID  |  tail -f /tmp/tts.log"

    # Poll for model_loaded (TTS takes longer to load)
    for i in $(seq 1 24); do
      sleep 5
      HEALTH=$(curl -s http://localhost:8003/health 2>/dev/null || true)
      if echo "$HEALTH" | grep -q '"model_loaded": true'; then
        echo "  ✓ TTS server ready"
        echo "      RAM free after TTS: $(free_gb) GB"
        break
      fi
      echo "  Waiting for TTS model... (${i}/24)"
    done
    echo ""
  else
    echo "[2/3] Skipping TTS server — only ${FREE}GB RAM free (need ${MIN_FOR_TTS}GB)"
    echo "      Voice output will be disabled for this session."
    echo ""
  fi
else
  echo "[2/3] TTS server skipped (--no-tts flag)"
  echo ""
fi

# ── Docker containers ─────────────────────────────────────────────────────────

if [ "$NO_DOCKER" = false ]; then
  echo "[3/3] Starting Docker containers (backend + frontend)..."
  docker compose up -d backend frontend 2>&1 | grep -E "Started|Running|Error" || true
  echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo "─────────────────────────────────────"
echo "  Vaani Sahayak running"
echo "  Frontend:  http://localhost:3002"
echo "  API:       http://localhost:8000"
echo "  Param-1:   http://localhost:8001"
if [ "$NO_TTS" = false ] && curl -s http://localhost:8003/health | grep -q '"model_loaded": true' 2>/dev/null; then
  echo "  TTS:       http://localhost:8003"
else
  echo "  TTS:       disabled"
fi
echo "  RAM free:  $(free_gb) GB"
echo "─────────────────────────────────────"
echo ""
echo "Logs:"
echo "  tail -f /tmp/param1.log"
echo "  tail -f /tmp/tts.log"
echo "  docker logs vaani-backend-1 -f"
echo ""
echo "To stop: pkill -f servers/server_param1.py; pkill -f servers/server_tts.py; docker compose down"
