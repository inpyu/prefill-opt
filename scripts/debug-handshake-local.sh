#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

N_WORKERS="${1:-7}"
BASE_PORT="${2:-12001}"
THREADS="${3:-2}"

pkill -f "dllama worker" || true
sleep 1

WORKERS=()
for ((i=0; i<N_WORKERS; i++)); do
  port=$((BASE_PORT + i))
  WORKERS+=("localhost:${port}")
  ./dllama worker --port "${port}" --nthreads "${THREADS}" > "worker_${port}.log" 2>&1 &
done

sleep 2

WORKER_ARGS=""
for w in "${WORKERS[@]}"; do
  WORKER_ARGS+=" ${w}"
done

set +e
timeout 120 ./dllama inference \
  --model dllama_model_original_q40.m \
  --tokenizer dllama_tokenizer_llama3_8B.t \
  --buffer-float-type q80 \
  --prompt "Hi" \
  --steps 1 \
  --nthreads "${THREADS}" \
  --collective auto \
  --workers ${WORKER_ARGS} > root_handshake.log 2>&1
EC=$?
set -e

pkill -f "dllama worker" || true
sleep 1

echo "==== ROOT HANDSHAKE LOG ===="
grep -E "Root expects|Socket\[[0-9]+\]: connecting|waiting initial ACK|connected|broadcasting root-ready|Network is initialized|Critical error|NET_TIMEOUT|Connection timeout" root_handshake.log || true
echo

for ((i=0; i<N_WORKERS; i++)); do
  port=$((BASE_PORT + i))
  echo "==== WORKER ${port} HANDSHAKE LOG ===="
  grep -E "The root node has connected|nNodes:|NodeIndex:|accepted root node|expecting|config received|waiting root-ready|root-ready ACK received|peer node=|Socket\[[0-9]+\]: wait|Socket\[[0-9]+\]: connecting|Socket\[[0-9]+\]: accepted|Socket\[[0-9]+\]: connected|Network is initialized|NET_TIMEOUT|Critical error" "worker_${port}.log" || true
  echo
done

exit "$EC"
