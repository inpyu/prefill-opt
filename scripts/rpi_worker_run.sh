#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HOSTS=(
  165.194.19.103
  165.194.19.94
  165.194.19.104
  165.194.19.54
  165.194.19.113
  165.194.19.166
  165.194.19.96
)

BIN_PATH="${BIN_PATH:-/home/ubuntu/dllama}"
PORT="${PORT:-9998}"
NTHREADS="${NTHREADS:-4}"
LOG_PATH="${LOG_PATH:-/home/ubuntu/dllama_worker_9998.log}"
CPUSET="${CPUSET:-0-3}"
STAGE_TIMING="${STAGE_TIMING:-0}"

if [ "$#" -gt 0 ]; then
  HOSTS=("$@")
else
  HOSTS=("${DEFAULT_HOSTS[@]}")
fi

for h in "${HOSTS[@]}"; do
  echo "===== deploy/start ${h} ====="
  ssh -o BatchMode=yes -o ConnectTimeout=5 ubuntu@"${h}" "
    set -e

    # 1) stop old workers first (worker-first rule)
    pkill -x dllama || true
    sleep 1

    # 2) best-effort governor performance
    if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
      echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null || true
    fi

    # 3) start worker pinned to first 4 cores
    nohup taskset -c ${CPUSET} ${BIN_PATH} worker \
      --port ${PORT} \
      --nthreads ${NTHREADS} \
      --stage-timing ${STAGE_TIMING} \
      > ${LOG_PATH} 2>&1 &

    sleep 1
    pgrep -af '${BIN_PATH} worker --port ${PORT}' || true
    ss -ltnp | grep ${PORT} || true
  " || echo "[WARN] SSH failed: ${h}"
  echo

done
