#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HOSTS=(
  165.194.19.130
  165.194.19.103
  165.194.19.96
  165.194.19.104
  165.194.19.54
  165.194.19.113
  165.194.19.166
)

if [ "$#" -gt 0 ]; then
  HOSTS=("$@")
else
  HOSTS=("${DEFAULT_HOSTS[@]}")
fi

for h in "${HOSTS[@]}"; do
  echo "===== ${h} ====="
  ssh -o BatchMode=yes -o ConnectTimeout=5 ubuntu@"${h}" '
    set -e
    GOV="N/A"
    if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
      GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    fi

    TEMP="N/A"
    if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
      t=$(cat /sys/class/thermal/thermal_zone0/temp)
      TEMP=$(awk -v t="$t" "BEGIN { printf \"%.1f\", t/1000.0 }")
    fi

    THR="N/A"
    if command -v vcgencmd >/dev/null 2>&1; then
      THR=$(vcgencmd get_throttled 2>/dev/null || true)
    fi

    echo "governor=${GOV} tempC=${TEMP} throttled=${THR}"
    pgrep -af "./dllama worker --port 9998|/home/ubuntu/dllama worker --port 9998|dllama_dist_20260416_run worker --port 9998" || true
    ss -ltnp | grep 9998 || true
  ' || echo "[WARN] SSH failed: ${h}"
  echo

done
