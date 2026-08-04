#!/usr/bin/env bash
# 노드 온도/주파수/부하 샘플러. run_breakdown.sh 가 각 run 동안 백그라운드로 띄운다.
# straggler의 원인이 열 스로틀링인지 OS 스케줄링인지 구분하기 위한 데이터.
#
#   bash monitor_node.sh <out.tsv> [interval_sec]
set -euo pipefail

OUT="${1:?usage: monitor_node.sh <out.tsv> [interval]}"
INTERVAL="${2:-1}"

printf 'ts\thost\ttemp_c\tfreq_khz\tthrottled\tload1\n' > "$OUT"

read_temp() {
  if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp
  else
    echo "NA"
  fi
}

read_freq() {
  if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
  else
    echo "NA"
  fi
}

read_throttled() {
  # Raspberry Pi 전용. 없으면 NA.
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || echo "NA"
  else
    echo "NA"
  fi
}

HOST="$(hostname)"
while :; do
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s.%N)" "$HOST" "$(read_temp)" "$(read_freq)" "$(read_throttled)" \
    "$(awk '{print $1}' /proc/loadavg)" >> "$OUT"
  sleep "$INTERVAL"
done
