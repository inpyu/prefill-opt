#!/usr/bin/env bash
# 클러스터 온도·주파수·부하 기록 (research/16 Phase 1 §11)
#
# 캘리브레이션은 워커가 놀 때 재지만 production 은 워커가 연속 부하를 받는다.
# 열/주파수 저하가 있으면 노드별 캘리브레이션 값이 낙관적이고, 그 오차는
# N 이 커질수록(동시 부하 워커가 늘수록) 커진다.
#
#   bash prefill_bench/monitor_cluster.sh <출력파일> [간격초]
set -u
OUT="${1:?usage: monitor_cluster.sh <out.tsv> [interval]}"
IV="${2:-10}"
HOSTS="113 166 191 103 94 104 54"
echo -e "ts\thost\ttemp_c\tfreq_mhz\tload1\tmax_mhz" > "$OUT"
while true; do
  T=$(date +%s)
  for h in root $HOSTS; do
    if [ "$h" = root ]; then
      c="cat /sys/class/thermal/thermal_zone0/temp; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null||echo 0; cut -d' ' -f1 /proc/loadavg; cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null||echo 0"
      out=$(bash -c "$c" 2>/dev/null)
    else
      out=$(ssh -o BatchMode=yes -o ConnectTimeout=4 ubuntu@165.194.19.$h \
        'cat /sys/class/thermal/thermal_zone0/temp; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null||echo 0; cut -d" " -f1 /proc/loadavg; cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null||echo 0' 2>/dev/null)
    fi
    [ -z "$out" ] && continue
    temp=$(echo "$out" | sed -n 1p); freq=$(echo "$out" | sed -n 2p)
    load=$(echo "$out" | sed -n 3p); thr=$(echo "$out" | sed -n 4p)
    echo -e "$T\t$h\t$(awk "BEGIN{printf \"%.1f\", $temp/1000}")\t$((freq/1000))\t$load\t$((thr/1000))" >> "$OUT"
  done
  sleep "$IV"
done
