#!/usr/bin/env bash
# EXP-1: prefill 비용 분해 스윕.
#   S(프롬프트 길이) × N(노드 수) × W(대역폭) × repeat
#
# 운영 규칙 (기존 프로젝트 관행 준수):
#   - worker-first: 워커를 root보다 먼저 기동한다
#   - 로그는 bench_prefill/<RUN_ID>/ 아래에만 쌓는다 (bench_logs/ 는 PiPP용, 건드리지 않음)
#   - run 사이 COOLDOWN_SEC 만큼 쉬어 열 조건을 정렬한다
#
#   bash run_breakdown.sh                 # 전체 스윕
#   DRY_RUN=1 bash run_breakdown.sh       # 실행 계획만 출력
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-exp1_breakdown_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/thermal"

MANIFEST="$OUT_DIR/manifest.tsv"
printf 'run\tseqlen\tnodes\tbw_mbit\trep\tlog\tstatus\n' > "$MANIFEST"

log_meta() {
  {
    echo "run_id: $RUN_ID"
    echo "date: $(date -Is)"
    echo "model: $(readlink -f "$MODEL")"
    echo "tokenizer: $(readlink -f "$TOKENIZER")"
    echo "bin: $BIN"
    echo "git: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo NA)"
    echo "nthreads: $NTHREADS  pp_size: $PP_SIZE  sp_size: $SP_SIZE  steps: $STEPS"
    echo "sweep_seqlens: ${SWEEP_SEQLENS[*]}"
    echo "sweep_nodes: ${SWEEP_NODES[*]}"
    echo "sweep_bw_mbit: ${SWEEP_BW_MBIT[*]}"
    echo "repeats: $REPEATS"
  } > "$OUT_DIR/meta.txt"
  cat "$OUT_DIR/meta.txt"
}

start_workers() {
  local n="$1"
  [ "$n" -le 1 ] && return 0
  local hosts=("${ALL_WORKERS[@]:0:$((n - 1))}")
  echo "  [worker-first] 기동: ${hosts[*]}"
  [ "$DRY_RUN" = "1" ] && return 0
  BIN_PATH="/home/ubuntu/dllama" PORT="$WORKER_PORT" NTHREADS="$NTHREADS" \
    bash "$REPO_DIR/scripts/rpi_worker_run.sh" "${hosts[@]}" >/dev/null
  sleep 3
}

start_monitors() {
  local tag="$1" n="$2"
  [ "$DRY_RUN" = "1" ] && return 0
  bash "$HERE/monitor_node.sh" "$OUT_DIR/thermal/${tag}_root.tsv" 1 &
  echo $! > "$OUT_DIR/thermal/${tag}_root.pid"
  local i=0
  while [ "$i" -lt $((n - 1)) ]; do
    local h="${ALL_WORKERS[$i]}"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_USER@$h" \
      "nohup bash -c 'while :; do printf \"%s\t%s\t%s\t%s\n\" \
        \$(date +%s.%N) \$(hostname) \
        \$(awk \"{printf \\\"%.1f\\\", \\\$1/1000}\" /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo NA) \
        \$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA); \
        sleep 1; done' > /tmp/prefill_mon_${tag}.tsv 2>/dev/null &" >/dev/null 2>&1 || true
    i=$((i + 1))
  done
}

stop_monitors() {
  local tag="$1" n="$2"
  [ "$DRY_RUN" = "1" ] && return 0
  if [ -f "$OUT_DIR/thermal/${tag}_root.pid" ]; then
    kill "$(cat "$OUT_DIR/thermal/${tag}_root.pid")" 2>/dev/null || true
    rm -f "$OUT_DIR/thermal/${tag}_root.pid"
  fi
  local i=0
  while [ "$i" -lt $((n - 1)) ]; do
    local h="${ALL_WORKERS[$i]}"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_USER@$h" \
      "pkill -f 'prefill_mon_${tag}' 2>/dev/null; cat /tmp/prefill_mon_${tag}.tsv 2>/dev/null" \
      > "$OUT_DIR/thermal/${tag}_${h}.tsv" 2>/dev/null || true
    i=$((i + 1))
  done
}

run_one() {
  local seqlen="$1" nodes="$2" bw="$3" rep="$4"
  local tag="s${seqlen}_n${nodes}_bw${bw}_r${rep}"
  local log="$OUT_DIR/logs/${tag}.log"
  local prompt_file="$PROMPT_DIR/prompt_${seqlen}.txt"

  if [ ! -f "$prompt_file" ]; then
    echo "  [skip] 프롬프트 없음: $prompt_file  (make_prompts.py 먼저 실행)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$seqlen" "$nodes" "$bw" "$rep" "$log" "no_prompt" >> "$MANIFEST"
    return 0
  fi

  local workers_arg=()
  local w
  w="$(workers_for "$nodes")"
  [ -n "$w" ] && workers_arg=(--workers $w)

  echo "== $tag =="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  $BIN inference $(common_args) --prompt @${prompt_file} --steps $STEPS ${workers_arg[*]:-}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$seqlen" "$nodes" "$bw" "$rep" "$log" "dry" >> "$MANIFEST"
    return 0
  fi

  start_monitors "$tag" "$nodes"
  local status="ok"
  # shellcheck disable=SC2046,SC2086
  timeout "$RUN_TIMEOUT" "$BIN" inference $(common_args) \
      --prompt "$(cat "$prompt_file")" \
      --steps "$STEPS" \
      "${workers_arg[@]}" \
      > "$log" 2>&1 || status="fail_rc$?"
  stop_monitors "$tag" "$nodes"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$seqlen" "$nodes" "$bw" "$rep" "$log" "$status" >> "$MANIFEST"
  echo "  -> $status  ($log)"
  echo "  cooldown ${COOLDOWN_SEC}s"
  sleep "$COOLDOWN_SEC"
}

# ---------------- main ----------------
log_meta
echo
echo "출력: $OUT_DIR"
echo

for bw in "${SWEEP_BW_MBIT[@]}"; do
  if [ "$bw" != "0" ]; then
    echo "### 대역폭 제한 ${bw} Mbit ###"
    [ "$DRY_RUN" = "1" ] || bash "$HERE/net_shape.sh" set "$bw" || echo "  [warn] tc 적용 실패 — 무제한으로 진행"
  else
    echo "### 대역폭 무제한 ###"
    [ "$DRY_RUN" = "1" ] || bash "$HERE/net_shape.sh" clear >/dev/null 2>&1 || true
  fi

  for nodes in "${SWEEP_NODES[@]}"; do
    start_workers "$nodes"
    for seqlen in "${SWEEP_SEQLENS[@]}"; do
      for rep in $(seq 1 "$REPEATS"); do
        run_one "$seqlen" "$nodes" "$bw" "$rep"
      done
    done
  done
done

[ "$DRY_RUN" = "1" ] || bash "$HERE/net_shape.sh" clear >/dev/null 2>&1 || true

echo
echo "완료: $OUT_DIR"
echo "다음: python3 $HERE/parse_breakdown.py $OUT_DIR -o $OUT_DIR/breakdown.tsv"
