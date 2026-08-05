#!/usr/bin/env bash
set -euo pipefail

# Sweep PP sizes for single-request decode and summarize wall metrics.
# Designed to keep tp=1, sp=1 by selecting (pp-1) workers for each run.

ROOT_BIN=${ROOT_BIN:-./dllama}
MODEL=${MODEL:-dllama_model_llama3-8b_q40.m}
TOKENIZER=${TOKENIZER:-dllama_tokenizer_llama3_8B.t}
PROMPT=${PROMPT:-"Once upon a time in a land far away"}
STEPS=${STEPS:-64}
NTHREADS=${NTHREADS:-4}
SP_SIZE=${SP_SIZE:-1}
COLLECTIVE=${COLLECTIVE:-auto}
NET_TURBO=${NET_TURBO:-0}
BUFFER_FLOAT_TYPE=${BUFFER_FLOAT_TYPE:-q80}
PIPELINE_FLOAT_TYPE=${PIPELINE_FLOAT_TYPE:-q40}
PP_TOKEN_ONLY=${PP_TOKEN_ONLY:-1}
STAGE_TIMING=${STAGE_TIMING:-1}
WALL_METRICS=${WALL_METRICS:-1}
DECODE_LOG_INTERVAL=${DECODE_LOG_INTERVAL:-0}

# Transport options (set these to baseline or experiment knobs)
PIPELINE_CHUNK_BYTES=${PIPELINE_CHUNK_BYTES:-16384}
PIPELINE_DELTA=${PIPELINE_DELTA:-0}
PIPELINE_DELTA_MIN_BYTES=${PIPELINE_DELTA_MIN_BYTES:-4096}

# Repeat count per PP for averaging
REPEATS=${REPEATS:-3}
PP_LIST=${PP_LIST:-"1 2 4 8"}
CASE_RETRY=${CASE_RETRY:-1}
CASE_COOLDOWN_SEC=${CASE_COOLDOWN_SEC:-2}
WATCHDOG_POLL_SEC=${WATCHDOG_POLL_SEC:-2}
WATCHDOG_STALL_SEC=${WATCHDOG_STALL_SEC:-180}
MAX_CASE_SECONDS=${MAX_CASE_SECONDS:-0}
PRESTART_MAX_SEC=${PRESTART_MAX_SEC:-0}

# Full worker pool; for PP=n this script uses first (n-1) workers.
WORKERS=${WORKERS:-"165.194.19.103:9998 165.194.19.94:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.113:9998 165.194.19.166:9998 165.194.19.96:9998"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-bench_logs}
mkdir -p "$OUT_DIR"
TSV_PATH="$OUT_DIR/pp_sweep_${TIMESTAMP}.tsv"
SUMMARY_PATH="$OUT_DIR/pp_sweep_${TIMESTAMP}_summary.txt"

echo -e "pp\trun\tstatus\tpred_tps\twall_tpot_ms\twall_decode_tps\troot_wait_p50\troot_wait_p95\tdecode_ms\tworkers\tlog" > "$TSV_PATH"

# split workers into bash array
read -r -a ALL_WORKERS <<< "$WORKERS"

subset_workers() {
  local need="$1"
  if [[ "$need" -le 0 ]]; then
    echo ""
    return 0
  fi
  if [[ "${#ALL_WORKERS[@]}" -lt "$need" ]]; then
    echo ""
    return 1
  fi
  local out=()
  local i
  for ((i=0; i<need; i++)); do
    out+=("${ALL_WORKERS[$i]}")
  done
  echo "${out[*]}"
}

run_case() {
  local pp="$1"
  local run_id="$2"

  local need_workers=$((pp - 1))
  local selected_workers
  if ! selected_workers=$(subset_workers "$need_workers"); then
    echo -e "${pp}\t${run_id}\tskipped_insufficient_workers\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA" >> "$TSV_PATH"
    echo "[pp=$pp run=$run_id] skipped: need ${need_workers} workers, have ${#ALL_WORKERS[@]}"
    return
  fi

  local log_path="$OUT_DIR/pp${pp}_run${run_id}_${TIMESTAMP}.log"

  local cmd=(
    "$ROOT_BIN" inference
    --model "$MODEL"
    --tokenizer "$TOKENIZER"
    --buffer-float-type "$BUFFER_FLOAT_TYPE"
    --pipeline-float-type "$PIPELINE_FLOAT_TYPE"
    --prompt "$PROMPT"
    --steps "$STEPS"
    --nthreads "$NTHREADS"
    --collective "$COLLECTIVE"
    --net-turbo "$NET_TURBO"
    --pp-size "$pp"
    --sp-size "$SP_SIZE"
    --pp-token-only "$PP_TOKEN_ONLY"
    --stage-timing "$STAGE_TIMING"
    --wall-metrics "$WALL_METRICS"
    --decode-log-interval "$DECODE_LOG_INTERVAL"
    --pipeline-chunk-bytes "$PIPELINE_CHUNK_BYTES"
    --pipeline-delta "$PIPELINE_DELTA"
    --pipeline-delta-min-bytes "$PIPELINE_DELTA_MIN_BYTES"
  )

  if [[ "$need_workers" -gt 0 ]]; then
    cmd+=(--workers)
    # shellcheck disable=SC2206
    local selected_arr=( $selected_workers )
    cmd+=("${selected_arr[@]}")
  fi

  local status="ok"
  local attempt=0
  local cmd_rc=0
  while true; do
    attempt=$((attempt + 1))
    : > "$log_path"
    "${cmd[@]}" > "$log_path" 2>&1 &
    local cmd_pid=$!
    local started_ts
    started_ts=$(date +%s)
    local last_change_ts="$started_ts"
    local last_size=-1
    local saw_progress=0
    local inference_started=0
    local forced_kill=0
    local force_reason=""

    while kill -0 "$cmd_pid" 2>/dev/null; do
      sleep "$WATCHDOG_POLL_SEC"

      local now_ts
      now_ts=$(date +%s)
      local size
      size=$(stat -c%s "$log_path" 2>/dev/null || echo 0)
      if [[ "$size" != "$last_size" ]]; then
        last_size="$size"
        last_change_ts="$now_ts"
        saw_progress=1
      fi
      if [[ "$inference_started" -eq 0 ]] && grep -Eq "🔷️ Eval|🔶 Pred|🧭 \\[ROOT_STAGE\\]|Evaluation|WallMetrics" "$log_path"; then
        inference_started=1
      fi

      if grep -Eq "Critical error|NET_TIMEOUT|Broken pipe|Pipeline recv error|Pipeline send error|tryReadSocket timeout|writeMany timeout|readMany timeout" "$log_path"; then
        forced_kill=1
        force_reason="fatal_log"
      fi

      if [[ "$MAX_CASE_SECONDS" -gt 0 ]] && [[ $((now_ts - started_ts)) -ge "$MAX_CASE_SECONDS" ]]; then
        forced_kill=1
        force_reason="max_case_seconds"
      fi

      # Stall watchdog is meaningful only after inference actually started.
      if [[ "$WATCHDOG_STALL_SEC" -gt 0 ]] && [[ "$inference_started" -eq 1 ]] && [[ "$saw_progress" -eq 1 ]] && [[ $((now_ts - last_change_ts)) -ge "$WATCHDOG_STALL_SEC" ]]; then
        forced_kill=1
        force_reason="log_stall"
      fi

      # Optional guard for pre-start hangs (disabled by default).
      if [[ "$PRESTART_MAX_SEC" -gt 0 ]] && [[ "$inference_started" -eq 0 ]] && [[ $((now_ts - started_ts)) -ge "$PRESTART_MAX_SEC" ]]; then
        forced_kill=1
        force_reason="prestart_timeout"
      fi

      if [[ "$forced_kill" -eq 1 ]]; then
        kill -TERM "$cmd_pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$cmd_pid" 2>/dev/null || true
        break
      fi
    done

    set +e
    wait "$cmd_pid"
    cmd_rc=$?
    set -e

    status="ok"
    if [[ "$forced_kill" -eq 1 ]]; then
      status="$force_reason"
    fi
    if [[ $cmd_rc -ne 0 && "$status" == "ok" ]]; then
      status="cmd_fail($cmd_rc)"
    fi
    # NOTE: plain "Socket closed"/"Network is closed" can appear during normal teardown.
    # Treat only truly fatal patterns as error.
    if grep -Eq "Critical error|NET_TIMEOUT|Broken pipe|Pipeline recv error|Pipeline send error|tryReadSocket timeout|writeMany timeout|readMany timeout" "$log_path"; then
      status="error"
    fi

    if [[ "$status" == "ok" || "$attempt" -gt "$((CASE_RETRY + 1))" ]]; then
      break
    fi
    echo "[pp=$pp run=$run_id] retry $attempt/$((CASE_RETRY + 1)) after status=$status"
    sleep "$CASE_COOLDOWN_SEC"
  done

  local pred_tps wall_tpot wall_decode_tps root_p50 root_p95 decode_ms
  pred_tps=$(awk '/tokens\/s:/ {c++; if(c==2){print $2; exit}}' "$log_path")
  wall_tpot=$(awk -F': ' '/wall_tpot_ms:/ {print $2; exit}' "$log_path")
  wall_decode_tps=$(awk -F': ' '/wall_decode_tps:/ {print $2; exit}' "$log_path")
  root_p50=$(awk -F': ' '/root_recv_wait_p50_ms:/ {print $2; exit}' "$log_path")
  root_p95=$(awk -F': ' '/root_recv_wait_p95_ms:/ {print $2; exit}' "$log_path")
  decode_ms=$(awk -F': ' '/decodeMs:/ {gsub(/^ +/,"",$2); print $2; exit}' "$log_path")

  pred_tps=${pred_tps:-NA}
  wall_tpot=${wall_tpot:-NA}
  wall_decode_tps=${wall_decode_tps:-NA}
  root_p50=${root_p50:-NA}
  root_p95=${root_p95:-NA}
  decode_ms=${decode_ms:-NA}

  echo -e "${pp}\t${run_id}\t${status}\t${pred_tps}\t${wall_tpot}\t${wall_decode_tps}\t${root_p50}\t${root_p95}\t${decode_ms}\t${need_workers}\t${log_path}" >> "$TSV_PATH"
  echo "[pp=$pp run=$run_id] status=$status wall_tpot=$wall_tpot pred_tps=$pred_tps wait_p95=$root_p95"
  sleep "$CASE_COOLDOWN_SEC"
}

echo "== PP sweep start =="
for pp in $PP_LIST; do
  for run_id in $(seq 1 "$REPEATS"); do
    run_case "$pp" "$run_id"
  done
done

# aggregate means by pp for successful runs
{
  echo "pp_sweep summary"
  echo "timestamp: $TIMESTAMP"
  echo "tsv: $TSV_PATH"
  echo
  echo "per-PP averages (ok runs only):"
  awk -F'\t' '
    NR==1 { next }
    $3=="ok" && $5!="NA" {
      pp=$1
      n[pp]++
      tpot[pp]+=$5
      pred[pp]+=$4
      wdps[pp]+=$6
      p95[pp]+=$8
    }
    END {
      printf("pp\tok_runs\tavg_wall_tpot_ms\tavg_pred_tps\tavg_wall_decode_tps\tavg_root_wait_p95_ms\n")
      for (pp in n) {
        printf("%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n", pp, n[pp], tpot[pp]/n[pp], pred[pp]/n[pp], wdps[pp]/n[pp], p95[pp]/n[pp])
      }
    }
  ' "$TSV_PATH" | sort -t$'\t' -k3,3n
  echo
  echo "best pp by avg_wall_tpot_ms:"
  awk -F'\t' '
    NR==1 { next }
    $3=="ok" && $5!="NA" {
      pp=$1
      n[pp]++
      tpot[pp]+=$5
    }
    END {
      best_pp=""; best=1e18
      for (pp in n) {
        avg=tpot[pp]/n[pp]
        if (avg < best) { best=avg; best_pp=pp }
      }
      if (best_pp=="") print "N/A"
      else printf("pp=%s avg_wall_tpot_ms=%.3f\n", best_pp, best)
    }
  ' "$TSV_PATH"
} > "$SUMMARY_PATH"

echo
echo "Sweep complete"
echo "TSV: $TSV_PATH"
echo "Summary: $SUMMARY_PATH"
