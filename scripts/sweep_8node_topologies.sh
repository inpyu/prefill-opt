#!/usr/bin/env bash
set -euo pipefail

# Compare 8-node topologies under same settings:
# A) pp=8,sp=1,tp=1 (current baseline)
# B) pp=2,sp=1,tp=4 (PP+TP redesign)

BIN=${BIN:-./dllama}
MODEL=${MODEL:-dllama_model_llama3-8b_q40.m}
TOKENIZER=${TOKENIZER:-dllama_tokenizer_llama3_8B.t}
PROMPT=${PROMPT:-"Once upon a time in a land far away"}
STEPS=${STEPS:-64}
NTHREADS=${NTHREADS:-4}
REPEATS=${REPEATS:-1}
PIPELINE_FLOAT_TYPE=${PIPELINE_FLOAT_TYPE:-q40}
BUFFER_FLOAT_TYPE=${BUFFER_FLOAT_TYPE:-q80}
WORKERS=${WORKERS:-"165.194.19.130:9998 165.194.19.103:9998 165.194.19.96:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.113:9998 165.194.19.166:9998"}

OUT_DIR=${OUT_DIR:-bench_logs}
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
TSV="$OUT_DIR/topology_8node_${TS}.tsv"

echo -e "name\trun\tstatus\tpred_tps\twall_tpot_ms\twall_decode_tps\troot_wait_p95\tdecode_ms\tlog" > "$TSV"

run_one() {
  local name="$1"
  local pp="$2"
  local sp="$3"
  local token_only="$4"
  local log="$OUT_DIR/${name}_${TS}.log"

  set +e
  "$BIN" inference \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --buffer-float-type "$BUFFER_FLOAT_TYPE" \
    --pipeline-float-type "$PIPELINE_FLOAT_TYPE" \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --nthreads "$NTHREADS" \
    --collective auto \
    --net-turbo 0 \
    --pp-size "$pp" \
    --sp-size "$sp" \
    --pp-token-only "$token_only" \
    --pp-topk 0 \
    --strict-kv-affinity 1 \
    --allow-kv-migration 0 \
    --stage-timing 1 \
    --wall-metrics 1 \
    --pipeline-chunk-bytes 16384 \
    --pipeline-delta 1 \
    --pipeline-delta-min-bytes 4096 \
    --workers $WORKERS > "$log" 2>&1
  local rc=$?
  set -e

  local status=ok
  if [[ $rc -ne 0 ]]; then status="cmd_fail($rc)"; fi
  if grep -Eq "Critical error|NET_TIMEOUT|Broken pipe|Pipeline recv error|Pipeline send error" "$log"; then status=error; fi

  local pred wall_tpot wall_dps wait95 decode_ms
  pred=$(awk '/tokens\/s:/ {c++; if(c==2){print $2; exit}}' "$log")
  wall_tpot=$(awk -F': ' '/wall_tpot_ms:/ {print $2; exit}' "$log")
  wall_dps=$(awk -F': ' '/wall_decode_tps:/ {print $2; exit}' "$log")
  wait95=$(awk -F': ' '/root_recv_wait_p95_ms:/ {print $2; exit}' "$log")
  decode_ms=$(awk -F': ' '/decodeMs:/ {gsub(/^ +/,"",$2); print $2; exit}' "$log")

  echo -e "$name\t$run\t$status\t${pred:-NA}\t${wall_tpot:-NA}\t${wall_dps:-NA}\t${wait95:-NA}\t${decode_ms:-NA}\t$log" >> "$TSV"
}

for run in $(seq 1 "$REPEATS"); do
  run_one "pp8_tp1" 8 1 1
  run_one "pp2_tp4" 2 1 0
  echo "run $run done"
done

echo "saved: $TSV"
