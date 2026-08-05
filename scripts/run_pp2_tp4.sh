#!/usr/bin/env bash
set -euo pipefail

# 8-node mode (root+7 workers): pp=2, sp=1 => tp=4 (auto from topology)
# KV immobility is preserved with strict-kv-affinity=1 and allow-kv-migration=0.

BIN=${BIN:-./dllama}
MODEL=${MODEL:-dllama_model_llama3-8b_q40.m}
TOKENIZER=${TOKENIZER:-dllama_tokenizer_llama3_8B.t}
PROMPT=${PROMPT:-"Once upon a time in a land far away"}
STEPS=${STEPS:-64}
NTHREADS=${NTHREADS:-4}
PIPELINE_FLOAT_TYPE=${PIPELINE_FLOAT_TYPE:-q40}
BUFFER_FLOAT_TYPE=${BUFFER_FLOAT_TYPE:-q80}
WORKERS=${WORKERS:-"165.194.19.130:9998 165.194.19.103:9998 165.194.19.96:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.113:9998 165.194.19.166:9998"}

# TP-enabled path: pp-token-only/topk fast path is TP=1 only, so disable it here.
exec "$BIN" inference \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --buffer-float-type "$BUFFER_FLOAT_TYPE" \
  --pipeline-float-type "$PIPELINE_FLOAT_TYPE" \
  --prompt "$PROMPT" \
  --steps "$STEPS" \
  --nthreads "$NTHREADS" \
  --collective auto \
  --net-turbo 0 \
  --pp-size 2 \
  --sp-size 1 \
  --pp-token-only 0 \
  --pp-topk 0 \
  --strict-kv-affinity 1 \
  --allow-kv-migration 0 \
  --stage-timing 1 \
  --wall-metrics 1 \
  --pipeline-chunk-bytes 16384 \
  --pipeline-delta 1 \
  --pipeline-delta-min-bytes 4096 \
  --workers $WORKERS
