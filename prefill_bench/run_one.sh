#!/usr/bin/env bash
# 단일 노드 prefill 측정 1회. OOM 방지 가드를 강제한다.
#
#   bash run_one.sh <bin> <tag> <prompt_file> <out_dir> [extra dllama args...]
#
# 왜 이 래퍼가 필요한가 (실제로 세 번 당했다):
#   1) 바이너리 이름이 dllama_XXX 처럼 변형되면 `pgrep -x dllama` 가 못 잡는다.
#      -> 커맨드라인 전체로 검사한다.
#   2) KV 캐시는 실제 프롬프트 길이가 아니라 --max-seq-len 전체로 할당된다.
#        size2D(F_32, seqLen, kvDim0) x 2(K,V) x nLayers
#      8B/seqLen 8192 이면 KV 만 2.1 GB. 모델 6.3 GB 와 합쳐 8.4 GB 가 되어
#      16 GB 머신에서도 다른 작업과 겹치면 OOM 이다.
#      -> 프롬프트 길이에 맞춰 자동으로 좁힌다.
#   3) 빌드/다른 추론과 동시에 돌리면 죽는다.
#      -> 시작 전 가용 메모리를 확인한다.
set -euo pipefail

BIN="${1:?usage: run_one.sh <bin> <tag> <prompt_file> <out_dir> [args...]}"
TAG="${2:?}"
PROMPT_FILE="${3:?}"
OUT_DIR="${4:?}"
shift 4

MODEL="${MODEL:-/home/ubuntu/prefill-opt/dllama_model_llama3-8b_q40.m}"
TOKENIZER="${TOKENIZER:-/home/ubuntu/prefill-opt/dllama_tokenizer_llama3.t}"
NTHREADS="${NTHREADS:-4}"
MIN_AVAIL_MB="${MIN_AVAIL_MB:-8000}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

mkdir -p "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/$TAG.log"

# --- 가드 1: 다른 dllama 추론이 돌고 있는가 (이름 변형 포함) ---
if pgrep -f "dllama[^ ]* +(inference|worker|chat)" >/dev/null 2>&1; then
    echo "[중단] 다른 dllama 프로세스가 실행 중입니다:"
    pgrep -af "dllama[^ ]* +(inference|worker|chat)" | sed 's/^/    /'
    exit 1
fi

# --- 가드 2: 가용 메모리 ---
AVAIL=$(free -m | awk 'NR==2{print $7}')
if [ "$AVAIL" -lt "$MIN_AVAIL_MB" ]; then
    echo "[중단] 가용 메모리 ${AVAIL}MB < ${MIN_AVAIL_MB}MB. 다른 작업을 먼저 정리하세요."
    free -h | head -2
    exit 1
fi

# --- 가드 3: KV 캐시 상한을 프롬프트에 맞춘다 ---
# 토큰 수는 단어 수로 근사(영문 산문 ~1.3 tok/word). 여유 256 을 더한다.
WORDS=$(wc -w < "$PROMPT_FILE")
EST_TOKENS=$(( WORDS * 13 / 10 ))
STEPS="${STEPS:-$(( EST_TOKENS + 120 ))}"
MAX_SEQ="${MAX_SEQ:-$(( EST_TOKENS + 256 ))}"
# dllama 는 seqLen % spSize == 0 을 요구하므로 32 배수로 올린다.
MAX_SEQ=$(( (MAX_SEQ + 31) / 32 * 32 ))

echo "== $TAG =="
echo "   bin=$BIN"
echo "   words=$WORDS  est_tokens=$EST_TOKENS  steps=$STEPS  max_seq_len=$MAX_SEQ"
echo "   avail=${AVAIL}MB  (KV 예상 $(( MAX_SEQ * 1024 * 4 * 2 * 32 / 1048576 )) MB)"

set +e
timeout "$RUN_TIMEOUT" "$BIN" inference \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --nthreads "$NTHREADS" --buffer-float-type q80 \
    --max-seq-len "$MAX_SEQ" --steps "$STEPS" \
    --seed 42 --temperature 0 \
    --wall-metrics 1 --stage-timing 1 \
    --prompt "$(cat "$PROMPT_FILE")" \
    "$@" > "$LOG" 2>&1
RC=$?
set -e

if [ $RC -ne 0 ]; then
    echo "   [실패] rc=$RC  (137/143 이면 OOM/kill)"
    tail -3 "$LOG" | sed 's/^/     /'
    exit $RC
fi
echo "   $(grep -m1 'prefillMs' "$LOG")  $(grep -m1 'ffnMs' "$LOG")"
