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
# 반복 측정 + 워밍업.
#
# 5회 측정 결과 첫 실행만 일관되게 이상치였다:
#   rep1 49,992 / rep2 34,077 / rep3 34,824 / rep4 34,418 / rep5 32,414 ms
# 시작 온도는 rep1 이 오히려 평범했으므로(60.6 vs 59.0) 열이 아니라 콜드 스타트다
# — 첫 실행이 6.32 GB 모델을 SD 에서 읽으며 페이지 캐시/메모리 상태를 바꾼다.
# rep1 을 버리면 편차가 51% -> 7% 로 떨어진다.
REPEATS="${REPEATS:-3}"      # 워밍업 제외한 측정 횟수
WARMUP="${WARMUP:-1}"
COOLDOWN="${COOLDOWN:-45}"

mkdir -p "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/$TAG.log"

# --- 가드 1: 다른 dllama 프로세스가 돌고 있는가 ---
# 커맨드라인(-f)으로 찾으면 이 스크립트를 감싼 셸까지 잡힌다(래퍼 인자에 "dllama worker"
# 같은 문자열이 들어가므로). 실행 파일 이름(comm)으로만 판정한다.
# comm 은 15자로 잘리므로 dllama / dllama_N4 / dllama_REPACK2 등 변형도 ^dllama 로 걸린다.
running_dllama() {
    ps -eo pid=,comm= | awk '$2 ~ /^dllama/ {print $1}'
}
if [ -n "$(running_dllama)" ]; then
    echo "[중단] 다른 dllama 프로세스가 실행 중입니다:"
    for p in $(running_dllama); do
        printf '    %s\n' "$(ps -o pid=,args= -p "$p" | cut -c1-140)"
    done
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

# --- 가드 4: 로그 크기 상한 ---
# 종료 경로의 트래픽 요약이 무한 반복되어 로그가 14 GB 까지 자란 사례가 있다.
# 코드에도 가드를 넣었지만, 다른 경로가 또 폭주해도 디스크가 죽지 않게 여기서도 막는다.
MAX_LOG_MB="${MAX_LOG_MB:-200}"
( while sleep 20; do
    [ -f "$LOG" ] || continue
    sz=$(( $(stat -c%s "$LOG" 2>/dev/null || echo 0) / 1048576 ))
    if [ "$sz" -gt "$MAX_LOG_MB" ]; then
        echo "[가드] 로그가 ${sz}MB 를 넘어 중단합니다 (상한 ${MAX_LOG_MB}MB)" >> "$LOG"
        ps -eo pid=,comm= | awk '$2 ~ /^dllama/ {print $1}' | xargs -r kill -9 2>/dev/null
        break
    fi
  done ) &
LOG_GUARD_PID=$!
trap 'kill $LOG_GUARD_PID 2>/dev/null' EXIT

# 시작 전 온도 기록 (열 상태가 편차의 주원인이다)
read_temp() {
    if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
        awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp
    else
        echo "NA"
    fi
}

vals=""
TOTAL=$(( WARMUP + REPEATS ))
for rep in $(seq 1 "$TOTAL"); do
    replog="$OUT_DIR/logs/${TAG}_r${rep}.log"
    t0=$(read_temp)
    set +e
    timeout "$RUN_TIMEOUT" "$BIN" inference \
        --model "$MODEL" --tokenizer "$TOKENIZER" \
        --nthreads "$NTHREADS" --buffer-float-type q80 \
        --max-seq-len "$MAX_SEQ" --steps "$STEPS" \
        --seed 42 --temperature 0 \
        --wall-metrics 1 --stage-timing 1 \
        --prompt "$(cat "$PROMPT_FILE")" \
        "$@" > "$replog" 2>&1
    RC=$?
    set -e
    t1=$(read_temp)

    if [ $RC -ne 0 ]; then
        echo "   [실패] rep=$rep rc=$RC  (137/143 이면 OOM/kill)"
        tail -3 "$replog" | sed 's/^/     /'
        exit $RC
    fi
    pf=$(grep -m1 'prefillMs' "$replog" | awk '{print $2}')
    if [ "$rep" -le "$WARMUP" ]; then
        printf '   warmup prefill=%-10s temp %s->%s  (측정에서 제외)\n' "$pf" "$t0" "$t1"
    else
        printf '   rep%-2s  prefill=%-10s temp %s->%s\n' "$(( rep - WARMUP ))" "$pf" "$t0" "$t1"
        vals="$vals $pf"
    fi
    [ "$rep" -lt "$TOTAL" ] && sleep "$COOLDOWN"
done

# 중앙값을 대표값으로, 마지막 실행 로그를 $LOG 로 남긴다
cp "$OUT_DIR/logs/${TAG}_r${TOTAL}.log" "$LOG"
echo "$vals" | tr ' ' '\n' | grep -v '^$' | sort -n > "$OUT_DIR/logs/${TAG}.vals"
med=$(awk '{a[NR]=$1} END{ if(NR%2) print a[(NR+1)/2]; else printf "%.2f",(a[NR/2]+a[NR/2+1])/2 }' \
      "$OUT_DIR/logs/${TAG}.vals")
mn=$(head -1 "$OUT_DIR/logs/${TAG}.vals"); mx=$(tail -1 "$OUT_DIR/logs/${TAG}.vals")
spread=$(awk -v a="$mn" -v b="$mx" -v m="$med" 'BEGIN{ if(m>0) printf "%.0f", (b-a)/m*100; else print 0 }')
printf '   => 중앙값 %s ms  (범위 %s~%s, 편차 %s%%)\n' "$med" "$mn" "$mx" "$spread"
[ "$spread" -gt 15 ] && echo "   ⚠️  편차 ${spread}% — 배수 비교에 쓰기 전 쿨다운을 늘리거나 반복을 키울 것"
echo "$med" > "$OUT_DIR/logs/${TAG}.median"
