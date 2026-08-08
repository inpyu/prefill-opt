#!/usr/bin/env bash
# 설정 여러 개를 "번갈아" 실행해서 비교한다.
#
#   bash run_ab.sh <프롬프트> <출력디렉토리> <라운드수> -- <이름1>:<인자...> <이름2>:<인자...>
#
#   bash run_ab.sh prompts_gen/prompt_512.txt out 4 -- \
#        "off:--attn-fused 0" "on:--attn-fused 1"
#
# 왜 필요한가:
#   run_one.sh 는 한 설정을 반복해 중앙값을 내지만, **설정 간 순서 효과**는 못 막는다.
#   설정을 순차로 돌리면 뒤에 오는 쪽이 일관되게 불리해진다. 실제로 융합 attention 을
#   이렇게 비교했다가 "1.3% 손해"라는 반대 결론을 냈었다 — 커널과 무관한 gemm(+558)과
#   ffn(+517)까지 같이 늘어난 것이 드리프트의 증거였다.
#   번갈아 돌리면 드리프트가 모든 설정에 똑같이 걸린다.
#
#   첫 라운드는 콜드 스타트라 버린다(research/06 §7).
set -euo pipefail

PROMPT="${1:?usage: run_ab.sh <prompt> <outdir> <rounds> -- <name>:<args> ...}"
OUT="${2:?}"; ROUNDS="${3:?}"; shift 3
[ "${1:-}" = "--" ] && shift

MODEL="${MODEL:-/home/ubuntu/prefill-opt/dllama_model_llama3-8b_q40.m}"
TOKENIZER="${TOKENIZER:-/home/ubuntu/prefill-opt/dllama_tokenizer_llama3.t}"
BIN="${BIN:-./dllama}"
NTHREADS="${NTHREADS:-4}"
COOLDOWN="${COOLDOWN:-30}"

if [ -n "$(ps -eo pid=,comm= | awk '$2 ~ /^dllama/ {print $1}')" ]; then
    echo "[중단] 다른 dllama 프로세스가 실행 중입니다."; exit 1
fi
AVAIL=$(free -m | awk 'NR==2{print $7}')
[ "$AVAIL" -lt "${MIN_AVAIL_MB:-8000}" ] && { echo "[중단] 가용 메모리 ${AVAIL}MB"; exit 1; }

WORDS=$(wc -w < "$PROMPT"); EST=$(( WORDS * 13 / 10 ))
STEPS=$(( EST + 3 )); MAX_SEQ=$(( (EST + 256 + 31) / 32 * 32 ))
mkdir -p "$OUT/logs"
echo "프롬프트 ~${EST}토큰, max_seq=$MAX_SEQ, ${ROUNDS}라운드 x $# 설정 (첫 라운드 제외)"

for r in $(seq 1 "$ROUNDS"); do
  for cfg in "$@"; do
    name="${cfg%%:*}"; args="${cfg#*:}"
    # shellcheck disable=SC2086
    timeout "${RUN_TIMEOUT:-3600}" "$BIN" inference --model "$MODEL" --tokenizer "$TOKENIZER" \
        --nthreads "$NTHREADS" --buffer-float-type q80 --max-seq-len "$MAX_SEQ" --steps "$STEPS" \
        --seed 42 --temperature 0 --wall-metrics 1 --stage-timing 1 \
        $args --prompt "$(cat "$PROMPT")" > "$OUT/logs/${name}_r${r}.log" 2>&1 || {
            echo "  [실패] $name r$r"; continue; }
    pf=$(grep -m1 prefillMs "$OUT/logs/${name}_r${r}.log" | awk '{printf "%.0f",$2}')
    at=$(grep -m1 attnMs "$OUT/logs/${name}_r${r}.log" | awk '{printf "%.0f",$2}')
    printf '  r%-2s %-12s prefill=%-8s attn=%s\n' "$r" "$name" "$pf" "$at"
    [ "$r" -gt 1 ] && echo "$pf $at" >> "$OUT/logs/${name}.vals"
    sleep "$COOLDOWN"
  done
done

echo
printf '%-14s %10s %10s\n' 설정 prefill attn
for cfg in "$@"; do
  name="${cfg%%:*}"
  [ -f "$OUT/logs/${name}.vals" ] || continue
  printf '%-14s %10s %10s\n' "$name" \
    "$(awk '{print $1}' "$OUT/logs/${name}.vals" | sort -n | awk '{a[NR]=$1}END{print a[int((NR+1)/2)]}')" \
    "$(awk '{print $2}' "$OUT/logs/${name}.vals" | sort -n | awk '{a[NR]=$1}END{print a[int((NR+1)/2)]}')"
done
