#!/usr/bin/env bash
# SharedPack-SDOT 일반성 검증 1차 — research/19 §10
#
# 조합: prompt {447, 1789, 7212} x B {16, 32} x threads {1, 4},  N=8 wave on
#
# 설계 원칙
#   1) 조합마다 A,B,B,A 4회를 **연달아** 돌린다. 비율은 같은 세션 안에서만 만든다.
#      세션 간 절대값 비교는 하지 않는다 (그것이 과거 1.013x 인공물의 원인이었다).
#   2) 첫 조합 앞에 웜업 1회를 버린다.
#   3) 매 회 워커 재기동.
#   4) 성분을 분리해 기록한다: prefillMs / syncWaitMs / syncXferMs / non-wait residual.
#   5) logits md5 를 같은 조합의 BASE 와 대조한다. 다르면 그 조합은 실패로 표시.
#
# 주의: S=7212 는 --max-seq-len 을 키워야 하고 KV 가 커진다. OOM 을 피하려고
#       조합별로 max-seq-len 을 프롬프트 길이에서 산정한다.
set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/generality
mkdir -p $AD/raw
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
W7="165.194.19.113:9998 165.194.19.191:9998 165.194.19.103:9998 165.194.19.94:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.96:9998"
WH=(165.194.19.113 165.194.19.191 165.194.19.103 165.194.19.94 165.194.19.104 165.194.19.54 165.194.19.96)
BIN=$(md5sum dllama | cut -c1-8)

run () {  # tag prompt maxseq steps batch threads mode
  local T=$1 P=$2 MS=$3 ST=$4 B=$5 TH=$6 M=$7
  for h in "${WH[@]}"; do
    timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h "pkill -x dllama 2>/dev/null; sleep 1; setsid ./dllama worker --port 9998 --nthreads $TH > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  done; sleep 6
  DLLAMA_SHARED_PACK=$M DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 5400 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads $TH --buffer-float-type q80 \
    --max-seq-len $MS --steps $ST --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline 1 --n-batches $B --prefill-chunk-size $B \
    --pp-layers 8,8,8,8,8,8,8,8 --workers $W7 \
    --prompt "$(cat $P)" > $AD/raw/${T}.log 2>&1
  g() { grep -oE "$1: [0-9.]+" $AD/raw/${T}.log | grep -oE '[0-9.]+' | tail -1; }
  local pm=$(g prefillMs) sw=$(g syncWaitMs) sx=$(g syncXferMs)
  local tok=$(grep -oE '[0-9]+ tokens' $AD/raw/${T}.log | tail -1 | grep -oE '[0-9]+')
  echo -e "$T\t$M\t${tok:-NA}\t$B\t$TH\t${pm:-FAIL}\t${sw:-NA}\t${sx:-NA}\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)\t$BIN"
}

echo -e "tag\tmode\ttokens\tB\tthreads\tprefillMs\tsyncWaitMs\tsyncXferMs\tmd5\tbin"
# 웜업 1회 폐기
run warmup prompts_gen/prompt_512.txt 1024 453 32 4 0 > /dev/null 2>&1

for spec in "s447:prompts_gen/prompt_512.txt:1024:453" \
            "s1789:prompts_gen/prompt_2048.txt:2560:1800" \
            "s7212:prompts_gen/prompt_8192.txt:8192:7300"; do
  IFS=: read -r S P MS ST <<< "$spec"
  for B in 16 32; do
    for TH in 4 1; do
      k=0
      for M in 0 1 1 0; do
        k=$((k+1))
        run "${S}_B${B}_t${TH}_${k}m${M}" "$P" "$MS" "$ST" "$B" "$TH" "$M"
      done
    done
  done
done
