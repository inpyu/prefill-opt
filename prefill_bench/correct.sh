#!/usr/bin/env bash
# N=8 정확성 매트릭스 — packed 표현까지 직접 대조 (research/19 §8.7)
#
# E2E logits hash 만으로는 커널 정확성을 보장하지 못한다. 실제로 축 규약 오독
# 빌드가 hash 일치로 통과한 적이 있다. DLLAMA_VERIFY_PACK=1 은 matmul 이
# 소비하는 packed 표현을 재계산해 대조하고, 종료 시 요약을 항상 출력한다.
# 워커는 SIGTERM 으로 죽으므로 시그널 핸들러에서도 요약을 찍는다.
#
# comparisons=0 인데 sp=1 이면 "검증이 아예 안 돌았다" 는 뜻이다 — 통과가 아니다.
set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/n8_ab
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
W7="165.194.19.113:9998 165.194.19.191:9998 165.194.19.103:9998 165.194.19.94:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.96:9998"
WH=(165.194.19.113 165.194.19.191 165.194.19.103 165.194.19.94 165.194.19.104 165.194.19.54 165.194.19.96)
BIN=$(md5sum dllama | cut -c1-8)
echo -e "wave\tsp\tprefillMs\tmd5\trootCmp\trootMis\twkCmp\twkMis\tbin"
for WV in 1 0; do
 for M in 0 1; do
  for h in "${WH[@]}"; do
    timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h "pkill -x dllama 2>/dev/null; sleep 1; setsid env DLLAMA_VERIFY_PACK=1 ./dllama worker --port 9998 --nthreads 4 > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  done; sleep 6
  T=correct_w${WV}_sp${M}
  DLLAMA_VERIFY_PACK=1 DLLAMA_SHARED_PACK=$M DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 3600 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len 1024 --steps 453 --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline $WV --n-batches 32 --prefill-chunk-size 32 \
    --pp-layers 8,8,8,8,8,8,8,8 --workers $W7 \
    --prompt "$(cat prompts_gen/prompt_512.txt)" > $AD/raw/${T}.log 2>&1
  WC=0; WM=0
  for h in "${WH[@]}"; do
    timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h 'pkill -x -TERM dllama 2>/dev/null; sleep 2; exit 0' 2>/dev/null
    L=$(timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h 'cat /home/ubuntu/w.log' 2>/dev/null)
    c=$(echo "$L" | grep -oE '^comparisons=[0-9]+' | cut -d= -f2 | tail -1)
    m=$(echo "$L" | grep -oE '^mismatches=[0-9]+' | cut -d= -f2 | tail -1)
    WC=$((WC+${c:-0})); WM=$((WM+${m:-0}))
  done
  RC=$(grep -oE '^comparisons=[0-9]+' $AD/raw/${T}.log | cut -d= -f2 | tail -1)
  RM=$(grep -oE '^mismatches=[0-9]+' $AD/raw/${T}.log | cut -d= -f2 | tail -1)
  echo -e "$WV\t$M\t$(grep -oE 'prefillMs: [0-9.]+' $AD/raw/${T}.log|grep -oE '[0-9.]+')\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)\t${RC:-NA}\t${RM:-NA}\t$WC\t$WM\t$BIN"
 done
done
