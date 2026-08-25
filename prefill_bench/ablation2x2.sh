#!/usr/bin/env bash
# 2x2 ablation — Wave Pipeline x SharedPack (research/DerivePP/04-codesign.md §4.1)
#
# 논문의 핵심 표. 두 기법을 나열하는 것과 두 계층이 서로를 강화한다고 보이는
# 것은 다르다. wave off 에서의 SP 이득과 wave on 에서의 SP 이득을 비교해야
# co-design 주장이 선다.
#
# 설계
#   조합 4개 x (A,B,B,A) — 조합마다 연달아 돌려 같은 세션 안에서만 비율을 만든다
#   첫 조합 앞에 웜업 1회 폐기
#   매 회 워커 재기동
#   성능 측정이므로 DLLAMA_VERIFY_PACK 은 끈다 (정확성은 correct.sh)
#
# 주의: wave on 과 wave off 는 로짓 해시가 다르다(wave 경로가 활성값을 q80 로
#       전송한다). 정확성 비교는 같은 wave 조건 안에서만 한다.
set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/ablation2x2
mkdir -p $AD/raw
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
W7="165.194.19.113:9998 165.194.19.191:9998 165.194.19.103:9998 165.194.19.94:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.96:9998"
WH=(165.194.19.113 165.194.19.191 165.194.19.103 165.194.19.94 165.194.19.104 165.194.19.54 165.194.19.96)
BIN=$(md5sum dllama | cut -c1-8)

run () { # tag wave mode
  local T=$1 WV=$2 M=$3
  for h in "${WH[@]}"; do
    timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h "pkill -x dllama 2>/dev/null; sleep 1; setsid ./dllama worker --port 9998 --nthreads 4 > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  done; sleep 6
  DLLAMA_SHARED_PACK=$M DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 3600 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len 1024 --steps 453 --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline $WV --n-batches 32 --prefill-chunk-size 32 \
    --pp-layers 8,8,8,8,8,8,8,8 --workers $W7 \
    --prompt "$(cat prompts_gen/prompt_512.txt)" > $AD/raw/${T}.log 2>&1
  g() { grep -oE "$1: [0-9.]+" $AD/raw/${T}.log | grep -oE '[0-9.]+' | tail -1; }
  echo -e "$T\t$WV\t$M\t$(g prefillMs)\t$(g syncWaitMs)\t$(g syncXferMs)\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)\t$BIN"
}

echo -e "tag\twave\tsp\tprefillMs\tsyncWaitMs\tsyncXferMs\tmd5\tbin"
run warmup 1 0 > /dev/null 2>&1
# wave on 쌍
k=0; for M in 0 1 1 0; do k=$((k+1)); run "w1_${k}m${M}" 1 $M; done
# wave off 쌍 (느리다 — 30초대 prefill)
k=0; for M in 0 1 1 0; do k=$((k+1)); run "w0_${k}m${M}" 0 $M; done
