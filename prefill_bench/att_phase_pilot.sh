#!/usr/bin/env bash
# Phase 1 — attention 내부 분해 pilot (research/DerivePP/05-attention-layer §5.4)
#
# 두 단계다.
#   A) 계측기 오버헤드 검증 — DLLAMA_ATT_PHASE 0/1/0/1 교차. 차이 >2% 면 중단
#   B) pilot 분해       — S_real 447/1789/7212 x B=32 x threads=4, 웜업1 + 본3
#
# 처음부터 3길이 x 2B x 4스레드 x 3회를 돌지 않는다. pilot 으로
# phase 회계가 맞는지, softmax 가 지배적인지, AV share 가 충분한지부터 본다.
#
# 단일 노드에서 잰다. attention 은 노드 로컬 연산이고, N=8 로 재면
# 스테이지당 레이어가 1/8 이라 표본이 줄고 통신이 섞인다.
set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/att_phase
mkdir -p $AD/raw
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
BIN=$(md5sum dllama | cut -c1-8)

run () { # tag prompt maxseq steps phase
  local T=$1 P=$2 MS=$3 ST=$4 PH=$5
  DLLAMA_ATT_PHASE=$PH DLLAMA_OP_PROFILE=1 DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 7200 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len $MS --steps $ST --seed 42 --temperature 0 \
    --tile-aligned 1 --n-batches 32 --prefill-chunk-size 32 \
    --stage-timing 1 \
    --prompt "$(cat $P)" > $AD/raw/${T}.log 2>&1
  local att=$(grep -E '^block_multihead_att' $AD/raw/${T}.log | awk '{print $2}' | tail -1)
  local pf=$(grep -oE 'prefillMs: [0-9.]+' $AD/raw/${T}.log | grep -oE '[0-9.]+' | tail -1)
  echo -e "${T}\t${PH}\t${pf:-FAIL}\t${att:-NA}\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)\t$BIN"
}

echo "=== A) 계측기 오버헤드 검증 (S_real=1789, B=32, 4스레드)"
echo -e "tag\tphase\tprefillMs\tattMs\tmd5\tbin"
P=prompts_gen/prompt_2048.txt
run ovh_1_off $P 2560 1800 0
run ovh_2_on  $P 2560 1800 1
run ovh_3_off $P 2560 1800 0
run ovh_4_on  $P 2560 1800 1

echo
echo "=== B) pilot 분해 (계측 on, 웜업1 + 본3)"
for spec in "s447:prompts_gen/prompt_512.txt:1024:453" \
            "s1789:prompts_gen/prompt_2048.txt:2560:1800" \
            "s7212:prompts_gen/prompt_8192.txt:8192:7300"; do
  IFS=: read -r S P MS ST <<< "$spec"
  run ${S}_warm $P $MS $ST 1 > /dev/null 2>&1
  for k in 1 2 3; do run ${S}_r$k $P $MS $ST 1; done
done
echo
echo "phase 표는 raw/*.log 의 [ATT_PHASE] 블록에 있다."
echo "분석: python3 prefill_bench/att_phase_analyze.py"
