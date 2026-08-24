set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/n8_ab
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
W7="165.194.19.113:9998 165.194.19.191:9998 165.194.19.103:9998 165.194.19.94:9998 165.194.19.104:9998 165.194.19.54:9998 165.194.19.96:9998"
WH=(165.194.19.113 165.194.19.191 165.194.19.103 165.194.19.94 165.194.19.104 165.194.19.54 165.194.19.96)
echo -e "idx\tmode\tprefillMs\tsyncWaitMs\tmd5"
i=0
for M in 0 0 1 1 0 0 1 1 0; do
  i=$((i+1))
  for h in "${WH[@]}"; do
    timeout 20 ssh -o ConnectTimeout=10 ubuntu@$h "pkill -x dllama 2>/dev/null; sleep 1; setsid ./dllama worker --port 9998 --nthreads 4 > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  done; sleep 6
  T=anchor_${i}_m${M}
  DLLAMA_SHARED_PACK=$M DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 1200 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len 1024 --steps 453 --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline 1 --n-batches 32 --prefill-chunk-size 32 \
    --pp-layers 8,8,8,8,8,8,8,8 --workers $W7 \
    --prompt "$(cat prompts_gen/prompt_512.txt)" > $AD/raw/${T}.log 2>&1
  echo -e "$i\t$M\t$(grep -oE 'prefillMs: [0-9.]+' $AD/raw/${T}.log|grep -oE '[0-9.]+')\t$(grep -oE 'syncWaitMs: [0-9.]+' $AD/raw/${T}.log|grep -oE '[0-9.]+')\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)"
done
echo "1번은 웜업으로 폐기. 순서 0,0,1,1,0,0,1,1 로 드리프트 상쇄"
