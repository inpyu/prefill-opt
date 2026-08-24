set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/n8_ab
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
H=165.194.19.113
echo -e "chunk\tsp\tprefillMs\tmd5"
for C in 512 256 64; do
 for M in 0 1; do
  timeout 20 ssh -o ConnectTimeout=10 ubuntu@$H "pkill -x dllama 2>/dev/null; sleep 1; setsid ./dllama worker --port 9998 --nthreads 4 > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  sleep 6
  T=n2_c${C}_sp$M
  DLLAMA_SHARED_PACK=$M DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 1200 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len 1024 --steps 453 --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline 1 --n-batches $C --prefill-chunk-size $C \
    --pp-layers 16,16 --workers $H:9998 \
    --prompt "$(cat prompts_gen/prompt_512.txt)" > $AD/raw/${T}.log 2>&1
  echo -e "$C\t$M\t$(grep -oE 'prefillMs: [0-9.]+' $AD/raw/${T}.log|grep -oE '[0-9.]+')\t$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)"
 done
done
