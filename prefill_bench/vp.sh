set -u
cd /home/ubuntu/prefill-opt
AD=/home/ubuntu/prefill-opt/artifacts/n8_ab
LOCK=/tmp/dllama_bench.lock; exec 9>$LOCK
flock -n 9 || { echo "다른 벤치 실행 중"; exit 1; }
H=165.194.19.113
timeout 20 ssh -o ConnectTimeout=10 ubuntu@$H 'pkill -x dllama 2>/dev/null; sleep 1; exit 0' 2>/dev/null
timeout 120 scp -o ConnectTimeout=15 dllama ubuntu@$H:/home/ubuntu/dllama >/dev/null 2>&1 || { echo scp실패; exit 1; }
for WV in 1 0; do
  timeout 20 ssh -o ConnectTimeout=10 ubuntu@$H "pkill -x dllama 2>/dev/null; sleep 1; setsid env DLLAMA_VERIFY_PACK=1 ./dllama worker --port 9998 --nthreads 4 > /home/ubuntu/w.log 2>&1 < /dev/null & disown" 2>/dev/null
  sleep 6
  T=vp_wave$WV
  DLLAMA_VERIFY_PACK=1 DLLAMA_SHARED_PACK=1 DLLAMA_DUMP_LOGITS=$AD/raw/${T}.bin \
  timeout 1200 ./dllama inference --model dllama_model_llama3-8b_q40.m \
    --tokenizer dllama_tokenizer_llama3.t --nthreads 4 --buffer-float-type q80 \
    --max-seq-len 1024 --steps 453 --seed 42 --temperature 0 \
    --tile-aligned 1 --wave-pipeline $WV --n-batches 32 --prefill-chunk-size 32 \
    --pp-layers 16,16 --workers $H:9998 \
    --prompt "$(cat prompts_gen/prompt_512.txt)" > $AD/raw/${T}.log 2>&1
  timeout 30 scp -o ConnectTimeout=10 ubuntu@$H:/home/ubuntu/w.log $AD/raw/${T}_worker.log >/dev/null 2>&1
  echo "== wave=$WV md5=$(md5sum $AD/raw/${T}.bin 2>/dev/null|cut -c1-12)"
  echo "  root   VERIFY 히트: $(grep -c VERIFY_PACK $AD/raw/${T}.log)"
  echo "  worker VERIFY 히트: $(grep -c VERIFY_PACK $AD/raw/${T}_worker.log 2>/dev/null)"
  grep -h VERIFY_PACK $AD/raw/${T}.log $AD/raw/${T}_worker.log 2>/dev/null | head -4
done
