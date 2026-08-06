#!/usr/bin/env bash
# EXP-1 공통 설정. run_breakdown.sh 가 source 한다.
# 이 파일만 고치면 스윕 전체가 바뀐다.

set -euo pipefail

# ---------- 경로 ----------
REPO_DIR="${REPO_DIR:-/home/ubuntu/prefill-opt}"
BIN="${BIN:-$REPO_DIR/dllama}"
MODEL="${MODEL:-$REPO_DIR/dllama_model_llama3-8b_q40.m}"          # symlink → layer-skip-bypass
TOKENIZER="${TOKENIZER:-$REPO_DIR/dllama_tokenizer_llama3.t}"     # symlink
PROMPT_DIR="${PROMPT_DIR:-$REPO_DIR/prompts_gen}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/bench_prefill}"

# ---------- 클러스터 ----------
# 전 노드 동일: Raspberry Pi 5 / Cortex-A76 4코어 / aarch64.
# 메모리만 이종이다.
#
#   16 GB: root(165.194.19.105) + .113 + .166 + .191
#    8 GB: .103 / .94 / .104 / .54 / .96
#
# ⚠️ 반드시 공인/LAN IP(165.x)를 쓸 것.
#    Tailscale 주소(100.x)도 살아 있지만 WireGuard 오버레이라 암호화 오버헤드와
#    DERP 릴레이 경유 가능성이 있어 통신 측정이 오염된다. 접속 불가 시 fallback 전용.
#    매핑: .105<->100.72.91.19  .113<->100.67.190.3  .166<->100.84.48.55  .191<->100.76.95.128
#
# 목록 순서가 곧 노드 배정 순서다(앞에서부터 N-1개 사용).
# 16 GB 를 앞에 두어 4노드 구성이 전부 16 GB 가 되게 한다.
#   - TP 는 노드당 가중치가 1/N 이라 8 GB 로도 충분하지만,
#   - CP(각 노드가 전체 가중치 보유)는 16 GB 가 필요하다. research/05 Part D 참조.
NODES_16GB=(
  165.194.19.113
  165.194.19.166
  165.194.19.191
)
NODES_8GB=(
  165.194.19.103
  165.194.19.94
  165.194.19.104
  165.194.19.54
  165.194.19.96
)
ALL_WORKERS=("${NODES_16GB[@]}" "${NODES_8GB[@]}")
WORKER_PORT="${WORKER_PORT:-9998}"
SSH_USER="${SSH_USER:-ubuntu}"
NTHREADS="${NTHREADS:-4}"

# ---------- 스윕 축 ----------
# 프롬프트 길이(목표 토큰 수). make_prompts.py 가 생성한 것과 일치해야 한다.
SWEEP_SEQLENS=(${SWEEP_SEQLENS:-128 512 2048 8192})
# 노드 수(root 포함). 1은 단일 노드 기준선.
SWEEP_NODES=(${SWEEP_NODES:-1 2 4 8})
# 대역폭 제한(Mbps). 0 = 무제한(1GbE 그대로). net_shape.sh 사용.
SWEEP_BW_MBIT=(${SWEEP_BW_MBIT:-0 100})
REPEATS="${REPEATS:-3}"

# ---------- 실행 파라미터 ----------
# TTFT만 보면 되므로 decode는 최소로.
STEPS="${STEPS:-4}"
COOLDOWN_SEC="${COOLDOWN_SEC:-60}"      # 열 조건 정렬. 줄이지 말 것.
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"

# TP 기준선 인자. PP 비교가 필요하면 PP_SIZE를 바꿔가며 별도 스윕한다.
PP_SIZE="${PP_SIZE:-1}"
SP_SIZE="${SP_SIZE:-1}"

common_args() {
  echo --model "$MODEL" \
       --tokenizer "$TOKENIZER" \
       --nthreads "$NTHREADS" \
       --collective auto \
       --net-turbo 0 \
       --pp-size "$PP_SIZE" --sp-size "$SP_SIZE" \
       --pipeline-float-type q40 --buffer-float-type q80 \
       --pipeline-chunk-bytes 16384 \
       --pipeline-delta 1 --pipeline-delta-min-bytes 4096 \
       --wall-metrics 1 \
       --stage-timing 1 \
       --decode-log-interval 0
}

# N개 노드 구성에서 root에 넘길 --workers 문자열 (N-1개)
workers_for() {
  local n="$1"
  local out=()
  local i=0
  while [ "$i" -lt $((n - 1)) ]; do
    out+=("${ALL_WORKERS[$i]}:$WORKER_PORT")
    i=$((i + 1))
  done
  echo "${out[@]:-}"
}
