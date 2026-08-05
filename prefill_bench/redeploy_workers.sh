#!/usr/bin/env bash
# 워커 노드에 수정된 빌드를 재배포한다.
#
# 왜 필요한가: research/02-baseline-dotprod-fix.md 참조.
#   워커에 남아 있는 구버전 바이너리는 dotprod 경로가 죽어 있어 prefill 연산이 3~4배 느리다.
#   root만 고치고 스윕을 돌리면 워커가 전부 straggler가 되어 wait_frac이 오염된다.
#   → EXP-1 본 스윕 전에 반드시 실행할 것.
#
# 각 노드에서 소스를 빌드하는 대신 root에서 만든 바이너리를 복사한다.
# (모든 노드가 같은 Cortex-A76 계열일 때만 유효. 이종 클러스터라면 BUILD_ON_NODE=1 사용)
#
#   bash redeploy_workers.sh                 # 전체 노드
#   bash redeploy_workers.sh 165.194.19.103  # 특정 노드만
#   BUILD_ON_NODE=1 bash redeploy_workers.sh # 각 노드에서 직접 빌드(이종 클러스터)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"

BUILD_ON_NODE="${BUILD_ON_NODE:-0}"
REMOTE_BIN="${REMOTE_BIN:-/home/ubuntu/dllama}"
REMOTE_SRC="${REMOTE_SRC:-/home/ubuntu/prefill-opt}"

if [ "$#" -gt 0 ]; then HOSTS=("$@"); else HOSTS=("${ALL_WORKERS[@]}"); fi

echo "== 로컬 바이너리 확인 =="
[ -x "$BIN" ] || { echo "빌드가 없습니다: $BIN  (make dllama 먼저)"; exit 1; }
ls -la "$BIN"
echo

for h in "${HOSTS[@]}"; do
  echo "===== $h ====="
  # 워커 먼저 정지 (worker-first 규칙의 역순: 교체 시에는 반드시 먼저 죽인다)
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_USER@$h" "pkill -x dllama || true" || {
    echo "  [skip] 접속 실패"; continue; }

  if [ "$BUILD_ON_NODE" = "1" ]; then
    echo "  -- 노드에서 직접 빌드"
    rsync -a --delete \
      --exclude='*.m' --exclude='*.t' --exclude='.git/' --exclude='bench_prefill/' \
      --exclude='bench_logs/' --exclude='logs/' --exclude='*.o' --exclude='dllama' \
      "$REPO_DIR/" "$SSH_USER@$h:$REMOTE_SRC/"
    ssh "$SSH_USER@$h" "cd $REMOTE_SRC && make -j\$(nproc) dllama && cp dllama $REMOTE_BIN"
  else
    echo "  -- 바이너리 복사"
    scp -q "$BIN" "$SSH_USER@$h:$REMOTE_BIN.new"
    ssh "$SSH_USER@$h" "chmod +x $REMOTE_BIN.new && mv $REMOTE_BIN.new $REMOTE_BIN"
  fi

  # 검증: dotprod 경로가 실제로 들어간 바이너리인지 크기로 1차 확인
  ssh "$SSH_USER@$h" "ls -la $REMOTE_BIN; $REMOTE_BIN 2>&1 | head -1"
  echo
done

echo "완료. 다음: bash $REPO_DIR/scripts/rpi_worker_run.sh 로 워커 기동"
