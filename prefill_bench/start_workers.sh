#!/usr/bin/env bash
# 워커를 "확실히" 새로 기동한다.
#
#   bash start_workers.sh <nodes>          # 사용할 노드 수(root 제외한 워커 수)
#   bash start_workers.sh 3 165.194.19.113 165.194.19.166 165.194.19.191
#
# 왜 필요한가:
#   실패하거나 강제 종료된 런 뒤에 워커가 이전 세션 상태로 남아 다음 연결과 어긋난다.
#   증상은 root 쪽 "Connection reset by peer" / "Socket closed" 로 나타나는데,
#   코드 회귀로 오인하기 쉽다(실제로 한 번 오인해서 한참 헤맸다).
#
#   그래서 매 실행 전에:
#     1) 기존 프로세스를 확실히 죽이고
#     2) 포트가 실제로 해제될 때까지 기다리고
#     3) 새로 띄운 뒤 LISTEN 상태를 확인하고
#     4) 워커 로그가 새로 시작됐는지 확인한다
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"

N="${1:?usage: start_workers.sh <nWorkers> [host...]}"
shift || true
if [ "$#" -gt 0 ]; then HOSTS=("$@"); else HOSTS=("${ALL_WORKERS[@]:0:$N}"); fi

REMOTE_BIN="${REMOTE_BIN:-/home/ubuntu/dllama}"
PORT="${WORKER_PORT:-9998}"
# 워커에만 필요한 추가 인자. 워커는 root 의 CLI 를 받지 않으므로 여기서 넘긴다.
#   WORKER_EXTRA="--cp-split 1" bash start_workers.sh 3
WORKER_EXTRA="${WORKER_EXTRA:-}"
LOG="/home/ubuntu/dllama_worker_${PORT}.log"

LOCAL_MD5="$(md5sum "$BIN" | awk '{print $1}')"
echo "root 바이너리 md5=$LOCAL_MD5"

fail=0
for h in "${HOSTS[@]}"; do
    printf '%-16s ' "$h"

    remote_md5=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$SSH_USER@$h" "md5sum $REMOTE_BIN 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "")
    if [ "$remote_md5" != "$LOCAL_MD5" ]; then
        echo "[중단] 바이너리 불일치 (worker=$remote_md5). redeploy_workers.sh 를 먼저 실행하세요."
        fail=1
        continue
    fi

    ok=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_USER@$h" "
        # 1) 확실히 종료 (이름 변형 포함)
        ps -eo pid=,comm= | awk '\$2 ~ /^dllama/ {print \$1}' | xargs -r kill -9 2>/dev/null
        # 2) 포트 해제 대기
        for i in \$(seq 1 20); do
            ss -ltn 2>/dev/null | grep -q ':$PORT ' || break
            sleep 0.5
        done
        if ss -ltn 2>/dev/null | grep -q ':$PORT '; then echo 'PORT_BUSY'; exit 0; fi
        # 3) 기동
        rm -f $LOG
        nohup taskset -c 0-3 $REMOTE_BIN worker --port $PORT --nthreads ${NTHREADS} ${WORKER_EXTRA} > $LOG 2>&1 &
        # 4) LISTEN 확인
        for i in \$(seq 1 20); do
            sleep 0.5
            ss -ltn 2>/dev/null | grep -q ':$PORT ' && { echo OK; exit 0; }
        done
        echo NO_LISTEN
    " 2>/dev/null || echo SSH_FAIL)

    case "$ok" in
        OK)        echo "기동 완료 (port $PORT)" ;;
        PORT_BUSY) echo "[중단] 포트 $PORT 가 해제되지 않음"; fail=1 ;;
        NO_LISTEN) echo "[중단] 기동했으나 LISTEN 안 됨"; fail=1 ;;
        *)         echo "[중단] SSH 실패"; fail=1 ;;
    esac
done

[ "$fail" -eq 0 ] || { echo; echo "워커 기동 실패. 위 노드를 확인하세요."; exit 1; }
echo "워커 ${#HOSTS[@]}대 준비 완료"
