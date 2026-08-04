#!/usr/bin/env bash
# 대역폭 제한 스윕용. root와 모든 워커의 egress를 tc로 제한한다.
# 체제 경계(phase diagram의 bandwidth 축)를 만들기 위한 도구.
#
#   bash net_shape.sh set 100        # 100 Mbit/s 로 제한
#   bash net_shape.sh clear          # 해제
#   bash net_shape.sh show
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"

IFACE="${IFACE:-$(ip route get 8.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')}"
: "${IFACE:?네트워크 인터페이스를 찾지 못했습니다. IFACE=eth0 형태로 지정하세요.}"

ACTION="${1:?usage: net_shape.sh {set <mbit>|clear|show}}"
MBIT="${2:-}"

apply_local() {
  case "$1" in
    set)
      sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
      sudo tc qdisc add dev "$IFACE" root tbf \
        rate "${2}mbit" burst 32kbit latency 400ms
      ;;
    clear) sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true ;;
    show)  tc qdisc show dev "$IFACE" ;;
  esac
}

apply_remote() {
  local host="$1"; shift
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_USER@$host" \
    "IFACE=\$(ip route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\") print \$(i+1); exit}');
     case '$1' in
       set)   sudo tc qdisc del dev \$IFACE root 2>/dev/null || true;
              sudo tc qdisc add dev \$IFACE root tbf rate ${2:-0}mbit burst 32kbit latency 400ms ;;
       clear) sudo tc qdisc del dev \$IFACE root 2>/dev/null || true ;;
       show)  tc qdisc show dev \$IFACE ;;
     esac" || echo "  [warn] $host 적용 실패"
}

case "$ACTION" in
  set)
    : "${MBIT:?rate(mbit)를 지정하세요}"
    echo "== egress ${MBIT}mbit 제한: local($IFACE) + ${#ALL_WORKERS[@]} workers =="
    apply_local set "$MBIT"
    for h in "${ALL_WORKERS[@]}"; do echo "-- $h"; apply_remote "$h" set "$MBIT"; done
    ;;
  clear)
    echo "== 제한 해제 =="
    apply_local clear
    for h in "${ALL_WORKERS[@]}"; do echo "-- $h"; apply_remote "$h" clear; done
    ;;
  show)
    echo "== local($IFACE) =="; apply_local show
    for h in "${ALL_WORKERS[@]}"; do echo "== $h =="; apply_remote "$h" show; done
    ;;
  *) echo "usage: net_shape.sh {set <mbit>|clear|show}" >&2; exit 2 ;;
esac
