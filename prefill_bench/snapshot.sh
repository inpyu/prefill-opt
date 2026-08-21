#!/usr/bin/env bash
# 측정 산출물을 불변 run-id 디렉터리로 고정한다 (reproducibility freeze).
#
# 왜 필요한가:
#   레이어별 캘리브레이션(pl_*.tsv)을 한 번 유실해 delta_p 산출이 막혔다.
#   "재수집 가능"으로는 부족하다 — 논문 수치가 **정확히 어떤 원자료에서 나왔는지**
#   재현 가능해야 한다. 클러스터 상태(온도/주파수/바이너리)는 두 번 같지 않다.
#
#   bash prefill_bench/snapshot.sh <run-id> <파일...>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"

RUN="${1:?usage: snapshot.sh <run-id> <file...>}"; shift
DIR="$HERE/../artifacts/$RUN"
mkdir -p "$DIR/raw" "$DIR/derived"

for f in "$@"; do [ -e "$f" ] && cp -a "$f" "$DIR/raw/"; done

# manifest — 코드 상태를 되돌릴 수 있어야 한다. dirty 면 패치 자체를 남긴다.
SHA=$(git -C "$HERE/.." rev-parse HEAD 2>/dev/null || echo none)
git -C "$HERE/.." diff HEAD > "$DIR/dirty.patch" 2>/dev/null || true
[ -s "$DIR/dirty.patch" ] || rm -f "$DIR/dirty.patch"
cat > "$DIR/manifest.json" <<JSON
{
  "run_id": "$RUN",
  "git_sha": "$SHA",
  "dirty": $([ -f "$DIR/dirty.patch" ] && echo true || echo false),
  "binary_md5": "$(md5sum "$HERE/../dllama" 2>/dev/null | awk '{print $1}')",
  "created_utc": "$(date -u +%FT%TZ)"
}
JSON

# 노드 상태 — 주파수 governor 와 온도는 결과를 바꾸므로 함께 남긴다.
{ echo "host	cores	governor	maxfreq	temp_c"
  for h in root "${ALL_WORKERS[@]}"; do
    if [ "$h" = root ]; then C="bash -c"; else C="ssh -o ConnectTimeout=5 ubuntu@$h"; fi
    $C 'echo -e "$(nproc)\t$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)\t$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo ?)\t$(awk "{printf \"%.1f\", \$1/1000}" /sys/class/thermal/thermal_zone0/temp)"' 2>/dev/null \
      | sed "s|^|$h\t|" || echo -e "$h\t(unreachable)"
  done
} > "$DIR/hosts.tsv"

{ echo "compiler: $(g++ --version | head -1)"; echo "kernel: $(uname -a)"; } > "$DIR/environment.txt"
( cd "$DIR" && find . -type f ! -name checksums.txt -exec md5sum {} + > checksums.txt )
echo "고정 완료: artifacts/$RUN  ($(find "$DIR/raw" -type f | wc -l) raw 파일)"
