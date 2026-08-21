"""CP2 island 의 capacity-contraction 상한 (구현 전 게이트).

━━ 무엇을 주장하고 무엇을 주장하지 않는가 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Balanced-capacity contraction lemma.**
동일한 물리 worker `N` 대로 이루어진 **이미 균형 잡힌** 파이프라인을 가정한다.
각 physical stage 의 정상상태 service time 이 `c` 이고, 두 worker 를 하나의
CP/TP group 으로 묶어도 group 의 service time 이 `c` 보다 **작아지지 않는다면**,
logical stage 수를 `N → N−1` 로 줄여 얻는 이상적 speedup 은

    T_N / T_{N−1}  ≤  (M + N − 1) / (M + N − 2)

이며, 순수 bubble 이득은  **1 / (M + N − 2)** 이다.

⚠️ 이 보조정리는 다음 경우 **적용되지 않는다**. 논문에서 반드시 명시할 것.
    - 기존 PP 가 불균형인 경우
    - CP/TP 가 stage service time 을 실제로 줄이는 경우
      (cache/메모리 대역폭/커널 효율이 바뀌어 `c` 자체가 내려가는 경우 포함)
    - global TP 처럼 파이프라인 의존 그래프 자체가 달라지는 경우
따라서 "모든 stage 내 병렬화가 같은 이유로 죽는다"는 **너무 넓은 주장**이다.
CoRePP(§12.3)는 logical stage 수가 그대로여서 선형 확장 시 이상적 이득이 0 이고,
CP2 island 는 위 조건이 성립할 때만 bubble 상한만 남는다.

━━ 통신 회계 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

앞선 판(B·d 한 번)은 **과소계상**이었다. helper 가 모든 `B` query 에 대해 자기
KV shard 를 처리하려면 레이어마다 최소한 다음이 필요하다.

    owner → helper : Q            ≈ B·d
    helper → owner : partial O    ≈ B·d          (+ softmax 통계, 무시할 수준)
    owner → helper : 새 K/V append ≈ B·2·d_kv

Q 전송과 partial 회수는 **서로 다른 의존 단계**라 full-duplex 로 지울 수 없다.
직렬로 더한다.

한편 query 를 `B/2` 씩 나누면 각 query 가 반대편 KV shard 도 봐야 하므로
prefix 에 비례하는 KV ring/all-gather 가 되살아난다. "B/2 query 전송 +
KV 이동 없음" 을 동시에 만족하는 프로토콜은 없다.
"""
import argparse

D_MODEL, D_KV, BYTES = 4096, 1024, 34.0 / 32.0     # llama3-8B, q80 활성값
INTRA, INTER = 115.7, 11.7                          # MB/s (§7.4 실측)

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=56, help="청크 수")
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--N", type=int, default=8, help="PP-only stage 수 = 노드 수")
ap.add_argument("--c-ms", type=float, default=396.8, help="stage 정상상태 service time")
ap.add_argument("--layers", type=int, default=32)
a = ap.parse_args()

Nc = a.N - 1
T_pp = (a.M + a.N - 1) * a.c_ms
T_ideal = (a.M + Nc - 1) * a.c_ms                   # 통신 0, service time 불변
print(f"PP{a.N}      makespan {T_pp:9.0f} ms   (M={a.M}, c={a.c_ms:.1f} ms)")
print(f"PP{Nc}+CP2  이상적    {T_ideal:9.0f} ms   "
      f"→ {T_pp/T_ideal:.4f}x  (+{1/(a.M+a.N-2)*100:.2f}% = 1/(M+N−2))")

# island 이 담는 레이어: 총 2L 서브블록을 Nc logical stage 로 나누고,
# island 은 그중 하나를 2노드가 처리한다.
lay_island = (2 * a.layers / Nc) / 2

# 레이어당 wire payload — 의존 단계가 달라 직렬로 더한다.
elems = a.B * D_MODEL + a.B * D_MODEL + a.B * 2 * D_KV
per_layer_MB = elems * BYTES / 1e6
print(f"\nisland 담당 레이어 ~{lay_island:.2f}")
print(f"레이어당 payload: Q {a.B*D_MODEL} + partialO {a.B*D_MODEL} + KV {a.B*2*D_KV}"
      f" = {elems} 원소 = {per_layer_MB*1e3:.0f} KB")

for tag, bw in (("intra-switch", INTRA), ("inter-switch", INTER)):
    comm = lay_island * per_layer_MB / bw * 1e3
    mk = (a.M + Nc - 1) * (a.c_ms + comm)
    print(f"\n  {tag} ({bw} MB/s): 통신 {comm:6.1f} ms/chunk  "
          f"makespan {mk:9.0f} ms  → {T_pp/mk:.4f}x ({(T_pp/mk-1)*100:+.2f}%)")
