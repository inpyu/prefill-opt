#!/usr/bin/env python3
"""DerivePP planner — 완료시각 DP 로 (N, B, 분할) 을 공동 결정한다.

research/16-derivepp.md §2~§3 의 구현.

입력은 **캘리브레이션 결과와 모델 구조뿐**이다. end-to-end 스윕을 하지 않는다.
  - prefill_bench/calibrate.cpp 가 만든 노드별 TSV
  - prefill_bench/bench_link.cpp 가 만든 링크 측정
  - 모델 config (d, d_ff, d_kv, L, vocab)

핵심 recurrence (§3.1):
    F[k][j] = max( F[k-1][j] + D_k(x_j),   # 앞 스테이지가 j 를 넘겨준 시각
                   F[k][j-1] )             # 이 스테이지가 j-1 을 끝낸 시각
              + C[k][j]

Σ 는 가법이고 Λ 는 max 라 스칼라 DP 로 분해되지 않는다. 완료시각 **벡터**를
DP 상태로 들고 가며 component-wise dominance 로 가지치기한다(보조정리 1).

    python3 prefill_bench/derivepp.py --S 7212
    python3 prefill_bench/derivepp.py --selftest      # brute force 대조
"""
import argparse, glob, os, sys, math
import numpy as np
from bisect import bisect_left

# ── 모델 구조 (측정이 아니라 config 에서 온다) ──────────────────────────
MODELS = {
    # name:        d,    d_ff,   d_kv, nLayers, vocab
    "llama3-8b":  (4096, 14336,  1024, 32, 128256),
    "llama3.2-3b":(3072,  8192,  1024, 28, 128256),
    "llama2-13b": (5120, 13824,  5120, 40,  32000),
    "llama3.1-70b":(8192,28672,  1024, 80, 128256),
}

# ── 클러스터 토폴로지 (측정으로 확인, research/16 §7.4) ────────────────
# 파이프라인 스테이지 순서. root 가 stage0.
STAGE_NODES = ["root", "113", "166", "191", "103", "94", "104", "54"]
# 스위치 그룹. 군 간 경로는 100 Mb 업링크를 지난다.
SWITCH = {"root": "A", "113": "A", "166": "A", "191": "A",
          "103": "B", "94": "B", "104": "B", "54": "B"}
LINK = {   # (alpha_ms, beta_MBps)
    "intra": (0.084, 115.7),
    "inter": (0.126,  11.7),
}
BYTES_PER_ELEM = 34.0 / 32.0      # q80 활성값

EXEC_PROF = {}


def load_calibration(d):
    """노드별 P_q(shape,B) 와 P_a(B,prefix) 를 GFLOPS 로 읽는다."""
    prof = {}
    for f in sorted(glob.glob(os.path.join(d, "cal_*.tsv"))):
        node = os.path.basename(f)[4:-4]
        Pq, Ta = {}, {}
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if p[0] == "Tq":
                Pq[(p[1], int(p[4]))] = float(p[6]) * 1e9      # FLOP/s
            elif p[0] == "Ta":
                Ta[(int(p[1]), int(p[2]))] = float(p[4]) * 1e9
        if Pq and Ta:
            prof[node] = (Pq, Ta)
    return prof


def load_perlayer_calibration(d):
    """calibrate_perlayer.cpp 의 레이어별 비용을 읽는다 (research/16 §7.6c).

    회귀가 아니라 **한 forward 안에서 레이어별로 직접 수집**한 값이다.
    모든 레이어가 같은 열·메모리 상태에서 측정되므로 시간 드리프트가
    레이어 효과로 오인되지 않는다(회귀는 1.87배 과소했다).

    ⚠️ layer 0 은 랜덤 편차가 아니다. 8노드 전부에서 `L0/평균 = 2.01~2.09배`
    로 재현되고 FFN 에는 나타나지 않는다. 매 forward 의 첫 attention 이 부담하는
    stage-start 비용(attention scratch cold-cache 로 추정)이다. 따라서

        C_att(n) = A_start + (n-1)*A_steady      (n >= 1)

    로 모델링한다. 평균을 모든 레이어에 적용하면 start 비용을 n 번 세게 된다.
    """
    prof = {}
    accLayer = {}
    for f in sorted(glob.glob(os.path.join(d, "pl_*.tsv"))):
        node = os.path.basename(f)[3:-4]
        acc = {"Latt": {}, "Lff": {}, "c0": {}}
        accLayer[node] = {}
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if not p:
                continue
            if p[0] in ("PLatt", "PLff"):
                key = (int(p[1]), int(p[2]))
                k = "Latt" if p[0] == "PLatt" else "Lff"
                acc[k].setdefault(key, []).append(float(p[4]))
                lay = accLayer[node].setdefault(key, {"att": {}, "ff": {}})
                lay["att" if p[0] == "PLatt" else "ff"][int(p[3])] = float(p[4])
            elif p[0] == "PLc0":
                acc["c0"][(int(p[1]), int(p[2]))] = float(p[4])
        if acc["Latt"]:
            # layer 0 = start,  layer >= 1 = steady
            start, steady, ffm = {}, {}, {}
            for (B, P), lay in accLayer.get(node, {}).items():
                a0 = lay["att"].get(0)
                arest = [v for l, v in lay["att"].items() if l > 0]
                f_all = list(lay["ff"].values())
                start[(B, P)] = a0 if a0 is not None else (sum(arest)/len(arest))
                steady[(B, P)] = (sum(arest)/len(arest)) if arest else start[(B, P)]
                ffm[(B, P)] = sum(f_all)/len(f_all)
            prof[node] = {
                "Latt": steady,          # 정상 상태 (하위호환)
                "Astart": start,
                "Asteady": steady,
                "Lff":  ffm,
                "c0":   acc["c0"],
            }
    return prof


def load_exec_calibration(d):
    """calibrate_exec.cpp 의 레이어 단위 비용을 읽는다 (research/16 §7.6 A).

    `Lall B prefix sec` — 레이어 1개분(att+ff 합) 시간이다.
    op 단위 처리율(P_q, P_a)과 달리 **FLOPs 로 나누지 않는다.**
    나누는 순간 executor 의 step 동기화와 thread imbalance 가 사라진다.
    """
    prof = {}
    for f in sorted(glob.glob(os.path.join(d, "ce_*.tsv"))):
        node = os.path.basename(f)[3:-4]
        tab = {}
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if not p or p[0] not in ("Lall", "Latt", "Lff"):
                continue
            tab.setdefault(p[0], {})[(int(p[1]), int(p[2]))] = float(p[3])
        if tab:
            prof[node] = tab
    return prof


def _nearest(keys, v):
    ks = sorted(keys)
    i = bisect_left(ks, v)
    if i == 0: return ks[0]
    if i >= len(ks): return ks[-1]
    return ks[i-1] if v - ks[i-1] <= ks[i] - v else ks[i]


def Pq_of(prof, node, shape, B):
    """측정 격자에서 가장 가까운 B 의 처리율.

    ⚠️ 매끄러운 tau+kappa*B 로 적합하지 않는다. B=24 저하가 8/8 노드에서
    재현되므로(research/16 §7.2b) c(B) 는 계단이고, 적합은 그 계단을 지운다.
    """
    Pq, _ = prof[node]
    bs = [b for (s, b) in Pq if s == shape]
    return Pq[(shape, _nearest(bs, B))]


def Pa_of(prof, node, B, prefix):
    _, Ta = prof[node]
    bs = sorted({b for (b, p) in Ta})
    ps = sorted({p for (b, p) in Ta})
    return Ta[(_nearest(bs, B), _nearest(ps, prefix))]


def chunk_geometry(S_exec, B):
    """§2.2 — 부분 청크와 causal 위치를 정확히 반영한다."""
    out = []
    u = 0
    while u < S_exec:
        b = min(B, S_exec - u)
        A = b * u + b * (b + 1) // 2          # 이 청크의 (query,key) 쌍 수
        out.append((u, b, u + b, A))          # u_j, b_j, l_j, A_j
        u += b
    return out


def subblock_costs(prof, node, geom, d, d_ff, d_kv):
    """서브블록 1개(att / ff)의 마이크로배치별 비용 [초]."""
    att, ff = [], []
    for (u, b, l, A) in geom:
        # attention 서브블록: Q/K/V/O projection + attention core
        f_proj = 4.0 * b * d * (d + d_kv)
        f_core = 4.0 * d * A
        att.append(f_proj / Pq_of(prof, node, "proj", b)
                   + f_core / Pa_of(prof, node, b, l))
        # FFN 서브블록: gate/up 은 (d_ff x d), down 은 (d x d_ff)
        f_up   = 4.0 * b * d * d_ff
        f_down = 2.0 * b * d * d_ff
        ff.append(f_up   / Pq_of(prof, node, "ffn13", b)
                  + f_down / Pq_of(prof, node, "ffn2", b))
    return att, ff


def subblock_costs_exec2(tab, geom):
    """att 를 start / steady 로 나눠 돌려준다 (§7.6c).

    반환: (att_start, att_steady, ff)  — 각각 마이크로배치별 리스트
    """
    A0, A1, F = tab["Astart"], tab["Asteady"], tab["Lff"]
    return ([_grid(A0, b, l) for (_u, b, l, _a) in geom],
            [_grid(A1, b, l) for (_u, b, l, _a) in geom],
            [_grid(F,  b, l) for (_u, b, l, _a) in geom])


def subblock_costs_exec(tab, geom, _unused=None):
    """executor 캘리브레이션에서 att/ff 서브블록 비용을 읽는다.

    `calibrate_exec` 가 production 과 **같은 계측기**
    (`getLastForwardOpBreakdown`)로 `Latt(B,prefix)` 와 `Lff(B,prefix)` 를 직접 낸다.
    따라서 비율 가정이 없다(§7.6b 에서 `att_share` 고정이 −38~−52 % 과소예측과
    순위 역전을 만들었다).

    실측 확인: `Lff` 는 prefix 에 대해 평평하고(22.3~23.4 ms, 편차 4.6 %)
    `Latt` 는 6.7 -> 48.3 ms 로 7.2배 증가한다 — 물리적 예상과 일치한다.
    """
    A, F = tab["Latt"], tab["Lff"]
    att = [_grid(A, b, l) for (u, b, l, _a) in geom]
    ff  = [_grid(F, b, l) for (u, b, l, _a) in geom]
    return att, ff


def _grid(tab, B, prefix):
    """`B` 는 최근접, `prefix` 는 선형 보간.

    두 축의 성질이 다르다.
      - `B`: GEMM 타일 때문에 **계단**이다(B=24 저하가 8/8 노드 재현, §7.2b).
        보간하면 계단이 지워지므로 최근접을 쓴다.
      - `prefix`: KV 스캔 길이라 **매끄럽다**. 격자가
        {128,512,2048,4096,8192} 로 성기어서 최근접을 쓰면 큰 오차가 난다.
        예: l=6000 이면 최근접 4096 -> 24.18 ms, 실제는 약 35 ms (31 % 과소).
        S=7,212 에서는 상당수 청크가 이 구간에 떨어져 −40 % 대 과소예측을 만들었다.
    """
    bs = sorted({b for (b, _) in tab})
    ps = sorted({p for (_, p) in tab})
    b = _nearest(bs, B)
    if prefix <= ps[0]:
        return tab[(b, ps[0])]
    if prefix >= ps[-1]:
        return tab[(b, ps[-1])]
    i = 0
    while i + 1 < len(ps) and ps[i + 1] < prefix:
        i += 1
    p0, p1 = ps[i], ps[i + 1]
    t = (prefix - p0) / float(p1 - p0)
    return tab[(b, p0)] * (1.0 - t) + tab[(b, p1)] * t


def costs_for_nodes(prof, nodes, geom, d, d_ff, d_kv):
    """서브블록 비용의 **단일 진입점**.

    DP 와 brute force 가 서로 다른 비용 소스를 쓰면 정확성 대조가 무의미해진다.
    executor 캘리브레이션(§7.6 A)이 모든 노드에 있으면 그것을 쓰고,
    하나라도 없으면 전부 op 단위로 되돌린다 — 두 소스를 섞으면 노드 간 비교가 무효다.
    """
    if EXEC_PROF and all(n in EXEC_PROF and "Astart" in EXEC_PROF[n] for n in nodes):
        # ⚠️ A_start 는 **캘리브레이터 인공물**이다.
        #
        # 벤치는 forward() 를 반복 호출하고, 매 호출의 첫 attention 이 식은
        # scratch 를 만난다(layer 0 이 2배). production 은 wave 파이프라인에서
        # 마이크로배치를 연속 처리하므로 scratch 가 따뜻하다.
        #
        # production 청크별 덤프로 확인: attnCore = 11.2 ms + (위치 비례항),
        # R^2 = 0.996, 절편은 평균 청크의 **1.8 %** 뿐이다. 첫 청크들은 오히려
        # 선형 예측보다 낮다(0.91~0.95). 따라서 start 항을 쓰지 않는다.
        return {n: (lambda t: (t[1], t[1], t[2]))(subblock_costs_exec2(EXEC_PROF[n], geom))
                for n in nodes}
    if EXEC_PROF and all(n in EXEC_PROF and "Latt" in EXEC_PROF[n] for n in nodes):
        # start/steady 가 없는 옛 형식: start = steady
        return {n: (lambda t: (t[0], t[0], t[1]))(subblock_costs_exec(EXEC_PROF[n], geom))
                for n in nodes}
    return {n: (lambda t: (t[0], t[0], t[1]))(subblock_costs(prof, n, geom, d, d_ff, d_kv))
            for n in nodes}


def link_cost(src, dst, b, d):
    a, beta = LINK["intra" if SWITCH[src] == SWITCH[dst] else "inter"]
    x = b * d * BYTES_PER_ELEM
    return (a + x / beta / 1e3) / 1e3          # ms -> s  (beta 는 MB/s)


def n_att_in(a, bnd):
    """[a,bnd) 안의 att 서브블록(짝수 인덱스) 개수 — 닫힌 식.

    루프로 세면 (i,e) 조합마다 O(e-i) 가 되어 플래너가 끝나지 않는다.
    """
    return ((bnd - a) + (1 if a % 2 == 0 else 0)) // 2


def stage_cost_vec(attA, ffA, headA, a, bnd, att0A=None):
    """구간 [a,bnd) 의 마이크로배치별 비용 벡터.

    att0A 가 주어지면 stage-start 비용을 **한 번만** 더한다:
        C_att(n) = A_start + (n-1)*A_steady
    평균을 n 번 곱하면 start 비용을 n 번 세게 된다(§7.6c).
    """
    na = n_att_in(a, bnd)
    nf = (bnd - a) - na
    if att0A is not None and na > 0:
        c = att0A + (na - 1) * attA + nf * ffA
    else:
        c = na * attA + nf * ffA
    return c if headA is None else c + headA


def stage_cost(att, ff, a, bnd, j, head=None, att0=None):
    """스칼라 버전(brute force 대조용). start 비용은 한 번만."""
    na = n_att_in(a, bnd)
    nf = (bnd - a) - na
    if att0 is not None and na > 0:
        c = att0[j] + (na - 1) * att[j] + nf * ff[j]
    else:
        c = na * att[j] + nf * ff[j]
    if head is not None:
        c += head[j]
    return c


def makespan(cost_rows, links, M):
    """§3.1 recurrence. cost_rows[k][j], links[k][j] (k>=1 에서 유입)."""
    N = len(cost_rows)
    prev = [0.0] * M
    for k in range(N):
        cur = [0.0] * M
        last = 0.0
        for j in range(M):
            ready = prev[j] + (links[k][j] if k > 0 else 0.0)
            start = ready if ready > last else last
            last = start + cost_rows[k][j]
            cur[j] = last
        prev = cur
    return prev[-1]


def dominates(a, b):
    """a ⪯ b : 모든 성분에서 a 가 작거나 같다 (보조정리 1)."""
    return bool(np.all(a <= b))


def plan_for(prof, S, N, B, model, pad4=True, keep=None, stats=None):
    d, d_ff, d_kv, nLayers, vocab = MODELS[model]
    L = 2 * nLayers
    if N > L:
        return None
    S_exec = 4 * ((S + 3) // 4) if pad4 else S
    geom = chunk_geometry(S_exec, B)
    M = len(geom)
    nodes = STAGE_NODES[:N]

    AF = costs_for_nodes(prof, nodes, geom, d, d_ff, d_kv)
    # 마지막 스테이지의 lm_head: 청크마다 마지막 행 1개 (research/16 §2.3)
    # 레이어 무관 항(embedding, final norm, lm_head).
    #
    # ⚠️ 이전에는 op 단위 처리율로 lm_head 를 계산했는데, executor 캘리브레이션과
    # 섞이는 소스 혼용이었다(§7.6b). 레이어별 수집은 c0 를 직접 재므로 그대로 쓴다.
    if EXEC_PROF and all(n in EXEC_PROF and "c0" in EXEC_PROF[n] for n in nodes):
        head = {n: [_grid(EXEC_PROF[n]["c0"], b, l) for (_u, b, l, _a) in geom]
                for n in nodes}
    else:
        head = {n: [2.0 * d * vocab / Pq_of(prof, n, "proj", 1) for _ in geom]
                for n in nodes}
    # 경계 전송
    lk = [[0.0] * M for _ in range(N)]
    for k in range(1, N):
        for j, (_, b, _, _) in enumerate(geom):
            lk[k][j] = link_cost(nodes[k-1], nodes[k], b, d)

    # DP: state = (완료벡터, 분할경계)
    # frontier[i] = 앞 i 개 서브블록을 지금까지의 스테이지에 배정한 Pareto 집합
    lk_np = [np.asarray(r, dtype=np.float64) for r in lk]

    # DP 상태를 **행렬로 묶는다**.
    #
    # 상태마다 numpy 를 부르면 (i,e,state) 조합이 8 x 64 x 64 x Phi ~ 3e7 회가 되어
    # 호출 오버헤드가 지배한다(실측: 40분에도 미완). 같은 (i,e) 를 공유하는 상태는
    # c 와 C 가 동일하므로 (Phi, M) 행렬 한 번으로 처리한다.
    #
    #   frontier[i] = (V, cuts)   V: (n_i, M) 완료시각 행렬,  cuts: 경계 튜플 리스트
    frontier = {0: (np.zeros((1, M)), [()])}
    peak = 0
    for k in range(N):
        att0, att, ff = AF[nodes[k]]
        hd = head[nodes[k]] if k == N - 1 else None
        remain = N - k - 1
        lkv = lk_np[k]
        att0A = np.asarray(att0, dtype=np.float64)
        attA = np.asarray(att, dtype=np.float64)
        ffA = np.asarray(ff, dtype=np.float64)
        hdA = np.asarray(hd, dtype=np.float64) if hd is not None else None
        ccache = {}
        acc = {}
        for i, (V, cuts) in frontier.items():
            for e in range(i + 1, L - remain + 1):
                # 비용은 (n_att, n_ff) 에만 의존하므로 캐시한다.
                key = (n_att_in(i, e), (e - i) - n_att_in(i, e))
                cc = ccache.get(key)
                if cc is None:
                    c = stage_cost_vec(attA, ffA, hdA, i, e, att0A)
                    cc = (c, np.cumsum(c))
                    ccache[key] = cc
                c, C = cc
                # last_j = C_j + max_{i<=j}(a_i - C_i),  a = V + lk + c   (행 단위 prefix max)
                cur = C + np.maximum.accumulate(V + lkv + c - C, axis=1)
                if e in acc:
                    acc[e][0].append(cur)
                    acc[e][1].extend(cut + (e,) for cut in cuts)
                else:
                    acc[e] = ([cur], [cut + (e,) for cut in cuts])
        frontier = {}
        for e, (mats, cuts) in acc.items():
            V = np.vstack(mats)
            # Pareto 가지치기(보조정리 1). 마지막 성분 기준 정렬 후,
            # 이미 남긴 것 중 하나라도 전 성분에서 작거나 같으면 버린다.
            order = np.argsort(V[:, -1], kind="stable")
            # ⚠️ kept 를 리스트로 두고 매 후보마다 np.asarray 로 재구성하면
            # O(Phi^2 * M) 복사가 되어 프로파일의 68 % 를 차지한다(asarray 629k회, 63s).
            # 미리 할당하고 뷰로 비교한다.
            kept = np.empty_like(V)
            nk = 0
            keptC = []
            for idx in order:
                v = V[idx]
                if nk and bool(np.any(np.all(kept[:nk] <= v, axis=1))):
                    continue
                kept[nk] = v
                nk += 1
                keptC.append(cuts[idx])
            if keep and nk > keep:
                nk = keep
                keptC = keptC[:keep]
            frontier[e] = (kept[:nk].copy(), keptC)
            peak = max(peak, nk)
    if stats is not None:
        stats["peak_frontier"] = max(stats.get("peak_frontier", 0), peak)
    if L not in frontier or len(frontier[L][1]) == 0:
        return None
    V, cuts = frontier[L]
    bi = int(np.argmin(V[:, -1]))
    return {"S": S, "S_exec": S_exec, "N": N, "B": B, "M": M,
            "T": float(V[bi, -1]), "cuts": cuts[bi], "nodes": nodes}


def brute_force(prof, S, N, B, model, pad4=True):
    """작은 문제에서 DP 정확성을 대조한다 (§0.4)."""
    import itertools
    d, d_ff, d_kv, nLayers, vocab = MODELS[model]
    L = 2 * nLayers
    S_exec = 4 * ((S + 3) // 4) if pad4 else S
    geom = chunk_geometry(S_exec, B); M = len(geom)
    nodes = STAGE_NODES[:N]
    AF = costs_for_nodes(prof, nodes, geom, d, d_ff, d_kv)
    # 레이어 무관 항(embedding, final norm, lm_head).
    #
    # ⚠️ 이전에는 op 단위 처리율로 lm_head 를 계산했는데, executor 캘리브레이션과
    # 섞이는 소스 혼용이었다(§7.6b). 레이어별 수집은 c0 를 직접 재므로 그대로 쓴다.
    if EXEC_PROF and all(n in EXEC_PROF and "c0" in EXEC_PROF[n] for n in nodes):
        head = {n: [_grid(EXEC_PROF[n]["c0"], b, l) for (_u, b, l, _a) in geom]
                for n in nodes}
    else:
        head = {n: [2.0 * d * vocab / Pq_of(prof, n, "proj", 1) for _ in geom]
                for n in nodes}
    lk = [[0.0] * M for _ in range(N)]
    for k in range(1, N):
        for j, (_, b, _, _) in enumerate(geom):
            lk[k][j] = link_cost(nodes[k-1], nodes[k], b, d)
    best = None
    for cuts in itertools.combinations(range(1, L), N - 1):
        bnds = (0,) + cuts + (L,)
        rows = []
        for k in range(N):
            att0, att, ff = AF[nodes[k]]
            hd = head[nodes[k]] if k == N - 1 else None
            rows.append([stage_cost(att, ff, bnds[k], bnds[k+1], j, hd, att0)
                         for j in range(M)])
        T = makespan(rows, lk, M)
        if best is None or T < best[0]:
            best = (T, bnds[1:])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default=os.environ.get("CAL_DIR", "."))
    ap.add_argument("--S", type=int, default=7212)
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--N", type=int, default=0, help="0 이면 모든 N 을 평가")
    ap.add_argument("--keep", type=int, default=0, help=">0 이면 frontier 상한(2단계 모드)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    global EXEC_PROF
    # 레이어별 직접 수집(§7.6c)을 최우선으로 쓴다. 없으면 이전 executor 회귀,
    # 그것도 없으면 op 단위로 내려간다.
    EXEC_PROF = load_perlayer_calibration(args.cal) or load_exec_calibration(args.cal)
    prof = load_calibration(args.cal)
    if EXEC_PROF:
        print(f"# 레이어별 캘리브레이션 {len(EXEC_PROF)}대 사용 (§7.6c)")
    if not prof and not EXEC_PROF:
        sys.exit(f"캘리브레이션을 찾을 수 없다: {args.cal}/(pl_*|ce_*|cal_*).tsv")
    src = EXEC_PROF or prof
    print(f"# 노드 {len(src)}대: {', '.join(sorted(src))}")

    if args.selftest:
        # L 을 줄여서 brute force 와 대조한다.
        global MODELS
        d, dff, dkv, nl, vocab = MODELS["llama3-8b"]
        MODELS["tiny"] = (d, dff, dkv, 4, vocab)      # L = 8
        ok = True
        for N in (2, 3, 4):
            for B in (16, 32):
                st = {}
                dp = plan_for(prof, 256, N, B, "tiny", stats=st)
                bf = brute_force(prof, 256, N, B, "tiny")
                same = abs(dp["T"] - bf[0]) < 1e-12
                ok &= same
                print(f"  N={N} B={B}  DP={dp['T']*1e3:9.3f} ms  BF={bf[0]*1e3:9.3f} ms  "
                      f"{'일치' if same else '불일치'}  frontier<={st['peak_frontier']}")
        print("selftest:", "통과" if ok else "실패")
        return

    Ns = [args.N] if args.N else [1, 2, 3, 4, 5, 6, 7, 8]
    Bs = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    best = None
    print(f"{'N':>3s}{'B':>5s}{'M':>6s}{'예측 TTFT(s)':>14s}{'frontier':>10s}  분할")
    for N in Ns:
        if N > len(STAGE_NODES): continue
        for B in Bs:
            st = {}
            p = plan_for(prof, args.S, N, B, args.model,
                         keep=args.keep or None, stats=st)
            if not p: continue
            cuts = p["cuts"]
            sizes = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))]
            print(f"{N:3d}{B:5d}{p['M']:6d}{p['T']:14.2f}{st['peak_frontier']:10d}  "
                  f"{','.join(map(str, sizes))}")
            if best is None or p["T"] < best["T"]:
                best = p
    if best:
        cuts = best["cuts"]
        sizes = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))]
        print(f"\n최적: N={best['N']} B={best['B']} M={best['M']} "
              f"예측 {best['T']:.2f}s  분할 {','.join(map(str, sizes))}")


if __name__ == "__main__":
    main()
