"""주어진 분할의 예측 TTFT 만 계산한다 (DP 탐색 없음).

용도: AxisCert 의 축별 paired residual bound `δ_x` 추정.

    e_x(q;q0) = log[T_meas(q)/T_meas(q0)] − log[T_pred(q)/T_pred(q0)]

절대 오차(현재 S=7212 에서 30%)는 대부분 공통 모드라 비율에서 상쇄된다.
게이트에 필요한 것은 절대 정확도가 아니라 **계획 간 비교의 교정 정확도**다.

DP 를 부르지 않으므로 다른 벤치가 도는 중에도 안전하다(root 부하 ~0).

    python3 prefill_bench/eval_plans.py --S 1789 --N 8 --B 32 \
        --plan P0=8,8,8,8,8,8,8,8 --plan P1=8,7,8,6,8,9,9,9 ...
"""
import argparse, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derivepp as D


def predict(sizes, S, N, B, model):
    d, d_ff, d_kv, nLayers, vocab = D.MODELS[model]
    assert sum(sizes) == 2 * nLayers, f"서브블록 합 {sum(sizes)} != {2*nLayers}"
    assert len(sizes) == N
    S_exec = 4 * ((S + 3) // 4)
    geom = D.chunk_geometry(S_exec, B)
    M = len(geom)
    nodes = D.STAGE_NODES[:N]
    AF = D.costs_for_nodes(None, nodes, geom, d, d_ff, d_kv)
    head = {n: [D._grid(D.EXEC_PROF[n]["c0"], b, l) for (_u, b, l, _a) in geom]
            for n in nodes}
    # 경계 누적 -> [a,bnd)
    bnds, a = [], 0
    for s in sizes:
        bnds.append((a, a + s)); a += s

    rows, links = [], []
    for k, n in enumerate(nodes):
        att0, att, ff = AF[n]
        hd = head[n] if k == N - 1 else None
        a, e = bnds[k]
        rows.append([D.stage_cost(att, ff, a, e, j, hd, att0) for j in range(M)])
        links.append([0.0] * M if k == 0 else
                     [D.link_cost(nodes[k-1], n, b, d) for (_u, b, _l, _a) in geom])
    return D.makespan(rows, links, M), M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default=os.environ.get("CAL_DIR", "."))
    ap.add_argument("--S", type=int, default=1789)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--ref", default="P0", help="기준 계획 q0")
    ap.add_argument("--plan", action="append", default=[], help="이름=8,8,...")
    ap.add_argument("--meas", action="append", default=[],
                    help="이름=측정ms (여러 회차면 쉼표로)")
    args = ap.parse_args()

    D.EXEC_PROF = D.load_perlayer_calibration(args.cal) or D.load_exec_calibration(args.cal)
    if not D.EXEC_PROF:
        sys.exit(f"레이어별 캘리브레이션 없음: {args.cal}")
    print(f"# 노드 {len(D.EXEC_PROF)}대, S={args.S} N={args.N} B={args.B}")

    pred, meas = {}, {}
    for p in args.plan:
        k, v = p.split("=", 1)
        pred[k], M = predict([int(x) for x in v.split(",")], args.S, args.N, args.B, args.model)
    for m in args.meas:
        k, v = m.split("=", 1)
        xs = [float(x) for x in v.split(",")]
        meas[k] = sum(xs) / len(xs)          # 기하평균이 아니라 산술평균: 회차 노이즈 대칭 가정

    print(f"M={M} chunks")
    r = args.ref
    print(f"\n{'plan':<6}{'예측(s)':>10}{'측정(s)':>10}{'R̂':>8}{'R':>8}{'e=lnR−lnR̂':>12}")
    es = []
    for k in sorted(pred):
        line = f"{k:<6}{pred[k]:10.3f}"
        if k in meas and r in meas and r in pred:
            Rh = pred[r] / pred[k]                     # 예측 speedup
            Rm = meas[r] / meas[k]                     # 실측 speedup
            e = math.log(Rm) - math.log(Rh)
            if k != r: es.append(abs(e))
            line += f"{meas[k]/1e3:10.3f}{Rh:8.3f}{Rm:8.3f}{e:12.4f}"
        print(line)
    if es:
        es.sort()
        print(f"\nδ 추정 (|e| 기준):  max {max(es):.4f}  "
              f"→ 배수로 {math.exp(max(es)):.3f}×")
        print(f"  즉 이 축에서 예측 speedup 은 실제와 최대 "
              f"{(math.exp(max(es))-1)*100:.1f}% 어긋난다.")
