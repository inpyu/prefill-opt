#!/usr/bin/env python3
"""깊이 분해(research/07)의 정확도 검증.

가설: 레이어 k 의 정규화 입력 phi(x_k) 만으로 상위 레이어 l>k 의 K_l, V_l 을
      선형 사상으로 재구성할 수 있다.

          K_l = A_l phi(x_k)

이 스크립트는 dllama 가 DLLAMA_CALIB_DIR 로 덤프한 raw f32 를 읽어
  1) 레이어별 최소제곱 적합과 재구성 오차 (dense A_l)
  2) 레이어 공유 저랭크 기저 A_l ~= U diag(d_l) V 의 오차와 비용 절감
를 측정한다. 2)가 SwiftKV 와의 차별점이다(research/07 §4).

  python3 fit_kv_projection.py <calib_dir> --k 16 --rank 256
"""
import argparse
import pathlib
import numpy as np


def load(path, cols):
    # float64 로 올린다. float32 로 4096x4096 정규방정식을 풀면 오차가 누적된다.
    a = np.fromfile(path, dtype=np.float32)
    if a.size % cols != 0:
        a = a[: (a.size // cols) * cols]
    return a.reshape(-1, cols).astype(np.float64)


def ls_fit(X, Y, ridge=1e-4):
    """min_A ||X A - Y||_F  (X: n x h, Y: n x d)  ->  A: h x d

    관측 행이 미지수(h)보다 적으면 미결정 시스템이라 유일해가 없다.
    lstsq 는 최소 노름 해를 주는데 그건 진짜 W 와 달라서 홀드아웃 오차가 남는다.
    -> 행 수가 h 의 3배 이상인지 확인할 것.
    """
    h = X.shape[1]
    XtX = X.T @ X
    XtX[np.diag_indices(h)] += ridge * np.trace(XtX) / h
    return np.linalg.solve(XtX, X.T @ Y)


def rel_err(Y, Yhat):
    return float(np.linalg.norm(Y - Yhat) / (np.linalg.norm(Y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir")
    ap.add_argument("--k", type=int, default=16, help="분해 시작 레이어")
    ap.add_argument("--rank", type=int, default=256, help="공유 기저 랭크")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--kvdim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args()

    d = pathlib.Path(args.calib_dir)
    xk_path = d / f"x_l{args.k:02d}.f32"
    if not xk_path.exists():
        raise SystemExit(f"없음: {xk_path}\n덤프가 레이어 {args.k} 를 포함하는지 확인하세요.")

    X = load(xk_path, args.hidden)
    n = X.shape[0]
    nTest = max(1, int(n * args.holdout))
    Xtr, Xte = X[:-nTest], X[-nTest:]
    print(f"phi(x_{args.k}): {X.shape}  (train {Xtr.shape[0]} / test {Xte.shape[0]})")
    if Xtr.shape[0] < args.hidden:
        print(f"  ⚠️  train 행({Xtr.shape[0]})이 hidden({args.hidden})보다 작습니다 — "
              f"ridge 로 정칙화되지만 프롬프트를 더 모으는 편이 좋습니다.")

    # ---- 0) 온전성 검사: l = k 에서는 오차가 0 에 가까워야 한다 ----
    # K_k = W_k^k phi(x_k) 이므로 같은 선형 사상을 복원하는 것이다.
    # 여기서 0 이 아니면 덤프가 잘못됐거나 행이 부족한 것이고,
    # 그 상태의 상위 레이어 수치는 판정에 쓸 수 없다.
    kp0 = d / f"k_l{args.k:02d}.f32"
    if kp0.exists():
        K0 = load(kp0, args.kvdim)
        m0 = min(len(X), len(K0))
        A0 = ls_fit(X[:m0][:-nTest], K0[:m0][:-nTest])
        e0 = rel_err(K0[:m0][-nTest:], X[:m0][-nTest:] @ A0)
        verdict = "OK" if e0 < 0.02 else ("행 부족 의심" if Xtr.shape[0] < 3 * args.hidden else "*** 덤프 이상 ***")
        print(f"\n[온전성] layer {args.k} 자기적합 rel.err = {e0:.5f}  ({verdict})")
        print(f"          학습행 {Xtr.shape[0]} / 미지수 {args.hidden} "
              f"= {Xtr.shape[0]/args.hidden:.1f}x  (3x 이상 권장)")
        if e0 >= 0.02:
            print("          ⚠️  이 값이 0.02 이상이면 아래 수치는 판정 근거로 쓸 수 없다.")

    # ---- 1) 레이어별 dense 적합 ----
    print(f"\n{'layer':>5} {'K rel.err':>10} {'V rel.err':>10}")
    results = {}
    kept = []
    for l in range(args.k, args.layers):
        kp, vp = d / f"k_l{l:02d}.f32", d / f"v_l{l:02d}.f32"
        if not (kp.exists() and vp.exists()):
            continue
        K, V = load(kp, args.kvdim), load(vp, args.kvdim)
        m = min(len(X), len(K), len(V))
        Ktr, Kte = K[:m][:-nTest], K[:m][-nTest:]
        Vtr, Vte = V[:m][:-nTest], V[:m][-nTest:]
        Ak = ls_fit(Xtr[:m - nTest], Ktr)
        Av = ls_fit(Xtr[:m - nTest], Vtr)
        ek, ev = rel_err(Kte, Xte @ Ak), rel_err(Vte, Xte @ Av)
        results[l] = (Ak, Av, ek, ev)
        kept.append(l)
        print(f"{l:>5} {ek:>10.4f} {ev:>10.4f}")

    if not kept:
        raise SystemExit("상위 레이어 덤프가 없습니다.")
    ek_all = np.mean([results[l][2] for l in kept])
    ev_all = np.mean([results[l][3] for l in kept])
    print(f"{'평균':>5} {ek_all:>10.4f} {ev_all:>10.4f}")

    # ---- 2) 레이어 공유 저랭크 기저 ----
    # 모든 레이어의 A_l 을 쌓아 공통 우측 부분공간 V 를 SVD 로 얻고,
    # 레이어별로는 diag(d_l) 만 남긴다.
    print(f"\n공유 저랭크 기저 (rank={args.rank})")
    stack = np.concatenate([results[l][0] for l in kept], axis=1)  # h x (d*nLayers)
    # 우측 특이벡터가 아니라, 입력측 부분공간을 찾는다: A_l = P B_l 형태
    U_in, S_in, _ = np.linalg.svd(stack, full_matrices=False)
    P = U_in[:, : args.rank]                     # h x rank  (입력 사영)
    energy = float(np.sum(S_in[: args.rank] ** 2) / np.sum(S_in ** 2))
    print(f"  상위 {args.rank}개 특이값이 담는 에너지: {energy:.4f}")

    Ztr, Zte = Xtr @ P, Xte @ P
    errs_k, errs_v = [], []
    for l in kept:
        kp, vp = d / f"k_l{l:02d}.f32", d / f"v_l{l:02d}.f32"
        K, V = load(kp, args.kvdim), load(vp, args.kvdim)
        m = min(len(X), len(K))
        Bk = ls_fit(Ztr[:m - nTest], K[:m][:-nTest])
        Bv = ls_fit(Ztr[:m - nTest], V[:m][:-nTest])
        errs_k.append(rel_err(K[:m][-nTest:], Zte @ Bk))
        errs_v.append(rel_err(V[:m][-nTest:], Zte @ Bv))
    print(f"  K rel.err {np.mean(errs_k):.4f} / V rel.err {np.mean(errs_v):.4f}"
          f"   (dense 대비 {np.mean(errs_k)/ek_all:.2f}x)")

    # ---- 비용 비교 ----
    nL = len(kept)
    h, dkv, r = args.hidden, args.kvdim, args.rank
    dense_params = nL * 2 * h * dkv
    shared_params = h * r + nL * 2 * r * dkv
    print(f"\n비용 (상위 {nL}개 레이어, K+V)")
    print(f"  dense  : {dense_params/1e6:8.2f} M params  ({dense_params*0.5625/1e6:7.2f} MB @Q4_0)")
    print(f"  shared : {shared_params/1e6:8.2f} M params  ({shared_params*0.5625/1e6:7.2f} MB @Q4_0)")
    print(f"  절감   : {dense_params/shared_params:.1f}x")

    print(f"\n[판정] rel.err < 0.10 이면 k={args.k} 분해가 유망. "
          f"레이어별 편차가 크면 k 를 그 지점으로 올릴 것.")


if __name__ == "__main__":
    main()
