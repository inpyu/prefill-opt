// DerivePP 레이어 단위 캘리브레이션 (research/16-derivepp.md §7.6)
//
// 왜 op 단위가 아니라 레이어 단위인가:
//   op 를 하나씩 타이트 루프로 재면 가중치가 캐시/프리페처에 유리한 상태가 되고,
//   실측 대비 15~25 % 낙관적인 값이 나온다(attnProj 0.82 / ffn 0.75 / attn 0.85).
//   빠진 연산이 있어서가 아니다 — RMSNorm/RoPE/residual/SwiGLU 를 다 더해도 2~4 % 다.
//   재현되지 않는 것은 **연산자들의 교대**다: 서로 캐시를 밀어내고, op 경계마다
//   동기화가 붙고, 프리페처가 스트림을 잃는다.
//
// 그래서 transformer 레이어 하나를 실제 순서대로 돌리고 att/ff 구간 시간을 잰다.
//
//   att 구간: RMSNorm -> Q/K/V proj -> RoPE -> attention core -> O proj -> residual
//   ff  구간: RMSNorm -> gate/up proj -> SwiGLU -> down proj -> residual
//
// ⚠️ 측정한 시간을 FLOPs 로 나눠 처리율로 만들지 않는다.
//    나누는 순간 교대 효과가 다시 사라진다. T_att(B,prefix), T_ff(B) 를 시간 그대로 쓴다.
//
// 가중치는 합성이고 형상만 모델 config 에서 온다. 따라서 모델 파일 없이,
// d/d_ff/d_kv 만 바꾸면 다른 모델에도 그대로 적용된다(zero-shot 유지).
//
// build:
//   g++ -std=c++11 -O3 -mcpu=native prefill_bench/calibrate_layer.cpp \
//       nn-quants.o nn-core.o llamafile-sgemm.o nn-repack.o -o calibrate_layer -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/nn-repack.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1] + v[v.size()/2]);
}

// ── 모델 형상 ────────────────────────────────────────────────
struct Cfg {
    int d, dFf, nHeads, nKvHeads, headDim;
    int qDim()  const { return nHeads * headDim; }
    int kvDim() const { return nKvHeads * headDim; }
};

// ── 합성 Q4_0 가중치 (repack 완료 상태) ──────────────────────
struct W {
    std::vector<NnBlockQ40> raw;
    int d, k, kBlocks;
    bool packed = false;
    void init(int d_, int k_) {
        d = d_; k = k_; kBlocks = k_ / QK4_0;
        raw.resize((size_t)d * kBlocks);
        for (auto &b : raw) { b.d = 0x3800; std::memset(b.qs, 0x42, sizeof(b.qs)); }
        packed = nnRepackSupported((NnUint)d, (NnUint)kBlocks)
              && nnRepackQ40InPlace((NnByte *)raw.data(), (NnUint)d, (NnUint)kBlocks);
    }
};

// 실제 커널과 같은 규약: 출력 "열"을 4의 배수 경계로 스레드에 나눈다.
static void gemm(const W &w, const NnBlockQ80 *x80, int B, float *y,
                 int nThreads, int t, std::vector<NnByte> &scratch) {
    const NnUint nCols4 = (NnUint)w.d / 4u;
    const NnUint per4 = (nCols4 + (NnUint)nThreads - 1u) / (NnUint)nThreads;
    const NnUint c0 = (NnUint)t * per4 * 4u;
    if (c0 >= (NnUint)w.d) return;
    NnUint cN = per4 * 4u;
    if (c0 + cN > (NnUint)w.d) cN = (NnUint)w.d - c0;

    const block_q4_0x4 *wCol = (const block_q4_0x4 *)
        ((const NnByte *)w.raw.data() + (size_t)(c0/4u) * w.kBlocks * sizeof(block_q4_0x4));
    const NnUint nGemm = (NnUint)B & ~3u;
    if (nGemm > 0u) {
        const size_t need = (size_t)(nGemm/4u) * w.kBlocks * sizeof(block_q8_0x4);
        if (scratch.size() < need) scratch.resize(need);
        block_q8_0x4 *xr = (block_q8_0x4 *)scratch.data();
        for (NnUint g = 0; g < nGemm/4u; g++)
            nnPackQ80To4x4(&x80[(size_t)g*4u*w.kBlocks], &xr[(size_t)g*w.kBlocks], (NnUint)w.kBlocks);
        ggml_gemm_q4_0_4x4_q8_0(w.k, &y[c0], w.d, wCol, xr, (int)nGemm, (int)cN);
    }
    for (NnUint b = nGemm; b < (NnUint)B; b++)
        ggml_gemv_q4_0_4x4_q8_0(w.k, &y[(size_t)b*w.d + c0], w.d,
            wCol, &x80[(size_t)b*w.kBlocks], 1, (int)cN);
}

// ── 저차수 연산들 (스레드는 행으로 나눈다) ───────────────────
static void rmsnormQuant(const float *x, NnBlockQ80 *out, int B, int d,
                         int nThreads, int t) {
    for (int b = t; b < B; b += nThreads) {
        const float *r = &x[(size_t)b*d];
        float ss = 0.0f;
        for (int i = 0; i < d; i++) ss += r[i]*r[i];
        const float inv = 1.0f / std::sqrt(ss/d + 1e-5f);
        // 정규화 결과를 바로 q80 으로 양자화한다(실제 경로와 동일).
        for (int i = 0; i < d; i += QK8_0) {
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                const float v = r[i+j]*inv;
                if (std::fabs(v) > amax) amax = std::fabs(v);
            }
            NnBlockQ80 &blk = out[(size_t)b*(d/QK8_0) + i/QK8_0];
            const float s = amax/127.0f, is = s ? 1.0f/s : 0.0f;
            blk.d = 0x3800;
            for (int j = 0; j < QK8_0; j++)
                blk.qs[j] = (std::int8_t)std::lround(r[i+j]*inv*is);
        }
    }
}

static void ropeInPlace(float *q, float *k, int B, int qDim, int kvDim,
                        int headDim, int pos0, int nThreads, int t) {
    for (int b = t; b < B; b += nThreads) {
        const int p = pos0 + b;
        for (int base = 0; base < qDim; base += headDim)
            for (int i = 0; i < headDim; i += 2) {
                const float f = powf(500000.0f, -(float)i/(float)headDim);
                const float c = cosf(p*f), s = sinf(p*f);
                float *v = &q[(size_t)b*qDim + base + i];
                const float a = v[0], d2 = v[1];
                v[0] = a*c - d2*s; v[1] = a*s + d2*c;
            }
        for (int base = 0; base < kvDim; base += headDim)
            for (int i = 0; i < headDim; i += 2) {
                const float f = powf(500000.0f, -(float)i/(float)headDim);
                const float c = cosf(p*f), s = sinf(p*f);
                float *v = &k[(size_t)b*kvDim + base + i];
                const float a = v[0], d2 = v[1];
                v[0] = a*c - d2*s; v[1] = a*s + d2*c;
            }
    }
}

static void attnCore(const float *Q, const float *K, const float *V, float *O,
                     int B, int prefix, const Cfg &c, int nThreads, int t) {
    const int kvMul = c.nHeads / c.nKvHeads;
    const int gPer = (c.nKvHeads + nThreads - 1)/nThreads;
    const int g0 = t*gPer, g1 = std::min(c.nKvHeads, g0+gPer);
    std::vector<float> scr(32 * kvMul * 128);
    for (int g = g0; g < g1; g++) {
        const float *hK = &K[(size_t)g*c.headDim];
        const float *hV = &V[(size_t)g*c.headDim];
        for (int b0 = 0; b0 < B; b0 += 32) {
            const int bc = std::min(32, B-b0);
            for (int t0 = 0; t0 < prefix; t0 += 128) {
                const int t1 = std::min(t0+128, prefix);
                for (int tt = t0; tt < t1; tt++) {
                    const float *pk = &hK[(size_t)tt*c.kvDim()];
                    for (int j = 0; j < kvMul; j++)
                    for (int b = 0; b < bc; b++) {
                        const float *q = &Q[((size_t)(b0+b)*c.nHeads + g*kvMul + j)*c.headDim];
                        float a = 0.0f;
#if defined(__ARM_NEON)
                        float32x4_t va = vdupq_n_f32(0.0f);
                        for (int i = 0; i < c.headDim; i += 4)
                            va = vmlaq_f32(va, vld1q_f32(&q[i]), vld1q_f32(&pk[i]));
                        const float32x2_t lo = vadd_f32(vget_low_f32(va), vget_high_f32(va));
                        a = vget_lane_f32(lo,0) + vget_lane_f32(lo,1);
#else
                        for (int i = 0; i < c.headDim; i++) a += q[i]*pk[i];
#endif
                        scr[((size_t)j*bc + b)*128 + (tt-t0)] = a;
                    }
                }
                for (int j = 0; j < kvMul; j++)
                for (int b = 0; b < bc; b++) {
                    float *tile = &scr[((size_t)j*bc + b)*128];
                    for (int tt = 0; tt < t1-t0; tt++)
                        tile[tt] = expf(tile[tt]*0.088388f);
                }
                for (int tt = t0; tt < t1; tt++) {
                    const float *pv = &hV[(size_t)tt*c.kvDim()];
                    for (int j = 0; j < kvMul; j++)
                    for (int b = 0; b < bc; b++) {
                        const float p = scr[((size_t)j*bc + b)*128 + (tt-t0)];
                        float *o = &O[((size_t)(b0+b)*c.nHeads + g*kvMul + j)*c.headDim];
#if defined(__ARM_NEON)
                        const float32x4_t vp = vdupq_n_f32(p);
                        for (int i = 0; i < c.headDim; i += 4)
                            vst1q_f32(&o[i], vmlaq_f32(vld1q_f32(&o[i]), vp, vld1q_f32(&pv[i])));
#else
                        for (int i = 0; i < c.headDim; i++) o[i] += p*pv[i];
#endif
                    }
                }
            }
        }
    }
}

static void swigluQuant(const float *g, const float *u, NnBlockQ80 *out,
                        int B, int dFf, int nThreads, int t) {
    for (int b = t; b < B; b += nThreads)
        for (int i = 0; i < dFf; i += QK8_0) {
            float tmp[QK8_0], amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                const float gv = g[(size_t)b*dFf + i + j];
                tmp[j] = (gv / (1.0f + expf(-gv))) * u[(size_t)b*dFf + i + j];
                if (std::fabs(tmp[j]) > amax) amax = std::fabs(tmp[j]);
            }
            NnBlockQ80 &blk = out[(size_t)b*(dFf/QK8_0) + i/QK8_0];
            const float s = amax/127.0f, is = s ? 1.0f/s : 0.0f;
            blk.d = 0x3800;
            for (int j = 0; j < QK8_0; j++)
                blk.qs[j] = (std::int8_t)std::lround(tmp[j]*is);
        }
}

static void residual(float *x, const float *y, int B, int d, int nThreads, int t) {
    for (int b = t; b < B; b += nThreads)
        for (int i = 0; i < d; i++) x[(size_t)b*d + i] += y[(size_t)b*d + i];
}

// ⚠️ 이 벤치는 **op 경계 배리어가 없다**.
//
// 각 스레드가 RMSNorm 부터 residual 까지 자기 op 열을 끝까지 독립 실행한다.
// 따라서 측정하는 것은 정확히 "연산자 교대는 포함하되 op 경계 동기화는 없는" 비용이다.
// production `NnExecutor` 는 op 마다 모든 스레드가 끝나야 다음 step 으로 간다:
//
//     배리어 있음:  Σ_o max_t c_{o,t}
//     배리어 없음:  max_t Σ_o c_{o,t}      <- 이 벤치
//
// 둘의 차이는 빈 배리어 비용이 아니라 **동기화가 노출하는 스레드 불균형**이고,
// 이것이 production 대비 22 % 잔차의 유력 후보다(research/16 §7.6).
// 여기에 배리어를 넣으려면 executor 와 동형인 atomic/yield 경로를 써야 하며,
// 그것이 §7.6 의 방안 A 다. 이 파일은 A 의 대조군으로 남긴다.

int main(int argc, char **argv) {
    const int nThreads = argc > 1 ? atoi(argv[1]) : 4;
    const int reps     = argc > 2 ? atoi(argv[2]) : 3;
    Cfg c{4096, 14336, 32, 8, 128};
    if (argc > 6) c = Cfg{atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
                          argc > 7 ? atoi(argv[7]) : 128};
    char host[256] = {0}; gethostname(host, sizeof(host)-1);
    printf("# host=%s threads=%d d=%d d_ff=%d nHeads=%d nKv=%d headDim=%d\n",
           host, nThreads, c.d, c.dFf, c.nHeads, c.nKvHeads, c.headDim);
    printf("# kind\tB\tprefix\tsec\n");

    // ⚠️ 레이어 사본을 nL 개 두고 순회한다.
    //
    // 레이어 1개분 가중치만 반복하면 DRAM 행 지역성이 실제보다 유리해져
    // 20 % 낙관적인 값이 나온다(실측 26.8 ms/레이어/청크 vs 벤치 21.5 ms).
    // 실제 스테이지는 자기 몫 레이어 L/N 개를 매 청크마다 순회하므로
    // **가중치 워킹셋 크기가 비용의 일부**다. 따라서 nL 을 축으로 잰다.
    const int nL = getenv("CAL_LAYERS") ? atoi(getenv("CAL_LAYERS")) : 4;
    const int onlyB = getenv("CAL_B") ? atoi(getenv("CAL_B")) : 0;
    const char *onlyKind = getenv("CAL_ONLY");
    printf("# layers=%d (가중치 워킹셋 축)\n", nL);
    std::vector<W> Wq(nL), Wk(nL), Wv(nL), Wo(nL), Wg(nL), Wu(nL), Wd(nL);
    for (int l = 0; l < nL; l++) {
        Wq[l].init(c.qDim(), c.d);   Wk[l].init(c.kvDim(), c.d); Wv[l].init(c.kvDim(), c.d);
        Wo[l].init(c.d, c.qDim());   Wg[l].init(c.dFf, c.d);     Wu[l].init(c.dFf, c.d);
        Wd[l].init(c.d, c.dFf);
    }
    if (!(Wq[0].packed && Wo[0].packed && Wg[0].packed && Wd[0].packed)) {
        fprintf(stderr, "repack 미지원 형상\n"); return 1;
    }

    static const int BATCHES[]  = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    static const int PREFIXES[] = {128, 512, 2048, 4096, 8192};
    const int Bmax = 128, Pmax = 8192;

    std::vector<float> x((size_t)Bmax*c.d), y((size_t)Bmax*c.d);
    std::vector<float> q((size_t)Bmax*c.qDim()), kk((size_t)Bmax*c.kvDim()), vv((size_t)Bmax*c.kvDim());
    std::vector<float> ao((size_t)Bmax*c.qDim());
    std::vector<float> gg((size_t)Bmax*c.dFf), uu((size_t)Bmax*c.dFf);
    std::vector<NnBlockQ80> xq((size_t)Bmax*(c.d/QK8_0)), hq((size_t)Bmax*(c.dFf/QK8_0));
    std::vector<float> K((size_t)Pmax*c.kvDim()), V((size_t)Pmax*c.kvDim());
    for (auto &v : x) v = (float)drand48()-0.5f;
    for (auto &v : K) v = (float)drand48()-0.5f;
    for (auto &v : V) v = (float)drand48()-0.5f;

    // ── ff 구간: prefix 무관 ──
    for (int B : BATCHES) {
        if (onlyB && B != onlyB) continue;
        std::vector<double> ts;
        int inner = std::max(20, (int)(2e11 / (6.0*B*c.d*c.dFf)));
        inner = ((inner + nL - 1)/nL)*nL;
        for (int rep = 0; rep < reps+1; rep++) {
            const double t0 = nowSec();
            std::vector<std::thread> th;
            for (int t = 0; t < nThreads; t++)
                th.emplace_back([&, t]() {
                    std::vector<NnByte> scr;
                    for (int it = 0; it < inner; it++) {
                        const int l = it % nL;
                        rmsnormQuant(x.data(), xq.data(), B, c.d, nThreads, t);
                        gemm(Wg[l], xq.data(), B, gg.data(), nThreads, t, scr);
                        gemm(Wu[l], xq.data(), B, uu.data(), nThreads, t, scr);
                        swigluQuant(gg.data(), uu.data(), hq.data(), B, c.dFf, nThreads, t);
                        gemm(Wd[l], hq.data(), B, y.data(), nThreads, t, scr);
                        residual(x.data(), y.data(), B, c.d, nThreads, t);
                    }
                });
            for (auto &t : th) t.join();
            if (rep > 0) ts.push_back((nowSec()-t0)/inner);
        }
        printf("Lff\t%d\t0\t%.9f\n", B, median(ts));
        fflush(stdout);
    }

    // ── att 구간: prefix 의존 ──
    if (onlyKind && strcmp(onlyKind, "ff") == 0) return 0;
    for (int prefix : PREFIXES)
    for (int B : BATCHES) {
        if (onlyB && B != onlyB) continue;
        std::vector<double> ts;
        const double f = 4.0*(double)B*prefix*c.headDim*c.nHeads
                       + 4.0*(double)B*c.d*(c.d + c.kvDim());
        int inner = std::max(20, (int)(1e11 / f));
        inner = ((inner + nL - 1)/nL)*nL;
        for (int rep = 0; rep < reps+1; rep++) {
            const double t0 = nowSec();
            std::vector<std::thread> th;
            for (int t = 0; t < nThreads; t++)
                th.emplace_back([&, t]() {
                    std::vector<NnByte> scr;
                    for (int it = 0; it < inner; it++) {
                        const int l = it % nL;
                        rmsnormQuant(x.data(), xq.data(), B, c.d, nThreads, t);
                        gemm(Wq[l], xq.data(), B, q.data(), nThreads, t, scr);
                        gemm(Wk[l], xq.data(), B, kk.data(), nThreads, t, scr);
                        gemm(Wv[l], xq.data(), B, vv.data(), nThreads, t, scr);
                        ropeInPlace(q.data(), kk.data(), B, c.qDim(), c.kvDim(),
                                    c.headDim, prefix - B, nThreads, t);
                        attnCore(q.data(), K.data(), V.data(), ao.data(),
                                 B, prefix, c, nThreads, t);
                        rmsnormQuant(ao.data(), xq.data(), B, c.qDim(), nThreads, t);
                        gemm(Wo[l], xq.data(), B, y.data(), nThreads, t, scr);
                        residual(x.data(), y.data(), B, c.d, nThreads, t);
                    }
                });
            for (auto &t : th) t.join();
            if (rep > 0) ts.push_back((nowSec()-t0)/inner);
        }
        printf("Latt\t%d\t%d\t%.9f\n", B, prefix, median(ts));
        fflush(stdout);
    }
    return 0;
}
