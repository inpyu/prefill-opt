// DerivePP 플랫폼 캘리브레이션 (research/16-derivepp.md §2.3)
//
// 노드마다 **두 커널 계열의 비용 함수**를 측정한다. end-to-end 스윕은 하지 않는다.
//
//   T_q,k(F, B)          Q4×Q8 GEMM      (projection / FFN)
//   T_a,k(F, B, prefix)  F32 attention core (QK^T + softmax + AV)
//
// 여기서 파생되는 값:
//   P_q,k(B), P_a,k(B,prefix)   처리율
//   tau_q,k, tau_a,k            호출 고정비 (B->0 절편)
//   B_min,k                     P_q 가 포화의 일정 비율에 도달하는 knee
//   rho_k = P_q,k / P_a,k       커널 계열 상대 처리율
//
// 출력은 파싱 가능한 TSV 한 줄씩. 노드별로 돌려 모은 뒤 회귀한다.
//
// build: g++ -std=c++11 -O3 -mcpu=native -I src prefill_bench/calibrate.cpp \
//            nn-quants.o nn-core.o llamafile-sgemm.o nn-repack.o -o calibrate -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <unistd.h>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/llamafile/sgemm.hpp"
#include "../src/nn/nn-repack.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 중앙값. 평균은 콜드 스타트와 스케줄러 간섭의 긴 꼬리에 오염된다(research/13 §3).
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// ─────────────────────────────────────────────────────────────
// T_q: Q4×Q8 GEMM  (nn-cpu-ops.cpp 의 repack 경로와 같은 호출 규약)
// ─────────────────────────────────────────────────────────────
struct QShape { const char *name; int d; int k; };

static void benchQ(int nThreads, int reps) {
    // Llama-3 8B 의 실제 형상. projection 과 FFN 을 모두 덮는다.
    static const QShape SHAPES[] = {
        { "proj",  4096,  4096 },   // Q/O
        { "ffn13", 14336,  4096 },  // gate/up
        { "ffn2",   4096, 14336 },  // down
    };
    static const int BATCHES[] = { 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

    for (const QShape &s : SHAPES) {
        const NnUint kBlocks = (NnUint)(s.k / QK4_0);
        if (!nnRepackSupported((NnUint)s.d, kBlocks)) {
            printf("# skip %s (repack unsupported)\n", s.name);
            continue;
        }
        // 가중치: [d행 x kBlocks열] 을 in-place repack
        std::vector<NnBlockQ40> w((size_t)s.d * kBlocks);
        for (auto &b : w) { b.d = 0x3800; std::memset(b.qs, 0x42, sizeof(b.qs)); }
        if (!nnRepackQ40InPlace((NnByte *)w.data(), (NnUint)s.d, kBlocks)) {
            printf("# skip %s (repack failed)\n", s.name);
            continue;
        }

        for (int B : BATCHES) {
            std::vector<NnBlockQ80> x80((size_t)B * kBlocks);
            for (auto &b : x80) { b.d = 0x3800; std::memset(b.qs, 0x11, sizeof(b.qs)); }
            std::vector<float> y((size_t)B * s.d);

            // ⚠️ 스레드는 **한 번만** 만든다.
            //
            // 매 rep 마다 std::thread 를 만들면 생성 비용(코어당 50~100 us)이 전부
            // 고정비 tau 로 잡힌다. 실제 커널은 상시 스레드 풀 위에서 돌고,
            // B0 = sqrt(S*tau/((N-1)*kappa)) 가 tau 에 직접 의존하므로
            // 이 오염은 후보 B 를 통째로 틀리게 만든다.
            // 내부 반복 횟수는 작은 B 에서 타이밍 해상도를 확보하도록 키운다.
            // 타이밍 구간이 최소 ~2e10 FLOP 이 되도록 내부 반복을 잡는다.
            // 9 ms 짜리 구간은 스케줄러 지터에 45 % 까지 흔들렸다.
            const double flops1 = 2.0 * (double)B * s.d * s.k;
            const int inner = std::max(20, (int)(2e10 / flops1));
            std::vector<double> ts;
            for (int rep = 0; rep < reps + 1; rep++) {     // rep0 은 버린다
                const double t0 = nowSec();
                std::vector<std::thread> th;
                for (int t = 0; t < nThreads; t++) {
                    th.emplace_back([&, t]() {
                        // 실제 커널과 동일하게 출력 "열"을 4의 배수 경계로 나눈다.
                        const NnUint nCols4 = (NnUint)s.d / 4u;
                        const NnUint per4 = (nCols4 + (NnUint)nThreads - 1u) / (NnUint)nThreads;
                        const NnUint c0 = (NnUint)t * per4 * 4u;
                        if (c0 >= (NnUint)s.d) return;
                        NnUint cN = per4 * 4u;
                        if (c0 + cN > (NnUint)s.d) cN = (NnUint)s.d - c0;

                        const block_q4_0x4 *wCol = (const block_q4_0x4 *)
                            ((const NnByte *)w.data()
                             + (size_t)(c0 / 4u) * kBlocks * sizeof(block_q4_0x4));

                        const NnUint nGemm = (NnUint)B & ~3u;
                        std::vector<NnByte> scratch;
                        if (nGemm > 0u)
                            scratch.resize((size_t)(nGemm / 4u) * kBlocks * sizeof(block_q8_0x4));
                        block_q8_0x4 *xr = (block_q8_0x4 *)scratch.data();
                        for (int it = 0; it < inner + 1; it++) {   // it==0 은 워밍업
                            if (it == 1) { /* 워밍업 경계 */ }
                            if (nGemm > 0u) {
                                for (NnUint g = 0; g < nGemm / 4u; g++)
                                    nnPackQ80To4x4(&x80[(size_t)g * 4u * kBlocks],
                                                   &xr[(size_t)g * kBlocks], kBlocks);
                                ggml_gemm_q4_0_4x4_q8_0(s.k, &y[c0], s.d, wCol, xr,
                                                        (int)nGemm, (int)cN);
                            }
                            for (NnUint b = nGemm; b < (NnUint)B; b++)
                                ggml_gemv_q4_0_4x4_q8_0(s.k, &y[(size_t)b * s.d + c0], s.d,
                                    wCol, &x80[(size_t)b * kBlocks], 1, (int)cN);
                        }
                    });
                }
                for (auto &t : th) t.join();
                if (rep > 0) ts.push_back((nowSec() - t0) / (inner + 1));
            }
            const double sec = median(ts);
            const double flops = flops1;
            printf("Tq\t%s\t%d\t%d\t%d\t%.9f\t%.3f\n",
                s.name, s.d, s.k, B, sec, flops / sec / 1e9);
            fflush(stdout);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// T_a: F32 attention core (QK^T + online softmax + AV)
// ─────────────────────────────────────────────────────────────
//
// dllama 커널의 구조를 그대로 흉내낸다: KV 는 kvDim 스트라이드로 놓고,
// KV 타일 128 / 쿼리 타일 32 로 돈다. 절대 성능이 아니라 **B 와 prefix 에 대한
// 의존성**을 재는 것이 목적이다.
static void benchA(int nThreads, int reps) {
    const int headDim = 128, nHeads = 32, nKvHeads = 8;
    const int kvDim = nKvHeads * headDim;      // 1024
    const int kvMul = nHeads / nKvHeads;       // 4
    static const int BATCHES[]  = { 4, 8, 16, 32, 48, 64, 96, 128 };
    static const int PREFIXES[] = { 128, 512, 2048, 4096, 8192 };

    for (int prefix : PREFIXES) {
        std::vector<float> K((size_t)prefix * kvDim), V((size_t)prefix * kvDim);
        for (auto &v : K) v = (float)drand48() - 0.5f;
        for (auto &v : V) v = (float)drand48() - 0.5f;

        for (int B : BATCHES) {
            std::vector<float> Q((size_t)B * nHeads * headDim);
            std::vector<float> O((size_t)B * nHeads * headDim);
            for (auto &v : Q) v = (float)drand48() - 0.5f;

            const double flops1 = 4.0 * (double)B * prefix * headDim * nHeads;
            const int inner = std::max(3, (int)(1e10 / flops1));
            std::vector<double> ts;
            for (int rep = 0; rep < reps + 1; rep++) {
                const double t0 = nowSec();
                std::vector<std::thread> th;
                for (int t = 0; t < nThreads; t++) {
                    th.emplace_back([&, t]() {
                      for (int it = 0; it < inner + 1; it++) {
                        const int gPer = (nKvHeads + nThreads - 1) / nThreads;
                        const int g0 = t * gPer;
                        const int g1 = std::min(nKvHeads, g0 + gPer);
                        std::vector<float> scr(32 * kvMul * 128);
                        for (int g = g0; g < g1; g++) {
                            const float *hK = &K[(size_t)g * headDim];
                            const float *hV = &V[(size_t)g * headDim];
                            for (int b0 = 0; b0 < B; b0 += 32) {
                                const int bc = std::min(32, B - b0);
                                for (int t0k = 0; t0k < prefix; t0k += 128) {
                                    const int t1k = std::min(t0k + 128, prefix);
                                    // (1) QK^T
                                    for (int tt = t0k; tt < t1k; tt++) {
                                        const float *pk = &hK[(size_t)tt * kvDim];
                                        for (int j = 0; j < kvMul; j++)
                                        for (int b = 0; b < bc; b++) {
                                            const float *q = &Q[((size_t)(b0+b) * nHeads
                                                + g * kvMul + j) * headDim];
                                            float a = 0.0f;
#if defined(__ARM_NEON)
                                            float32x4_t va = vdupq_n_f32(0.0f);
                                            for (int i = 0; i < headDim; i += 4)
                                                va = vmlaq_f32(va, vld1q_f32(&q[i]), vld1q_f32(&pk[i]));
                                            const float32x2_t lo =
                                                vadd_f32(vget_low_f32(va), vget_high_f32(va));
                                            a = vget_lane_f32(lo,0) + vget_lane_f32(lo,1);
#else
                                            for (int i = 0; i < headDim; i++) a += q[i]*pk[i];
#endif
                                            scr[((size_t)j*bc + b)*128 + (tt - t0k)] = a;
                                        }
                                    }
                                    // (2) softmax (여기서는 exp 만; 상대 비용 재현이 목적)
                                    for (int j = 0; j < kvMul; j++)
                                    for (int b = 0; b < bc; b++) {
                                        float *tile = &scr[((size_t)j*bc + b)*128];
                                        for (int tt = 0; tt < t1k - t0k; tt++)
                                            tile[tt] = expf(tile[tt] * 0.088388f);
                                    }
                                    // (3) AV
                                    for (int tt = t0k; tt < t1k; tt++) {
                                        const float *pv = &hV[(size_t)tt * kvDim];
                                        for (int j = 0; j < kvMul; j++)
                                        for (int b = 0; b < bc; b++) {
                                            const float p =
                                                scr[((size_t)j*bc + b)*128 + (tt - t0k)];
                                            float *o = &O[((size_t)(b0+b) * nHeads
                                                + g * kvMul + j) * headDim];
#if defined(__ARM_NEON)
                                            const float32x4_t vp = vdupq_n_f32(p);
                                            for (int i = 0; i < headDim; i += 4)
                                                vst1q_f32(&o[i], vmlaq_f32(vld1q_f32(&o[i]),
                                                    vp, vld1q_f32(&pv[i])));
#else
                                            for (int i = 0; i < headDim; i++) o[i] += p*pv[i];
#endif
                                        }
                                    }
                                }
                            }
                        }
                      }
                    });
                }
                for (auto &t : th) t.join();
                if (rep > 0) ts.push_back((nowSec() - t0) / (inner + 1));
            }
            const double sec = median(ts);
            const double flops = flops1;
            printf("Ta\t%d\t%d\t%.9f\t%.3f\n", B, prefix, sec, flops / sec / 1e9);
            fflush(stdout);
        }
    }
}

int main(int argc, char **argv) {
    const int nThreads = argc > 1 ? atoi(argv[1]) : 4;
    const int reps     = argc > 2 ? atoi(argv[2]) : 3;
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    printf("# host=%s threads=%d reps=%d\n", host, nThreads, reps);
    printf("# kind\t...\tsec\tGFLOPS\n");
    benchQ(nThreads, reps);
    benchA(nThreads, reps);
    return 0;
}
