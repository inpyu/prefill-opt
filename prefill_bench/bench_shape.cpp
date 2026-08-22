// SA-EDOT 1단계 — Down 이 왜 Gate/Up 보다 1.5배 느린가 (research/18 §5)
//
// 대조 설계:
//   총 weight 원소 수를 **같게** 두고 K 만 바꾼다.
//     Gate/Up  K=4096,  N=14336   (58,720,256)
//     중간     K=7168,  N=8192    (58,720,256)
//     Down     K=14336, N=4096    (58,720,256)
//   연산량이 고정되므로 "K 가 길어질 때만" 느려지는지 분리된다.
//
// activation pack 과 GEMM core 를 **분리 계측**한다 — production 은 projection·스레드
// 마다 activation 을 중복 repack 하며(nn-cpu-ops.cpp:2012) 그 비용이 matmul 시간
// 안에 숨어 있다. QCFuse graph-level 기각(§2)이 이 부분을 덮지 않는다.
//
// build: g++ -O3 -mcpu=native prefill_bench/bench_shape.cpp nn-quants.o nn-repack.o -o bench_shape -lpthread
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <thread>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/nn-repack.hpp"

struct Shape { const char *name; int d; int k; };   // d = output, k = input
// K sweep — 총 K*N 을 최대한 일정하게 유지하며 K 만 늘린다.
// 예측: 1 thread 는 K 와 무관하게 일정, 4 thread 는 activation panel 이 커질수록 하락.
// ⚠️ 실제 재사용 단위는 **16-row panel** 이다(커널이 batch 를 16씩 처리).
//    W_X(R=16, K) = (16/4) * (K/32) * 136 B
//    K=4096 -> 68 KiB,  K=7168 -> 119 KiB,  K=14336 -> 238 KiB
//    B=16 과 B=32 모두 활성 panel 은 같다 — 관측된 "둘 다 Down 확장 나쁨" 과 일치.
static const Shape SHAPES[] = {
    { "K= 4096 N=14336", 14336,  4096 },
    { "K= 6144 N= 9556",  9556,  6144 },
    { "K= 8192 N= 7168",  7168,  8192 },
    { "K=10240 N= 5732",  5732, 10240 },
    { "K=12288 N= 4776",  4776, 12288 },
    { "K=14336 N= 4096",  4096, 14336 },
};
static const int BATCHES[] = { 16, 32 };
static const int THREADS[] = { 1, 2, 3, 4 };

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    printf("# shape\tthreads\tbatch\tmode\tgemm_ms\tGOPS\n");

    for (const Shape &s : SHAPES) {
        const int kBlocks = s.k / Q40_BLOCK_SIZE;
        std::vector<NnBlockQ40> W0((size_t)s.d * kBlocks);
        for (size_t i = 0; i < W0.size(); i++) {
            W0[i].d = 0.01f;
            for (int j = 0; j < Q40_BLOCK_SIZE / 2; j++) W0[i].qs[j] = (std::uint8_t)(i + j);
        }
        if (!nnRepackQ40InPlace((NnByte *)W0.data(), (NnUint)s.d, (NnUint)kBlocks)) {
            printf("# %s: repack 미지원\n", s.name); continue;
        }
        const block_q4_0x4 *W = (const block_q4_0x4 *)W0.data();

        for (int nt : THREADS) {
            for (int b : BATCHES) {
                std::vector<NnBlockQ80> X((size_t)b * kBlocks);
                for (size_t i = 0; i < X.size(); i++) {
                    X[i].d = 0.02f;
                    for (int j = 0; j < Q80_BLOCK_SIZE; j++) X[i].qs[j] = (std::int8_t)(i + j);
                }
                std::vector<float> C((size_t)b * s.d, 0.0f);

                for (int mode = 0; mode < 2; mode++) {   // 0 = 중복(production), 1 = 공유
                    double bestGemm = 1e18;
                    std::vector<block_q8_0x4> shared;
                    if (mode == 1) {
                        shared.resize((size_t)(b/4)*kBlocks);
                        for (int g = 0; g < b/4; g++)
                            nnPackQ80To4x4(&X[(size_t)g*4*kBlocks], &shared[(size_t)g*kBlocks], kBlocks);
                    }
                    for (int r = 0; r < reps; r++) {
                        std::vector<std::vector<block_q8_0x4>> Xr(nt);
                        if (mode == 0) {
                            std::vector<std::thread> ts;
                            for (int t = 0; t < nt; t++)
                                ts.emplace_back([&, t]() {
                                    Xr[t].resize((size_t)(b/4)*kBlocks);
                                    for (int g = 0; g < b/4; g++)
                                        nnPackQ80To4x4(&X[(size_t)g*4*kBlocks], &Xr[t][(size_t)g*kBlocks], kBlocks);
                                });
                            for (auto &t : ts) t.join();
                        }
                        double g0 = nowSec();
                        {
                            std::vector<std::thread> ts;
                            for (int t = 0; t < nt; t++)
                                ts.emplace_back([&, t]() {
                                    const int nCols4 = s.d / 4, per4 = (nCols4 + nt - 1) / nt;
                                    const int c0 = t * per4 * 4;
                                    if (c0 >= s.d) return;
                                    int cN = per4 * 4; if (c0 + cN > s.d) cN = s.d - c0;
                                    const block_q8_0x4 *src = (mode == 0) ? Xr[t].data() : shared.data();
                                    ggml_gemm_q4_0_4x4_q8_0(s.k, &C[c0], s.d,
                                        &W[(size_t)(c0/4)*kBlocks], src, b, cN);
                                });
                            for (auto &t : ts) t.join();
                        }
                        const double gemmT = nowSec() - g0;
                        if (gemmT < bestGemm) bestGemm = gemmT;
                    }
                    const double ops = 2.0 * s.d * s.k * b;
                    printf("%s\t%d\t%d\t%s\t%.3f\t%.1f\n", s.name, nt, b,
                           mode == 0 ? "dup" : "shared", bestGemm*1e3, ops/bestGemm/1e9);
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}
