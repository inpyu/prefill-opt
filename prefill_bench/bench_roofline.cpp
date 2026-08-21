// WCEP 0a — Q4_0×Q8_0 GEMM 의 roofline 판정 (research/17 §7.4.3)
//
// 목적: arithmetic-only 최적화(WCEP)의 물리적 상한을 먼저 확인한다.
//   실행이 대역폭에 포화돼 있다면 정수 µop 을 줄여도 이득이 없다.
//
// ⚠️ 단일 B 로 판정하지 않는다. 실용 범위 B ∈ {16,32,64} 전부에서 memory-bound
//    여야 기각이며, B=16 만 그렇고 32 이상이 compute-bound 면 B 이동을 포함해
//    다시 판단해야 한다.
//
// ⚠️ `BW_achieved ≈ BW_STREAM` 만으로 판정하지 않는다. 접근 패턴이 다르고,
//    이 기기의 4-thread STREAM 자체가 1-thread 보다 낮다(9.59 → 6.36 GB/s).
//    그래서 **thread scaling** 을 함께 본다 — 성능이 thread 에 따라 포화되고
//    achieved BW 도 함께 포화되면 대역폭 증거가 강해진다.
//
// weight traffic 회계 (§7.4.1):
//   Q4_0 는 32 weight 마다 code 16 B + FP16 scale 2 B = 18 B
//   → bytes = d * kBlocks * 18
//   activation/output 은 별도로 계산해 함께 출력한다.
//
// build: g++ -O3 -mcpu=native prefill_bench/bench_roofline.cpp \
//          nn-quants.o llamafile-sgemm.o -o bench_roofline -lpthread
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <thread>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/nn-repack.hpp"

struct Shape { const char *name; int d; int k; };
static const Shape SHAPES[] = {
    { "Q/O    4096->4096",   4096,  4096 },
    { "K/V    4096->1024",   1024,  4096 },
    { "Gate/Up 4096->14336",14336,  4096 },
    { "Down  14336->4096",   4096, 14336 },
};
static const int BATCHES[] = { 4, 8, 16, 32, 64 };
static const int THREADS[] = { 1, 2, 4 };

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    printf("# shape\tthreads\tbatch\tms\tGOPS\tW_MiB\ttot_MiB\tBW_GBps\tintensity\n");

    for (const Shape &s : SHAPES) {
        const int kBlocks = s.k / Q40_BLOCK_SIZE;
        // 가중치는 production 과 동일하게 block_q4_0x4 로 repack 한 상태를 만든다.
        std::vector<NnBlockQ40> W0((size_t)s.d * kBlocks);
        for (size_t i = 0; i < W0.size(); i++) {
            W0[i].d = 0.01f;
            for (int j = 0; j < Q40_BLOCK_SIZE / 2; j++) W0[i].qs[j] = (std::uint8_t)(i + j);
        }
        // 가중치 repack 은 in-place 다(block_q4_0x4 == 4 * NnBlockQ40).
        if (!nnRepackQ40InPlace((NnByte *)W0.data(), (NnUint)s.d, (NnUint)kBlocks)) {
            printf("# %s: repack 미지원\n", s.name); continue;
        }
        const block_q4_0x4 *W = (const block_q4_0x4 *)W0.data();
        // weight bytes: code 16 B + scale 2 B per 32 weights
        const double wBytes = (double)s.d * kBlocks * 18.0;

        for (int nt : THREADS) {
            for (int b : BATCHES) {
                std::vector<NnBlockQ80> X((size_t)b * kBlocks);
                for (size_t i = 0; i < X.size(); i++) {
                    X[i].d = 0.02f;
                    for (int j = 0; j < Q80_BLOCK_SIZE; j++) X[i].qs[j] = (std::int8_t)(i + j);
                }
                std::vector<float> C((size_t)b * s.d, 0.0f);
                // 활성화도 production 과 동일하게 block_q8_0x4 로 repack
                std::vector<block_q8_0x4> Xr((size_t)(b / 4) * kBlocks);
                for (int g = 0; g < b / 4; g++)
                    nnPackQ80To4x4(&X[(size_t)g * 4 * kBlocks], &Xr[(size_t)g * kBlocks], kBlocks);
                // activation: Q8_0 는 32 원소마다 32 B + scale 2 B = 34 B
                const double xBytes = (double)b * kBlocks * 34.0;
                const double cBytes = (double)b * s.d * 4.0;
                const double totBytes = wBytes + xBytes + cBytes;

                bool ok = true; double best = 1e18;
                for (int r = 0; r < reps; r++) {
                    const double t0 = nowSec();
                    std::vector<std::thread> ts;
                    for (int t = 0; t < nt; t++)
                        ts.emplace_back([&, t]() {
                            // production 과 동일하게 출력 열을 4의 배수 경계로 분할
                            const int nCols4 = s.d / 4, per4 = (nCols4 + nt - 1) / nt;
                            const int c0 = t * per4 * 4;
                            if (c0 >= s.d) return;
                            int cN = per4 * 4; if (c0 + cN > s.d) cN = s.d - c0;
                            ggml_gemm_q4_0_4x4_q8_0(s.k, &C[c0], s.d,
                                &W[(size_t)(c0 / 4) * kBlocks], Xr.data(), b, cN);
                        });
                    for (std::thread &t : ts) t.join();
                    const double dt = nowSec() - t0;
                    if (dt < best) best = dt;
                }
                if (!ok) { printf("%s\t%d\t%d\tFALLBACK\n", s.name, nt, b); continue; }
                const double ops = 2.0 * s.d * s.k * b;
                printf("%s\t%d\t%d\t%.3f\t%.1f\t%.1f\t%.1f\t%.2f\t%.1f\n",
                       s.name, nt, b, best*1e3, ops/best/1e9,
                       wBytes/1048576.0, totBytes/1048576.0,
                       totBytes/best/1e9, ops/totBytes);
                fflush(stdout);
            }
        }
    }
    return 0;
}
