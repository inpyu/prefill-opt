// EXP-1 진단: prefill의 배치 이득이 실제로 나오는가?
//
// 모델 로딩 없이 llamafile_sgemm(Q40 weight × Q80 activation)만 떼어내
// batch(=처리 토큰 수) 1..N 에서 GFLOPS가 어떻게 변하는지 측정한다.
//
//   batch가 커져도 GFLOPS가 평평하면 → 배치 이득 없음(토큰별 matvec와 동일)
//   batch에 따라 GFLOPS가 오르면    → sgemm 배치 경로가 정상 동작
//
// build: prefill_bench/build_bench_sgemm.sh
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/llamafile/sgemm.hpp"

// Llama-3-8B 형상
struct Shape { const char *name; int d; int k; };
static const Shape SHAPES[] = {
    { "qkv/o   (d=4096,  k=4096)",  4096,  4096 },
    { "ffn_w1  (d=14336, k=4096)", 14336,  4096 },
    { "ffn_w2  (d=4096,  k=14336)", 4096, 14336 },
};
static const int BATCHES[] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const int nThreads = argc > 1 ? atoi(argv[1]) : (int)std::thread::hardware_concurrency();
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    printf("threads=%d reps=%d\n\n", nThreads, reps);

    for (const Shape &s : SHAPES) {
        const int kBlocks = s.k / Q40_BLOCK_SIZE;
        std::vector<NnBlockQ40> W((size_t)s.d * kBlocks);
        for (size_t i = 0; i < W.size(); i++) {
            W[i].d = 0.01f;
            for (int j = 0; j < Q40_BLOCK_SIZE / 2; j++) W[i].qs[j] = (std::uint8_t)(i + j);
        }

        printf("%s\n", s.name);
        printf("  %6s %10s %10s %10s\n", "batch", "ms", "GFLOPS", "speedup/b1");
        double base = 0.0;

        for (int b : BATCHES) {
            std::vector<NnBlockQ80> X((size_t)b * kBlocks);
            for (size_t i = 0; i < X.size(); i++) {
                X[i].d = 0.02f;
                for (int j = 0; j < Q80_BLOCK_SIZE; j++) X[i].qs[j] = (std::int8_t)(i + j);
            }
            std::vector<float> C((size_t)b * s.d, 0.0f);

            bool ok = true;
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                const double t0 = nowSec();
                std::vector<std::thread> ts;
                for (int t = 0; t < nThreads; t++) {
                    ts.emplace_back([&, t]() {
                        if (!llamafile_sgemm(
                                s.d, b, kBlocks,
                                W.data(), kBlocks,
                                X.data(), kBlocks,
                                C.data(), s.d,
                                t, nThreads, 0,
                                F_Q40, F_Q80, F_32))
                            ok = false;
                    });
                }
                for (std::thread &t : ts) t.join();
                const double dt = nowSec() - t0;
                if (dt < best) best = dt;
            }

            if (!ok) {
                printf("  %6d %10s  <-- llamafile_sgemm이 false 반환 (폴백 경로)\n", b, "FALLBACK");
                continue;
            }
            const double gflops = 2.0 * s.d * s.k * b / best / 1e9;
            if (b == 1) base = gflops;
            printf("  %6d %10.2f %10.2f %10.2fx\n", b, best * 1e3, gflops,
                   base > 0 ? gflops / base : 0.0);
        }
        printf("\n");
    }
    return 0;
}
