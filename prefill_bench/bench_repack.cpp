// EXP: Q4_0 가중치 repack 이 실제로 얼마나 빠른가?
//
// 현재 dllama 의 llamafile tinyBLAS_Q0_ARM 경로와,
// llama.cpp 가 쓰는 repack 경로(block_q4_0x4 + ggml_gemm_q4_0_4x4_q8_0)를
// 같은 형상/스레드 수에서 직접 비교한다.
//
// 이식 전에 이득을 확인하기 위한 검증용이므로, repack 커널은 손으로 옮기지 않고
// llama.cpp 가 빌드해 둔 libggml-cpu.so 의 심볼을 그대로 링크해서 부른다.
//
// build: prefill_bench/build_bench_repack.sh
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/llamafile/sgemm.hpp"

// ---- ggml 쪽 레이아웃 (ggml/src/ggml-cpu/repack.h 의 block<K,N> 과 동일해야 한다) ----
// block<4,4>: q4_0 4개 블록을 4행 인터리브. 4*2 + (32*4*4)/8 = 72 bytes
// block<8,4>: q8_0 4개 블록을 4행 인터리브. 4*2 + (32*4*8)/8 = 136 bytes





#include "../src/nn/nn-repack.hpp"



struct Shape { const char *name; int d; int k; };
static const Shape SHAPES[] = {
    { "qkv/o   (d=4096,  k=4096)",  4096,  4096 },
    { "ffn_w1  (d=14336, k=4096)", 14336,  4096 },
    { "ffn_w2  (d=4096,  k=14336)", 4096, 14336 },
};
static const int BATCHES[] = { 4, 8, 16, 32, 64, 128 };

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const int nThreads = argc > 1 ? atoi(argv[1]) : 4;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    printf("threads=%d reps=%d\n", nThreads, reps);
    printf("%-28s %6s %12s %12s %9s\n", "shape", "batch", "현재(GFLOPS)", "repack(GFLOPS)", "배율");

    for (const Shape &s : SHAPES) {
        const int kBlocks = s.k / Q40_BLOCK_SIZE;

        // 원본 Q40 가중치
        std::vector<NnBlockQ40> W((size_t)s.d * kBlocks);
        for (size_t i = 0; i < W.size(); i++) {
            W[i].d = 0x2e66;                      // fp16 ~0.1
            for (int j = 0; j < Q40_BLOCK_SIZE / 2; j++)
                W[i].qs[j] = (uint8_t)((i * 7 + j * 13) & 0xff);
        }
        // repack: 실제 프로덕션 경로와 동일하게 in-place repack 함수를 쓴다.
        // (NnBlockQ40 4개 = block_q4_0x4 로 크기가 같아 같은 버퍼에서 변환된다)
        std::vector<NnBlockQ40> Wbuf(W);
        if (!nnRepackQ40InPlace((NnByte *)Wbuf.data(), s.d, kBlocks)) {
            printf("%-28s  repack 미지원 형상\n", s.name);
            continue;
        }
        const block_q4_0x4 *Wr = (const block_q4_0x4 *)Wbuf.data();

        for (int nb : BATCHES) {
            // --- 현재 경로: NnBlockQ80 활성화 ---
            std::vector<NnBlockQ80> X((size_t)nb * kBlocks);
            for (size_t i = 0; i < X.size(); i++) {
                X[i].d = 0x2e66;
                for (int j = 0; j < Q80_BLOCK_SIZE; j++)
                    X[i].qs[j] = (int8_t)((i * 3 + j) & 0x7f);
            }
            std::vector<float> C((size_t)nb * s.d, 0.0f);

            double bestCur = 1e18;
            bool ok = true;
            for (int r = 0; r < reps; r++) {
                const double t0 = nowSec();
                std::vector<std::thread> ts;
                for (int t = 0; t < nThreads; t++)
                    ts.emplace_back([&, t]() {
                        if (!llamafile_sgemm(s.d, nb, kBlocks, W.data(), kBlocks,
                                             X.data(), kBlocks, C.data(), s.d,
                                             t, nThreads, 0, F_Q40, F_Q80, F_32))
                            ok = false;
                    });
                for (std::thread &t : ts) t.join();
                bestCur = std::min(bestCur, nowSec() - t0);
            }

            // --- repack 경로: 활성화도 block_q8_0x4 로 재배치 ---
            std::vector<float> Xf((size_t)nb * s.k);
            for (size_t i = 0; i < Xf.size(); i++)
                Xf[i] = 0.01f * (float)((i % 197) - 98);
            std::vector<block_q8_0x4> Xr((size_t)(nb / 4) * kBlocks);
            for (int rr = 0; rr < nb / 4; rr++)
                ggml_quantize_mat_q8_0_4x4(&Xf[(size_t)rr * 4 * s.k],
                                           &Xr[(size_t)rr * kBlocks], s.k);
            std::vector<float> Cr((size_t)nb * s.d, 0.0f);

            double bestRep = 1e18;
            for (int r = 0; r < reps; r++) {
                const double t0 = nowSec();
                std::vector<std::thread> ts;
                for (int t = 0; t < nThreads; t++)
                    ts.emplace_back([&, t]() {
                        // 출력 열(s.d)을 스레드로 분할
                        const int per = (s.d / 4 + nThreads - 1) / nThreads;
                        const int c0 = t * per * 4;
                        int cN = per * 4;
                        if (c0 >= s.d) return;
                        if (c0 + cN > s.d) cN = s.d - c0;
                        ggml_gemm_q4_0_4x4_q8_0(
                            s.k, &Cr[c0], s.d,
                            &Wr[(size_t)(c0 / 4) * kBlocks],
                            Xr.data(), nb, cN);
                    });
                for (std::thread &t : ts) t.join();
                bestRep = std::min(bestRep, nowSec() - t0);
            }

            const double fl = 2.0 * s.d * s.k * nb;
            const double gCur = ok ? fl / bestCur / 1e9 : 0.0;
            const double gRep = fl / bestRep / 1e9;
            printf("%-28s %6d %12.1f %12.1f %8.2fx\n",
                   nb == BATCHES[0] ? s.name : "", nb, gCur, gRep,
                   gCur > 0 ? gRep / gCur : 0.0);
        }
        printf("\n");
    }
    return 0;
}
