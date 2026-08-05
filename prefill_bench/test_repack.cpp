// repack 커널 이식본의 수치 정확성 검증.
//
// 같은 F32 활성화 / 같은 Q40 가중치에 대해
//   (a) 기존 경로: quantizeF32toQ80 -> llamafile_sgemm
//   (b) repack 경로: ggml_quantize_mat_q8_0_4x4 -> ggml_gemm_q4_0_4x4_q8_0
//   (c) repack gemv 경로 (배치 1행)
// 의 출력을 비교한다.
//
// 두 경로는 활성화 양자화 구현이 달라(반올림 방식) 비트 단위 일치는 기대하지 않는다.
// Q80 양자화 오차 수준(상대오차 ~1e-2) 안에 들어오는지를 본다.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../src/nn/nn-quants.hpp"
#include "../src/nn/nn-repack.hpp"
#include "../src/nn/llamafile/sgemm.hpp"

static void stats(const char *label, const std::vector<float> &a, const std::vector<float> &b) {
    double maxAbs = 0.0, sumSq = 0.0, refSq = 0.0, maxRel = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        const double r = std::fabs((double)a[i]) + 1e-6;
        maxAbs = std::max(maxAbs, d);
        maxRel = std::max(maxRel, d / r);
        sumSq += d * d;
        refSq += (double)a[i] * (double)a[i];
    }
    const double rel = std::sqrt(sumSq / (refSq + 1e-12));
    printf("  %-22s maxAbs=%.5f  maxRel=%.4f  L2rel=%.6f  %s\n",
           label, maxAbs, maxRel, rel, rel < 0.05 ? "OK" : "*** 확인 필요 ***");
}

int main(int argc, char **argv) {
    const int d = argc > 1 ? atoi(argv[1]) : 512;    // 출력 차원
    const int k = argc > 2 ? atoi(argv[2]) : 1024;   // 입력 차원
    const int nb = argc > 3 ? atoi(argv[3]) : 8;     // 배치
    const int kBlocks = k / Q40_BLOCK_SIZE;
    printf("d=%d k=%d batch=%d\n", d, k, nb);
    initQuants();

    // 가중치: 원본 Q40
    std::vector<NnBlockQ40> W((size_t)d * kBlocks);
    for (size_t i = 0; i < W.size(); i++) {
        W[i].d = CONVERT_F32_TO_F16(0.05f + 0.001f * (float)(i % 17));
        for (int j = 0; j < Q40_BLOCK_SIZE / 2; j++)
            W[i].qs[j] = (std::uint8_t)((i * 31 + j * 7) & 0xff);
    }

    // 활성화: F32
    std::vector<float> Xf((size_t)nb * k);
    for (size_t i = 0; i < Xf.size(); i++)
        Xf[i] = 0.02f * (float)((int)(i % 251) - 125);

    // (a) 기존 경로
    std::vector<NnBlockQ80> X80((size_t)nb * kBlocks);
    for (int r = 0; r < nb; r++)
        quantizeF32toQ80(&Xf[(size_t)r * k], &X80[(size_t)r * kBlocks], k, 1, 0);
    std::vector<float> Cbase((size_t)nb * d, 0.0f);
    if (!llamafile_sgemm(d, nb, kBlocks, W.data(), kBlocks, X80.data(), kBlocks,
                         Cbase.data(), d, 0, 1, 0, F_Q40, F_Q80, F_32)) {
        printf("기존 sgemm 경로 사용 불가\n");
        return 1;
    }

    // repack 가중치
    std::vector<NnBlockQ40> Wbuf(W);
    if (!nnRepackQ40InPlace((NnByte *)Wbuf.data(), d, kBlocks)) {
        printf("repack 미지원 형상\n");
        return 1;
    }
    const void *Wr = Wbuf.data();

    // (b) repack gemm (배치 4의 배수 부분)
    if (nb % 4 == 0) {
        std::vector<block_q8_0x4> Xr((size_t)(nb / 4) * kBlocks);
        for (int r = 0; r < nb / 4; r++)
            ggml_quantize_mat_q8_0_4x4(&Xf[(size_t)r * 4 * k], &Xr[(size_t)r * kBlocks], k);
        std::vector<float> Crep((size_t)nb * d, 0.0f);
        ggml_gemm_q4_0_4x4_q8_0(k, Crep.data(), d, Wr, Xr.data(), nb, d);
        stats("gemm (batch=%d)", Cbase, Crep);
    }

    // (c) repack gemv (배치 1행씩) — 활성화는 평범한 Q80 을 그대로 쓴다
    {
        std::vector<float> Cgemv((size_t)nb * d, 0.0f);
        for (int r = 0; r < nb; r++)
            ggml_gemv_q4_0_4x4_q8_0(k, &Cgemv[(size_t)r * d], d, Wr,
                                    &X80[(size_t)r * kBlocks], 1, d);
        stats("gemv (row-by-row)", Cbase, Cgemv);
    }

    return 0;
}
