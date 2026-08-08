// 배치 attention 커널 두 개를 같은 입력으로 돌려 수치 차이를 잰다.
//
// 왜 필요한가: 배치 perplexity 에서 융합 커널이 기존 커널과 0.27% 다른 값을 냈다.
// "부동소수점 재결합일 것"이라고 단정할 수 없어서 직접 비교한다.
//   - 상대오차가 1e-6 수준이면 반올림 차이다.
//   - 1e-2 수준이면 버그다.
//
// 두 함수 모두 static 이므로 .cpp 를 통째로 include 한다.
#include "../src/nn/nn-cpu-ops.cpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static float frand() { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

int main(int argc, char **argv) {
    const NnUint batchSize = (argc > 1) ? (NnUint)atoi(argv[1]) : 112u;
    const NnUint nHeads = 32u, nKvHeads = 8u, headDim = 128u;
    const NnUint nHeads0 = nHeads;                 // 단일 노드(tpSize=1)
    const NnUint kvDim0 = nKvHeads * headDim;      // 1024
    const NnUint qSliceD0 = nHeads0 * headDim;     // 4096
    const NnUint seqLen = 704u;
    const NnUint nThreads = 4u;

    srand(42);
    std::vector<float> query((size_t)batchSize * qSliceD0);
    std::vector<float> keyCache((size_t)seqLen * kvDim0);
    std::vector<float> valueCache((size_t)seqLen * kvDim0);
    std::vector<float> positions(batchSize);
    for (auto &v : query) v = frand();
    for (auto &v : keyCache) v = frand();
    for (auto &v : valueCache) v = frand();
    // prefill 청크: 위치가 연속이다.
    const NnUint base = seqLen - batchSize - 1u;
    for (NnUint b = 0; b < batchSize; b++) positions[b] = (float)(base + b);

    std::vector<float> outA((size_t)batchSize * qSliceD0, 0.0f);
    std::vector<float> outB((size_t)batchSize * qSliceD0, 0.0f);
    std::vector<float> att((size_t)batchSize * nHeads0 * seqLen, 0.0f);
    std::vector<NnByte *> ptrA(batchSize), ptrB(batchSize);
    for (NnUint b = 0; b < batchSize; b++) {
        ptrA[b] = (NnByte *)&outA[(size_t)b * qSliceD0];
        ptrB[b] = (NnByte *)&outB[(size_t)b * qSliceD0];
    }

    // 두 커널 모두 스레드 분할을 내부에서 하므로 nThreads 번 호출한다.
    for (NnUint t = 0; t < nThreads; t++)
        multiheadAttBatch_F32(ptrA.data(), query.data(), qSliceD0, att.data(),
            keyCache.data(), valueCache.data(), positions.data(), batchSize,
            nHeads, nHeads0, nKvHeads, kvDim0, headDim, seqLen, nThreads, t);
    for (NnUint t = 0; t < nThreads; t++)
        multiheadAttFused_F32(ptrB.data(), query.data(), qSliceD0,
            keyCache.data(), valueCache.data(), positions.data(), batchSize,
            nHeads, nHeads0, nKvHeads, kvDim0, headDim, seqLen, nThreads, t);

    // float64 기준해. 어느 커널이 "맞는지"는 서로 비교해서는 알 수 없다.
    // 근사 지수(expf_neon)와 온라인 소프트맥스를 섞으면 exp~(a)*exp~(b) != exp~(a+b)
    // 이므로 두 커널은 원리적으로 다른 값을 낸다. 진짜 값과 대야 판정이 된다.
    std::vector<double> ref((size_t)batchSize * qSliceD0, 0.0);
    {
        const NnUint kvMul = nHeads / nKvHeads;
        const double root = std::sqrt((double)headDim);
        for (NnUint b = 0; b < batchSize; b++) {
            const NnUint qPos = (NnUint)positions[b];
            for (NnUint h = 0; h < nHeads0; h++) {
                const NnUint kvHead = h / kvMul;
                std::vector<double> sc(qPos + 1);
                double mx = -1e300;
                for (NnUint t = 0; t <= qPos; t++) {
                    double d = 0.0;
                    for (NnUint i = 0; i < headDim; i++)
                        d += (double)query[(size_t)b * qSliceD0 + h * headDim + i] *
                             (double)keyCache[(size_t)t * kvDim0 + kvHead * headDim + i];
                    sc[t] = d / root;
                    if (sc[t] > mx) mx = sc[t];
                }
                double sum = 0.0;
                for (NnUint t = 0; t <= qPos; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
                for (NnUint t = 0; t <= qPos; t++) {
                    const double w = sc[t] / sum;
                    for (NnUint i = 0; i < headDim; i++)
                        ref[(size_t)b * qSliceD0 + h * headDim + i] +=
                            w * (double)valueCache[(size_t)t * kvDim0 + kvHead * headDim + i];
                }
            }
        }
    }
    auto l2vs = [&](const std::vector<float> &o) {
        double sd = 0.0, sr = 0.0;
        for (size_t i = 0; i < o.size(); i++) {
            const double d = (double)o[i] - ref[i];
            sd += d * d; sr += ref[i] * ref[i];
        }
        return std::sqrt(sd / std::max(1e-30, sr));
    };

    double maxAbs = 0.0, maxRel = 0.0, sumSqA = 0.0, sumSqD = 0.0;
    size_t argMax = 0;
    for (size_t i = 0; i < outA.size(); i++) {
        const double a = outA[i], b = outB[i];
        const double d = std::fabs(a - b);
        const double r = d / std::max(1e-12, std::fabs(a));
        if (d > maxAbs) { maxAbs = d; argMax = i; }
        if (std::fabs(a) > 1e-4 && r > maxRel) maxRel = r;
        sumSqA += a * a; sumSqD += d * d;
    }
    printf("batchSize=%u  요소 %zu개\n", batchSize, outA.size());
    printf("  max |차이|      = %.3e   (기존=%.6f 융합=%.6f, idx=%zu)\n",
        maxAbs, outA[argMax], outB[argMax], argMax);
    printf("  max 상대오차    = %.3e   (|기존|>1e-4 인 원소만)\n", maxRel);
    printf("  두 커널 간 상대 L2 = %.3e\n", std::sqrt(sumSqD / std::max(1e-30, sumSqA)));
    printf("  --- float64 기준해 대비 ---\n");
    printf("  기존 커널 상대 L2 오차 = %.3e\n", l2vs(outA));
    printf("  융합 커널 상대 L2 오차 = %.3e\n", l2vs(outB));
    return 0;
}
