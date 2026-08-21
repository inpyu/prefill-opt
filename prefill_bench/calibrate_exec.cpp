// DerivePP 방안 A — production executor 로 att/ff 서브블록 비용을 잰다
// (research/16-derivepp.md §7.6)
//
// 왜 필요한가:
//   op 단위 마이크로벤치는 −25 %, 연산자를 교대시킨 레이어 macrobench 도 −22 %.
//   가중치 working-set 가설은 기각됐고(16배 늘려도 6 %),
//   순수 step 동기화도 기각됐다(1.246 us/step = 잔차의 0.3 %).
//   남은 유력 후보는 **동기화가 노출하는 thread load-imbalance** 다:
//
//       배리어 있음: Σ_o max_t c_{o,t}      <- production
//       배리어 없음: max_t Σ_o c_{o,t}      <- calibrate_layer.cpp
//
//   이 차이는 빈 배리어 비용으로 잡히지 않으므로 executor 자체를 써야 한다.
//
// 무엇을 하는가:
//   production `buildLlmNet` 으로 그래프를 만들고, synthetic weight 를 주입한 뒤
//   실제 `NnExecutor::forward()` 를 돌린다. 모델 파일이 없어도 되고
//   (N,B,p) 조합을 end-to-end 로 쓸어보지도 않으므로 zero-tuning 정의는 유지된다.
//   달라지는 것은 캘리브레이션이 production executor 와 **동형**이 된다는 점뿐이다.
//
// 출력: TSV  kind(Latt|Lff|Lall)  B  prefix  sec
//
// build:
//   g++ -std=c++11 -O3 -mcpu=native prefill_bench/calibrate_exec.cpp \
//       nn-quants.o nn-core.o nn-executor.o nn-cpu.o nn-cpu-ops.o nn-network.o \
//       nn-pipeline.o llamafile-sgemm.o nn-repack.o llm.o tokenizer.o -o calibrate_exec -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <map>
#include <unistd.h>
#include "../src/nn/nn-core.hpp"
#include "../src/nn/nn-executor.hpp"
#include "../src/nn/nn-cpu.hpp"
#include "../src/llm.hpp"

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1] + v[v.size()/2]);
}

// 모델 파일 없이 헤더를 만든다. 형상만 있으면 그래프가 나온다.
static LlmHeader synthHeader(NnUint nLayers, NnUint seqLen,
                             NnUint dim, NnUint hiddenDim,
                             NnUint nHeads, NnUint nKvHeads, NnUint headDim,
                             NnUint vocabSize) {
    LlmHeader h{};
    h.version = 1;
    h.archType = LLAMA;
    h.dim = dim;
    h.nLayers = nLayers;
    h.nHeads = nHeads;
    h.headDim = headDim;
    h.nKvHeads = nKvHeads;
    h.nExperts = 0;
    h.nActiveExperts = 0;
    h.origSeqLen = seqLen;
    h.seqLen = seqLen;
    h.hiddenDim = hiddenDim;
    h.moeHiddenDim = 0;
    h.hiddenAct = HIDDEN_ACT_SILU;
    h.qDim = nHeads * headDim;
    h.kvDim = nKvHeads * headDim;
    h.vocabSize = vocabSize;
    h.ropeTheta = 500000.0f;
    h.ropeType = ROPE_LLAMA;
    h.ropeScalingFactor = 1.0f;
    h.ropeScalingLowFreqFactor = 1.0f;
    h.ropeScalingHighFreqFactory = 1.0f;
    h.ropeScalingOrigMaxSeqLen = seqLen;
    h.normEpsilon = 1e-5f;
    h.weightType = F_Q40;
    h.syncType = F_Q80;
    return h;
}

// 모든 op 의 가중치를 유효한 합성값으로 채운다.
//
// 미초기화 메모리를 그대로 두면 denormal 이 섞여 타이밍이 왜곡될 수 있다.
// Q4_0 은 스케일(fp16)과 4비트 니블이므로 유효 패턴을 넣어 준다.
static void loadSyntheticWeights(NnExecutor &executor, NnNodeConfig *nodeConfig) {
    std::vector<NnByte> buf;
    for (NnUint s = 0; s < nodeConfig->nSegments; s++) {
        NnSegmentConfig *seg = &nodeConfig->segments[s];
        for (NnUint o = 0; o < seg->nOps; o++) {
            NnOpConfig *op = &seg->ops[o];
            const NnSize nBytes = op->weightSize.nBytes;
            if (nBytes == 0) continue;
            if (buf.size() < nBytes) buf.resize(nBytes);
            if (op->weightSize.floatType == F_Q40) {
                NnBlockQ40 *b = (NnBlockQ40 *)buf.data();
                const NnSize n = nBytes / sizeof(NnBlockQ40);
                for (NnSize i = 0; i < n; i++) {
                    b[i].d = 0x3800;                       // fp16 0.5
                    std::memset(b[i].qs, 0x42, sizeof(b[i].qs));
                }
            } else if (op->weightSize.floatType == F_32) {
                float *f = (float *)buf.data();
                for (NnSize i = 0; i < nBytes / sizeof(float); i++) f[i] = 0.05f;
            } else {
                std::memset(buf.data(), 0x11, nBytes);
            }
            executor.loadWeight(op->name, op->index, 0, nBytes, buf.data());
        }
    }
}

int main(int argc, char **argv) {
    const NnUint nThreads = argc > 1 ? (NnUint)atoi(argv[1]) : 4;
    const int reps        = argc > 2 ? atoi(argv[2]) : 3;
    // 형상은 모델 config 에서 온다(측정 아님).
    const NnUint dim      = argc > 3 ? (NnUint)atoi(argv[3]) : 4096;
    const NnUint hidden   = argc > 4 ? (NnUint)atoi(argv[4]) : 14336;
    const NnUint nHeads   = argc > 5 ? (NnUint)atoi(argv[5]) : 32;
    const NnUint nKvHeads = argc > 6 ? (NnUint)atoi(argv[6]) : 8;
    const NnUint headDim  = argc > 7 ? (NnUint)atoi(argv[7]) : 128;
    // 레이어 수는 워킹셋 축(§7.6에서 6 % 로 작음). 기본 4 = PP8 의 스테이지 몫.
    const NnUint nLayers  = getenv("CAL_LAYERS") ? (NnUint)atoi(getenv("CAL_LAYERS")) : 4;
    const NnUint vocab    = 128256;

    char host[256] = {0}; gethostname(host, sizeof(host)-1);
    printf("# host=%s threads=%u layers=%u dim=%u hidden=%u\n",
           host, nThreads, nLayers, dim, hidden);
    printf("# kind\tB\tprefix\tsec\n");

    static const NnUint BATCHES[]  = {4, 8, 16, 24, 32, 48, 64, 96, 128};
    static const NnUint PREFIXES[] = {128, 512, 2048, 4096, 8192};
    const NnUint onlyB = getenv("CAL_B") ? (NnUint)atoi(getenv("CAL_B")) : 0;
    std::map<NnUint,double> lastLatt;

    // ⚠️ 레이어 차분으로 잰다.
    //
    // 그래프에는 embedding, final norm, lm_head 같은 **레이어와 무관한 상수항**이
    // 들어 있다. T(nLayers) = c0 + nLayers * c_layer 이므로
    //     c_layer = (T(L2) - T(L1)) / (L2 - L1)
    // 로 상수항을 정확히 소거한다. 단순히 nLayers 로 나누면 c0 가 섞인다.
    // ⚠️ 2점 차분은 불안정하다.
    //
    // T(L) = c0 + L*c_layer 에서 c0 가 전체의 15~25 % 라 두 큰 값의 차이가
    // 잡음에 취약하고, prefix 가 크면 KV 가 L 에 비례해(L=4 268 MB, L=8 537 MB)
    // 메모리 압력이 달라져 affine 가정 자체가 깨진다.
    // 실제로 2점 차분은 Latt 를 비단조로, 심하면 464 ms 로 내놓았다.
    // 여러 L 로 회귀하고 적합도(R^2)를 함께 낸다.
    static const NnUint LS[] = {1, 2, 3, 4};
    const int nLS = (int)(sizeof(LS) / sizeof(LS[0]));
    for (NnUint prefix : PREFIXES) {
        for (NnUint B : BATCHES) {
            if (onlyB && B != onlyB) continue;
            if (B > prefix) continue;
            double tt[8], ta[8], tf[8];
            for (int which = 0; which < nLS; which++) {
            const NnUint nL = LS[which];
            LlmHeader header = synthHeader(nL, prefix, dim, hidden,
                                           nHeads, nKvHeads, headDim, vocab);

            NnParallelTopology topology(1, 1, 1);
            LlmNet net = buildLlmNet(&header, topology, B);
            NnNodeConfig *nodeConfig = &net.nodeConfigs[0];

            std::vector<double> ts, tsA, tsF;
            {
                NnNetExecution execution(nThreads, &net.netConfig);
                // NnExecutorDevice 는 unique_ptr 로 소유권을 가져간다(스택 금지).
                NnCpuDevice *device = new NnCpuDevice(&net.netConfig, nodeConfig, &execution);
                std::vector<NnExecutorDevice> devices;
                devices.push_back(NnExecutorDevice(device, 0, -1));
                NnFakeNodeSynchronizer sync;
                // benchmark=true 로 만들어야 op 별 시간이 누적된다.
                // production 의 --stage-timing 과 **같은 계측기**를 쓰므로
                // 캘리브레이션과 production 비교가 동형이 된다.
                NnExecutor executor(&net.netConfig, nodeConfig, &devices,
                                    &execution, &sync, true);
                // breakdown 은 step profiling 을 따로 켜야 채워진다
                // (production 의 --stage-timing 경로와 동일).
                executor.setStepProfilingEnabled(true);
                loadSyntheticWeights(executor, nodeConfig);
                // prefill 경로로 고정한다. 설정하지 않으면 lm_head 가 배치 전체 행을
                // 계산해(decode 기본) 레이어 비용이 크게 부풀려진다.
                executor.setDecodePhase(false);

                execution.setBatchSize(B);
                // 위치를 prefix 근처로 두어 KV 스캔 길이를 실제와 맞춘다.
                float *pos = (float *)execution.pipes[net.positionPipeIndex];
                for (NnUint i = 0; i < B; i++)
                    pos[i] = (float)(prefix - B + i);

                // ⚠️ inner 를 FFN FLOPs 로만 잡으면 안 된다.
                //
                // prefix 가 크면 attention 이 훨씬 무거운데 inner 가 같아서
                // 레이어 차분 (T(L2)-T(L1))/(L2-L1) 이 잡음에 묻힌다.
                // 실제로 prefix=8192 의 Latt 가 4096 보다 **작게** 나왔다
                // (52.94 vs 64.95 ms) — 물리적으로 불가능한 비단조다.
                // attention FLOPs 를 포함해 잡는다.
                const double perFwd = (double)nL;
                const double fFfn = 6.0 * B * dim * hidden;
                const double fAtt = 4.0 * B * (double)prefix * headDim * nHeads
                                  + 4.0 * B * dim * (dim + nKvHeads * headDim);
                int inner = (int)(6.0e9 / ((fFfn + fAtt) * perFwd));
                if (inner < 5) inner = 5;
                // 진단용: prefix 가 큰 구간에서 측정 구간이 짧아 회귀가 불안정하다.
                // CAL_INNER 로 강제해 재현성을 검증한다.
                if (getenv("CAL_INNER")) inner = atoi(getenv("CAL_INNER"));
                for (int r = 0; r < reps + 1; r++) {
                    double aSum = 0, fSum = 0, tot0 = nowSec();
                    for (int it = 0; it < inner; it++) {
                        executor.forward();
                        NnExecutorOpBreakdown bd{};
                        if (executor.getLastForwardOpBreakdown(&bd)) {
                            // att 서브블록 = attention 계열 전체(proj + core)
                            // ff  서브블록 = FFN
                            // norm/other 는 두 구간에 비례 배분한다.
                            const double a = (double)bd.attnUs;
                            const double f = (double)bd.ffnUs;
                            const double rest = (double)(bd.normUs + bd.otherUs);
                            const double tot = a + f;
                            aSum += (a + (tot > 0 ? rest * a / tot : 0)) * 1e-6;
                            fSum += (f + (tot > 0 ? rest * f / tot : 0)) * 1e-6;
                        }
                    }
                    const double wall = (nowSec() - tot0) / inner;
                    if (r > 0) {
                        ts.push_back(wall);
                        tsA.push_back(aSum / inner);
                        tsF.push_back(fSum / inner);
                    }
                }
            }
            tt[which] = median(ts);
            ta[which] = median(tsA);
            tf[which] = median(tsF);
            releaseLlmNet(&net);
            }
            // 최소제곱: y = c0 + L*slope,  R^2 함께 낸다.
            auto fit = [&](const double *y, double *slope, double *c0, double *r2) {
                double sx=0, sy=0, sxx=0, sxy=0;
                for (int i = 0; i < nLS; i++) { const double x = LS[i];
                    sx+=x; sy+=y[i]; sxx+=x*x; sxy+=x*y[i]; }
                const double n = nLS;
                *slope = (n*sxy - sx*sy) / (n*sxx - sx*sx);
                *c0 = (sy - *slope*sx) / n;
                const double ym = sy/n; double ss=0, rs=0;
                for (int i = 0; i < nLS; i++) { const double x = LS[i];
                    ss += (y[i]-ym)*(y[i]-ym);
                    const double e = y[i] - (*c0 + *slope*x); rs += e*e; }
                *r2 = ss > 0 ? 1.0 - rs/ss : 0.0;
            };
            // 레이어 1개분. calibrate_layer.cpp 와 직접 비교 가능하다.
            double sA, cA, rA, sF, cF, rF, sT, cT, rT;
            fit(ta, &sA, &cA, &rA); fit(tf, &sF, &cF, &rF); fit(tt, &sT, &cT, &rT);
            printf("Lall\t%u\t%u\t%.9f\t(c0=%.6f r2=%.4f)\n", B, prefix, sT, cT, rT);
            if (getenv("CAL_RAW")) {
                printf("# raw B=%u prefix=%u  att:", B, prefix);
                for (int i = 0; i < nLS; i++) printf(" L%u=%.1fms", LS[i], ta[i]*1e3);
                printf("  ff:");
                for (int i = 0; i < nLS; i++) printf(" L%u=%.1fms", LS[i], tf[i]*1e3);
                printf("  r2_att=%.4f\n", rA);
            }
            // att/ff 를 나눠서 낸다 — planner 의 att_share 가정을 제거한다(§7.6b).
            const double lattV = sA, lffV = sF;
            printf("Latt\t%u\t%u\t%.9f\n", B, prefix, lattV);
            printf("Lff\t%u\t%u\t%.9f\n", B, prefix, lffV);
            // 단조성 검사. attention 은 prefix 에 단조 증가해야 한다.
            // 비단조면 조용히 틀린 값이 planner 까지 흘러가므로 표시한다.
            if (lastLatt.count(B) && lattV < lastLatt[B])
                printf("# WARN nonmonotonic Latt B=%u prefix=%u  %.3f < %.3f ms\n",
                       B, prefix, lattV * 1e3, lastLatt[B] * 1e3);
            if (lattV <= 0.0 || lffV <= 0.0)
                printf("# WARN nonpositive B=%u prefix=%u\n", B, prefix);
            if (rA < 0.97 || rF < 0.97)
                printf("# WARN poor fit B=%u prefix=%u  r2_att=%.4f r2_ff=%.4f\n",
                       B, prefix, rA, rF);
            lastLatt[B] = lattV;
            fflush(stdout);
        }
    }
    return 0;
}
