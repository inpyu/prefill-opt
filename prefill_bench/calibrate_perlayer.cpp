// DerivePP 레이어별 직접 캘리브레이션 (research/16 §7.6)
//
// 왜 레이어 수 회귀를 버리는가:
//   회귀는 L=1..4 를 **서로 다른 시각·온도**에서 재고 기울기를 뽑는다.
//   루프 순서상 L 이 커질수록 나중이고 뜨거우므로, 시간 드리프트가 레이어 효과로
//   오인된다. 실제로 같은 조건이 103.6 / 146.2 / 302.5 ms 로 2.9배 흔들렸고,
//   주파수는 8 % 밖에 안 떨어져 열만으로 설명되지 않았다.
//
//   또 하나: normUs+otherUs 를 att/ff 에 비례 배분하면 레이어 독립 항(embedding,
//   final norm)이 다시 섞여, 회귀로 c0 를 빼려는 시도를 스스로 방해한다.
//
// 대신: 한 번의 forward 에서 **step 별 시간**을 읽어 레이어별로 모은다.
//   모든 레이어가 같은 열·메모리 상태에서 수집되므로 그 교란이 사라지고,
//   c0(embedding/lm_head)는 layerIndex 로 자연히 분리된다.
//
// 출력: TSV
//   PLatt <B> <prefix> <layer> <sec>     레이어별 attention 구간
//   PLff  <B> <prefix> <layer> <sec>     레이어별 FFN 구간
//   PLc0  <B> <prefix> <sec>             레이어 무관 항 (embedding/final/lm_head)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
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
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1] + v[v.size()/2]);
}

static LlmHeader synthHeader(NnUint nLayers, NnUint seqLen, NnUint dim, NnUint hiddenDim,
                             NnUint nHeads, NnUint nKvHeads, NnUint headDim, NnUint vocab) {
    LlmHeader h{};
    h.version = 1; h.archType = LLAMA; h.dim = dim; h.nLayers = nLayers;
    h.nHeads = nHeads; h.headDim = headDim; h.nKvHeads = nKvHeads;
    h.nExperts = 0; h.nActiveExperts = 0;
    h.origSeqLen = seqLen; h.seqLen = seqLen;
    h.hiddenDim = hiddenDim; h.moeHiddenDim = 0; h.hiddenAct = HIDDEN_ACT_SILU;
    h.qDim = nHeads * headDim; h.kvDim = nKvHeads * headDim; h.vocabSize = vocab;
    h.ropeTheta = 500000.0f; h.ropeType = ROPE_LLAMA;
    h.ropeScalingFactor = 1.0f; h.ropeScalingLowFreqFactor = 1.0f;
    h.ropeScalingHighFreqFactory = 1.0f; h.ropeScalingOrigMaxSeqLen = seqLen;
    h.normEpsilon = 1e-5f; h.weightType = F_Q40; h.syncType = F_Q80;
    return h;
}

static void loadSyntheticWeights(NnExecutor &ex, NnNodeConfig *nc) {
    std::vector<NnByte> buf;
    for (NnUint s = 0; s < nc->nSegments; s++)
        for (NnUint o = 0; o < nc->segments[s].nOps; o++) {
            NnOpConfig *op = &nc->segments[s].ops[o];
            const NnSize nB = op->weightSize.nBytes;
            if (nB == 0) continue;
            if (buf.size() < nB) buf.resize(nB);
            if (op->weightSize.floatType == F_Q40) {
                NnBlockQ40 *b = (NnBlockQ40 *)buf.data();
                for (NnSize i = 0; i < nB / sizeof(NnBlockQ40); i++) {
                    b[i].d = 0x3800; std::memset(b[i].qs, 0x42, sizeof(b[i].qs));
                }
            } else if (op->weightSize.floatType == F_32) {
                float *f = (float *)buf.data();
                for (NnSize i = 0; i < nB / sizeof(float); i++) f[i] = 0.05f;
            } else std::memset(buf.data(), 0x11, nB);
            ex.loadWeight(op->name, op->index, 0, nB, buf.data());
        }
}

// op 이름으로 att / ff / 레이어무관 을 가른다.
// 실측 op 목록(DLLAMA_DUMP_STEPS)에 근거한다:
//   att: norm_pre, norm, matmul_q/k/v, rope_q/k, shift_k/v, multihead_att,
//        cast_y, matmul_wo, merge_add
//   ff : cast_d, matmul_w1/w3, act, mul, cast_d2/d3, matmul_w2, cast_y2/y3, merge_add2
enum Part { PART_ATT, PART_FF, PART_C0 };
static Part classify(const char *name) {
    if (std::strncmp(name, "block_", 6) != 0) return PART_C0;   // embedding, final_*, cast_x
    const char *n = name + 6;
    if (std::strstr(n, "w1") || std::strstr(n, "w2") || std::strstr(n, "w3")
        || std::strcmp(n, "act") == 0 || std::strcmp(n, "mul") == 0
        || std::strncmp(n, "cast_d", 6) == 0 || std::strncmp(n, "cast_y2", 7) == 0
        || std::strncmp(n, "cast_y3", 7) == 0 || std::strcmp(n, "merge_add2") == 0)
        return PART_FF;
    return PART_ATT;
}

int main(int argc, char **argv) {
    const NnUint nThreads = argc > 1 ? (NnUint)atoi(argv[1]) : 4;
    const int reps        = argc > 2 ? atoi(argv[2]) : 5;
    const NnUint dim      = argc > 3 ? (NnUint)atoi(argv[3]) : 4096;
    const NnUint hidden   = argc > 4 ? (NnUint)atoi(argv[4]) : 14336;
    const NnUint nHeads   = argc > 5 ? (NnUint)atoi(argv[5]) : 32;
    const NnUint nKvHeads = argc > 6 ? (NnUint)atoi(argv[6]) : 8;
    const NnUint headDim  = argc > 7 ? (NnUint)atoi(argv[7]) : 128;
    const NnUint vocab    = 128256;
    // 한 그래프에 레이어를 여러 개 두고 **레이어별로** 읽는다. 회귀 없음.
    const NnUint nLayers  = getenv("CAL_LAYERS") ? (NnUint)atoi(getenv("CAL_LAYERS")) : 4;
    const NnUint onlyB    = getenv("CAL_B") ? (NnUint)atoi(getenv("CAL_B")) : 0;

    char host[256] = {0}; gethostname(host, sizeof(host)-1);
    printf("# host=%s threads=%u layers=%u dim=%u hidden=%u reps=%d\n",
           host, nThreads, nLayers, dim, hidden, reps);
    printf("# kind\tB\tprefix\tlayer\tsec\n");

    static const NnUint BATCHES[]  = {4, 8, 16, 24, 32, 48, 64, 96, 128};
    static const NnUint PREFIXES[] = {128, 512, 2048, 4096, 8192};

    for (NnUint prefix : PREFIXES) {
        LlmHeader header = synthHeader(nLayers, prefix, dim, hidden,
                                       nHeads, nKvHeads, headDim, vocab);
        for (NnUint B : BATCHES) {
            if (onlyB && B != onlyB) continue;
            if (B > prefix) continue;
            NnParallelTopology topology(1, 1, 1);
            LlmNet net = buildLlmNet(&header, topology, B);
            NnNodeConfig *nc = &net.nodeConfigs[0];

            std::vector<std::vector<double>> att(nLayers), ff(nLayers);
            std::vector<double> c0, wall2, stepsum2;
            {
                NnNetExecution execution(nThreads, &net.netConfig);
                NnCpuDevice *dev = new NnCpuDevice(&net.netConfig, nc, &execution);
                std::vector<NnExecutorDevice> devs;
                devs.push_back(NnExecutorDevice(dev, 0, -1));
                NnFakeNodeSynchronizer sync;
                NnExecutor ex(&net.netConfig, nc, &devs, &execution, &sync, true);
                ex.setStepProfilingEnabled(true);
                loadSyntheticWeights(ex, nc);
                ex.setDecodePhase(false);
                execution.setBatchSize(B);
                float *pos = (float *)execution.pipes[net.positionPipeIndex];
                for (NnUint i = 0; i < B; i++) pos[i] = (float)(prefix - B + i);

                std::vector<const char *> nm(4096); std::vector<NnUint> ly(4096), us(4096);
                
                for (int r = 0; r < reps + 1; r++) {
                    const double w0 = nowSec();
                    ex.forward();
                    const double w = nowSec() - w0;
                    if (r == 0) continue;                    // 워밍업
                    const NnUint n = ex.getLastForwardStepTimes(nm.data(), ly.data(),
                                                                us.data(), 4096);
                    std::vector<double> a(nLayers, 0.0), f(nLayers, 0.0);
                    double z = 0.0;
                    for (NnUint i = 0; i < n; i++) {
                        const Part p = classify(nm[i]);
                        if (p == PART_C0) { z += us[i] * 1e-6; continue; }
                        const NnUint l = ly[i] < nLayers ? ly[i] : 0;
                        (p == PART_ATT ? a : f)[l] += us[i] * 1e-6;
                    }
                    for (NnUint l = 0; l < nLayers; l++) {
                        att[l].push_back(a[l]); ff[l].push_back(f[l]);
                    }
                    c0.push_back(z);
                    double ssum = z;
                    for (NnUint l = 0; l < nLayers; l++) ssum += a[l] + f[l];
                    wall2.push_back(w); stepsum2.push_back(ssum);
                }
            }
            for (NnUint l = 0; l < nLayers; l++) {
                printf("PLatt\t%u\t%u\t%u\t%.9f\n", B, prefix, l, median(att[l]));
                printf("PLff\t%u\t%u\t%u\t%.9f\n", B, prefix, l, median(ff[l]));
            }
            printf("PLc0\t%u\t%u\t-\t%.9f\n", B, prefix, median(c0));
            // ⚠️ 정합성 검사: step 시간 합 == forward 벽시계 여야 한다.
            // 어긋나면 step 계측이 다른 것을 재고 있다는 뜻이므로 값을 못 쓴다.
            {
                const double mw = median(wall2), ms = median(stepsum2);
                printf("PLchk\t%u\t%u\t-\twall=%.4fs stepsum=%.4fs ratio=%.3f\n",
                       B, prefix, mw, ms, mw > 0 ? ms / mw : 0.0);
            }
            fflush(stdout);
            releaseLlmNet(&net);
        }
    }
    return 0;
}
