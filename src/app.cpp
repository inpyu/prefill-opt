#include "app.hpp"
#include <cassert>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include "nn/nn-quants.hpp"

// SpGroupScheduler implementation

SpGroupScheduler::SpGroupScheduler(NnUint nSpGroups)
    : nSpGroups(nSpGroups), groupRoles(nSpGroups, SpGroupRole::IDLE),
      groupRequestIds(nSpGroups, UINT32_MAX), nextRequestId(0) {}

NnUint SpGroupScheduler::assignPrefill() {
    // Prefer IDLE groups first
    for (NnUint i = 0; i < nSpGroups; i++) {
        if (groupRoles[i] == SpGroupRole::IDLE) {
            groupRoles[i] = SpGroupRole::PREFILL;
            groupRequestIds[i] = nextRequestId++;
            return i;
        }
    }
    return UINT32_MAX; // no group available
}

void SpGroupScheduler::transitionToDecode(NnUint spGroupId) {
    assert(spGroupId < nSpGroups);
    assert(groupRoles[spGroupId] == SpGroupRole::PREFILL);
    groupRoles[spGroupId] = SpGroupRole::DECODE;
}

void SpGroupScheduler::releaseGroup(NnUint spGroupId) {
    assert(spGroupId < nSpGroups);
    groupRoles[spGroupId] = SpGroupRole::IDLE;
    groupRequestIds[spGroupId] = UINT32_MAX;
}

NnUint SpGroupScheduler::findDecodeGroup() const {
    for (NnUint i = 0; i < nSpGroups; i++) {
        if (groupRoles[i] == SpGroupRole::DECODE)
            return i;
    }
    return UINT32_MAX;
}

NnUint SpGroupScheduler::findPrefillGroup() const {
    for (NnUint i = 0; i < nSpGroups; i++) {
        if (groupRoles[i] == SpGroupRole::PREFILL)
            return i;
    }
    return UINT32_MAX;
}

bool SpGroupScheduler::hasIdle() const {
    for (NnUint i = 0; i < nSpGroups; i++)
        if (groupRoles[i] == SpGroupRole::IDLE) return true;
    return false;
}

bool SpGroupScheduler::hasDecode() const {
    return findDecodeGroup() != UINT32_MAX;
}

bool SpGroupScheduler::hasPrefill() const {
    return findPrefillGroup() != UINT32_MAX;
}
#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif

static bool shouldLogControlPackets() {
    const char *env = std::getenv("DLLAMA_LOG_CONTROL_PACKET");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

static bool ensureParentDirectoryForFile(const char *filePath) {
    if (filePath == nullptr || filePath[0] == '\0')
        return false;

    std::string path(filePath);
    const std::string::size_type lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos)
        return true; // current directory

    const std::string dirPath = path.substr(0, lastSlash);
    if (dirPath.empty())
        return true;

    std::string current;
    // Preserve absolute path root.
    if (dirPath[0] == '/')
        current = "/";

    size_t i = (dirPath[0] == '/') ? 1 : 0;
    while (i <= dirPath.size()) {
        size_t j = dirPath.find('/', i);
        const bool end = (j == std::string::npos);
        const std::string part = end ? dirPath.substr(i) : dirPath.substr(i, j - i);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/')
                current.push_back('/');
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (end)
            break;
        i = j + 1;
    }

    return true;
}

static inline unsigned long long nowUs() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static float percentileMs(const std::vector<NnUint> &samples, float q) {
    if (samples.empty())
        return 0.0f;
    std::vector<NnUint> copy = samples;
    if (q < 0.0f) q = 0.0f;
    if (q > 1.0f) q = 1.0f;
    const size_t k = (size_t)((copy.size() - 1) * q);
    std::nth_element(copy.begin(), copy.begin() + k, copy.end());
    return (float)copy[k] / 1000.0f;
}

static float percentileValue(const std::vector<float> &samples, float q) {
    if (samples.empty())
        return 0.0f;
    std::vector<float> copy = samples;
    if (q < 0.0f) q = 0.0f;
    if (q > 1.0f) q = 1.0f;
    const size_t k = (size_t)((copy.size() - 1) * q);
    std::nth_element(copy.begin(), copy.begin() + k, copy.end());
    return copy[k];
}

static bool computeDeltaNormEncoded(
    const NnByte *curr,
    const NnByte *prev,
    NnSize rowBytes,
    NnFloatType dtype,
    float *deltaNormOut
) {
    if (curr == nullptr || prev == nullptr || deltaNormOut == nullptr)
        return false;
    const double eps = 1e-12;
    double diffSq = 0.0;
    double prevSq = 0.0;

    if (dtype == F_32) {
        if (rowBytes % sizeof(float) != 0)
            return false;
        const NnSize n = rowBytes / sizeof(float);
        const float *c = (const float *)curr;
        const float *p = (const float *)prev;
        for (NnSize i = 0; i < n; i++) {
            const double dv = (double)c[i] - (double)p[i];
            diffSq += dv * dv;
            prevSq += (double)p[i] * (double)p[i];
        }
    } else if (dtype == F_16) {
        if (rowBytes % sizeof(std::uint16_t) != 0)
            return false;
        const NnSize n = rowBytes / sizeof(std::uint16_t);
        const std::uint16_t *c = (const std::uint16_t *)curr;
        const std::uint16_t *p = (const std::uint16_t *)prev;
        for (NnSize i = 0; i < n; i++) {
            const double cf = (double)CONVERT_F16_TO_F32(c[i]);
            const double pf = (double)CONVERT_F16_TO_F32(p[i]);
            const double dv = cf - pf;
            diffSq += dv * dv;
            prevSq += pf * pf;
        }
    } else if (dtype == F_Q80) {
        if (rowBytes % sizeof(NnBlockQ80) != 0)
            return false;
        const NnSize nBlocks = rowBytes / sizeof(NnBlockQ80);
        const NnBlockQ80 *c = (const NnBlockQ80 *)curr;
        const NnBlockQ80 *p = (const NnBlockQ80 *)prev;
        for (NnSize i = 0; i < nBlocks; i++) {
            const double cd = (double)CONVERT_F16_TO_F32(c[i].d);
            const double pd = (double)CONVERT_F16_TO_F32(p[i].d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE; k++) {
                const double cv = cd * (double)c[i].qs[k];
                const double pv = pd * (double)p[i].qs[k];
                const double dv = cv - pv;
                diffSq += dv * dv;
                prevSq += pv * pv;
            }
        }
    } else if (dtype == F_Q40) {
        if (rowBytes % sizeof(NnBlockQ40) != 0)
            return false;
        const NnSize nBlocks = rowBytes / sizeof(NnBlockQ40);
        const NnBlockQ40 *c = (const NnBlockQ40 *)curr;
        const NnBlockQ40 *p = (const NnBlockQ40 *)prev;
        for (NnSize i = 0; i < nBlocks; i++) {
            const double cd = (double)CONVERT_F16_TO_F32(c[i].d);
            const double pd = (double)CONVERT_F16_TO_F32(p[i].d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int c0 = (c[i].qs[k] & 0x0F) - 8;
                const int c1 = (c[i].qs[k] >> 4) - 8;
                const int p0 = (p[i].qs[k] & 0x0F) - 8;
                const int p1 = (p[i].qs[k] >> 4) - 8;
                const double cv0 = cd * (double)c0;
                const double cv1 = cd * (double)c1;
                const double pv0 = pd * (double)p0;
                const double pv1 = pd * (double)p1;
                const double d0 = cv0 - pv0;
                const double d1 = cv1 - pv1;
                diffSq += d0 * d0 + d1 * d1;
                prevSq += pv0 * pv0 + pv1 * pv1;
            }
        }
    } else {
        return false;
    }

    *deltaNormOut = (float)(std::sqrt(diffSq) / (std::sqrt(prevSq) + eps));
    return true;
}

static bool computeActivationPairMetricsEncoded(
    const NnByte *curr,
    const NnByte *prev,
    NnSize rowBytes,
    NnFloatType dtype,
    float *deltaNormOut,
    float *cosineOut,
    float *normRatioOut,
    float *currNormOut,
    float *prevNormOut
) {
    if (curr == nullptr || prev == nullptr || deltaNormOut == nullptr ||
        cosineOut == nullptr || normRatioOut == nullptr ||
        currNormOut == nullptr || prevNormOut == nullptr) {
        return false;
    }
    const double eps = 1e-12;
    double diffSq = 0.0;
    double currSq = 0.0;
    double prevSq = 0.0;
    double dot = 0.0;

    auto addPair = [&](double cv, double pv) {
        const double dv = cv - pv;
        diffSq += dv * dv;
        currSq += cv * cv;
        prevSq += pv * pv;
        dot += cv * pv;
    };

    if (dtype == F_32) {
        if (rowBytes % sizeof(float) != 0)
            return false;
        const NnSize n = rowBytes / sizeof(float);
        const float *c = (const float *)curr;
        const float *p = (const float *)prev;
        for (NnSize i = 0; i < n; i++)
            addPair((double)c[i], (double)p[i]);
    } else if (dtype == F_16) {
        if (rowBytes % sizeof(std::uint16_t) != 0)
            return false;
        const NnSize n = rowBytes / sizeof(std::uint16_t);
        const std::uint16_t *c = (const std::uint16_t *)curr;
        const std::uint16_t *p = (const std::uint16_t *)prev;
        for (NnSize i = 0; i < n; i++)
            addPair((double)CONVERT_F16_TO_F32(c[i]), (double)CONVERT_F16_TO_F32(p[i]));
    } else if (dtype == F_Q80) {
        if (rowBytes % sizeof(NnBlockQ80) != 0)
            return false;
        const NnSize nBlocks = rowBytes / sizeof(NnBlockQ80);
        const NnBlockQ80 *c = (const NnBlockQ80 *)curr;
        const NnBlockQ80 *p = (const NnBlockQ80 *)prev;
        for (NnSize i = 0; i < nBlocks; i++) {
            const double cd = (double)CONVERT_F16_TO_F32(c[i].d);
            const double pd = (double)CONVERT_F16_TO_F32(p[i].d);
            for (NnUint k = 0; k < Q80_BLOCK_SIZE; k++)
                addPair(cd * (double)c[i].qs[k], pd * (double)p[i].qs[k]);
        }
    } else if (dtype == F_Q40) {
        if (rowBytes % sizeof(NnBlockQ40) != 0)
            return false;
        const NnSize nBlocks = rowBytes / sizeof(NnBlockQ40);
        const NnBlockQ40 *c = (const NnBlockQ40 *)curr;
        const NnBlockQ40 *p = (const NnBlockQ40 *)prev;
        for (NnSize i = 0; i < nBlocks; i++) {
            const double cd = (double)CONVERT_F16_TO_F32(c[i].d);
            const double pd = (double)CONVERT_F16_TO_F32(p[i].d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                addPair(cd * (double)((c[i].qs[k] & 0x0F) - 8), pd * (double)((p[i].qs[k] & 0x0F) - 8));
                addPair(cd * (double)((c[i].qs[k] >> 4) - 8), pd * (double)((p[i].qs[k] >> 4) - 8));
            }
        }
    } else {
        return false;
    }

    const double currNorm = std::sqrt(currSq);
    const double prevNorm = std::sqrt(prevSq);
    *deltaNormOut = (float)(std::sqrt(diffSq) / (prevNorm + eps));
    *cosineOut = (float)(dot / ((currNorm * prevNorm) + eps));
    *normRatioOut = (float)(currNorm / (prevNorm + eps));
    *currNormOut = (float)currNorm;
    *prevNormOut = (float)prevNorm;
    return true;
}

static float deterministicStageSkipUnit(NnUint position, NnUint targetStage) {
    std::uint32_t x = position * 2654435761u ^ targetStage * 2246822519u ^ 0x9E3779B9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)(x & 0x00FFFFFFu) / (float)0x01000000u;
}

static NnUint sampleArgmaxToken(const float *logits, NnUint logitsDim) {
    if (logitsDim == 0u)
        return 0u;
    NnUint bestIdx = 0;
    float bestVal = logits[0];
    for (NnUint i = 1; i < logitsDim; i++) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

static void packTopKLogits(const float *logits, NnUint logitsDim, NnUint k, std::vector<float> &packed) {
    if (k == 0 || logitsDim == 0) {
        packed.clear();
        return;
    }
    k = std::min(k, logitsDim);
    std::vector<NnUint> topIdx(k, 0u);
    std::vector<float> topVal(k, -std::numeric_limits<float>::infinity());
    NnUint filled = 0;
    for (NnUint i = 0; i < logitsDim; i++) {
        const float v = logits[i];
        if (filled < k) {
            NnUint pos = filled;
            while (pos > 0 && v > topVal[pos - 1]) {
                topVal[pos] = topVal[pos - 1];
                topIdx[pos] = topIdx[pos - 1];
                pos--;
            }
            topVal[pos] = v;
            topIdx[pos] = i;
            filled++;
            continue;
        }
        if (v <= topVal[k - 1])
            continue;
        NnUint pos = k - 1;
        while (pos > 0 && v > topVal[pos - 1]) {
            topVal[pos] = topVal[pos - 1];
            topIdx[pos] = topIdx[pos - 1];
            pos--;
        }
        topVal[pos] = v;
        topIdx[pos] = i;
    }
    packed.resize((size_t)k * 2u);
    for (NnUint i = 0; i < k; i++) {
        packed[i * 2u] = (float)topIdx[i];
        packed[i * 2u + 1u] = topVal[i];
    }
}

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

static CollectiveType parseCollectiveType(char *val) {
    if (std::strcmp(val, "auto") == 0) return COLLECTIVE_AUTO;
    if (std::strcmp(val, "star") == 0) return COLLECTIVE_STAR;
    if (std::strcmp(val, "ring") == 0) return COLLECTIVE_RING;
    throw std::runtime_error("Invalid collective type: " + std::string(val) + " (expected: auto, star, ring)");
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.promptsFile = nullptr;
    args.cleanOutputPrefix = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.collectiveType = COLLECTIVE_AUTO;
    args.ppSize = 1;
    args.spSize = 1;
    args.ppSizeExplicit = false;
    args.spSizeExplicit = false;
    args.autoPpForDistributed = true;
    args.netMonitor = false;
    args.prefillChunkSize = 0;
    args.prefillChunkThreshold = 128;
    args.wavePipeline = false;
    args.concurrentPrefillDecode = false;
    args.prefillInterleave = true;
    args.prefillQuota = 8;
    args.spPrefillOnly = true;
    args.prefillSpOnly = true;
    args.spPrefillThreshold = 256;
    // Default relaxed policy:
    // - strictKvAffinity=0: don't hard-fail decode-path KV movement experiments
    // - allowKvMigration=1: permit migration/sync when topology/scheduler needs it
    // Users can still enforce zero-KV-move runs via:
    //   --strict-kv-affinity 1 --allow-kv-migration 0 --prefill-sp-only 1
    args.strictKvAffinity = false;
    args.allowKvMigration = true;
    args.pipelineFloatType = F_UNK;
    args.ppTokenOnly = false;
    args.ppTopK = 0;
    args.stageTiming = false;
    args.wallMetrics = true;
    args.decodeLogInterval = 1;
    args.decodeCbMaxActive = 0;
    args.decodeCbMaxNewTokens = 0;
    args.pipelineDelta = false;
    args.pipelineDeltaMinBytes = 4096;
    args.pipelineChunkBytes = 0;
    args.ppStageSkipShadow = false;
    args.ppStageSkip = false;
    args.ppStageSkipTarget = 4;
    args.ppStageSkipTheta = 0.10f;
    args.ppStageSkipAlpha = 1.0f;
    args.ppStageSkipVerifierDelta = true;
    args.ppStageSkipMaxConsecutive = 3;
    args.ppStageSkipMaxRejectStreak = 8;
    args.ppStageSkipLog = false;
    args.ppStageSkipLogFile = nullptr;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // First see if any of the args are asking for help/usage and fail fast
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--prompts-file") == 0) {
            args.promptsFile = value;
        } else if (std::strcmp(name, "--clean-output-prefix") == 0) {
            args.cleanOutputPrefix = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i += count - 1;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else if (std::strcmp(name, "--collective") == 0) {
            args.collectiveType = parseCollectiveType(value);
        } else if (std::strcmp(name, "--pp-size") == 0) {
            args.ppSize = (unsigned int)atoi(value);
            args.ppSizeExplicit = true;
        } else if (std::strcmp(name, "--sp-size") == 0) {
            args.spSize = (unsigned int)atoi(value);
            args.spSizeExplicit = true;
        } else if (std::strcmp(name, "--auto-pp") == 0) {
            args.autoPpForDistributed = atoi(value) == 1;
        } else if (std::strcmp(name, "--net-monitor") == 0) {
            args.netMonitor = atoi(value) == 1;
        } else if (std::strcmp(name, "--prefill-chunk-size") == 0) {
            args.prefillChunkSize = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--prefill-chunk-threshold") == 0) {
            args.prefillChunkThreshold = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--wave-pipeline") == 0) {
            args.wavePipeline = atoi(value) == 1;
        } else if (std::strcmp(name, "--concurrent-pd") == 0) {
            args.concurrentPrefillDecode = atoi(value) == 1;
        } else if (std::strcmp(name, "--prefill-interleave") == 0) {
            args.prefillInterleave = atoi(value) == 1;
        } else if (std::strcmp(name, "--prefill-quota") == 0) {
            args.prefillQuota = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--sp-prefill-only") == 0) {
            args.spPrefillOnly = atoi(value) == 1;
            args.prefillSpOnly = args.spPrefillOnly;
        } else if (std::strcmp(name, "--prefill-sp-only") == 0) {
            args.prefillSpOnly = atoi(value) == 1;
            args.spPrefillOnly = args.prefillSpOnly;
        } else if (std::strcmp(name, "--sp-prefill-threshold") == 0) {
            args.spPrefillThreshold = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--strict-kv-affinity") == 0) {
            args.strictKvAffinity = atoi(value) == 1;
        } else if (std::strcmp(name, "--allow-kv-migration") == 0) {
            args.allowKvMigration = atoi(value) == 1;
        } else if (std::strcmp(name, "--pipeline-float-type") == 0) {
            args.pipelineFloatType = parseFloatType(value);
        } else if (std::strcmp(name, "--pp-token-only") == 0) {
            args.ppTokenOnly = atoi(value) == 1;
        } else if (std::strcmp(name, "--pp-topk") == 0) {
            args.ppTopK = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--stage-timing") == 0) {
            args.stageTiming = atoi(value) == 1;
        } else if (std::strcmp(name, "--wall-metrics") == 0) {
            args.wallMetrics = atoi(value) == 1;
        } else if (std::strcmp(name, "--decode-log-interval") == 0) {
            args.decodeLogInterval = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--decode-cb-max-active") == 0) {
            args.decodeCbMaxActive = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--decode-cb-max-new-tokens") == 0) {
            args.decodeCbMaxNewTokens = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pipeline-delta") == 0) {
            args.pipelineDelta = atoi(value) == 1;
        } else if (std::strcmp(name, "--pipeline-delta-min-bytes") == 0) {
            args.pipelineDeltaMinBytes = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pipeline-chunk-bytes") == 0) {
            args.pipelineChunkBytes = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp-stage-skip-shadow") == 0) {
            args.ppStageSkipShadow = atoi(value) == 1;
        } else if (std::strcmp(name, "--pp-stage-skip") == 0) {
            args.ppStageSkip = atoi(value) == 1;
        } else if (std::strcmp(name, "--pp-stage-skip-target") == 0) {
            args.ppStageSkipTarget = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp-stage-skip-theta") == 0) {
            args.ppStageSkipTheta = (float)atof(value);
        } else if (std::strcmp(name, "--pp-stage-skip-alpha") == 0) {
            args.ppStageSkipAlpha = (float)atof(value);
        } else if (std::strcmp(name, "--pp-stage-skip-verifier") == 0) {
            if (std::strcmp(value, "delta") != 0) {
                throw std::runtime_error("Unsupported --pp-stage-skip-verifier (v1 supports only: delta)");
            }
            args.ppStageSkipVerifierDelta = true;
        } else if (std::strcmp(name, "--pp-stage-skip-max-consecutive") == 0) {
            args.ppStageSkipMaxConsecutive = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp-stage-skip-max-reject-streak") == 0) {
            args.ppStageSkipMaxRejectStreak = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp-stage-skip-log") == 0) {
            args.ppStageSkipLog = atoi(value) == 1;
        } else if (std::strcmp(name, "--pp-stage-skip-log-file") == 0) {
            args.ppStageSkipLogFile = value;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    if (args.ppSize < 1)
        throw std::runtime_error("Pipeline size must be at least 1");
    if (args.spSize < 1)
        throw std::runtime_error("Sequence parallelism size must be at least 1");
    if (args.prefillQuota < 1)
        throw std::runtime_error("prefill quota must be at least 1");
    if (args.spPrefillThreshold < 1)
        throw std::runtime_error("sp prefill threshold must be at least 1");
    if (args.decodeCbMaxActive > MAX_CONTROL_BATCH_POS)
        throw std::runtime_error("decode-cb-max-active exceeds MAX_CONTROL_BATCH_POS");
    if (args.ppStageSkipTheta < 0.0f)
        throw std::runtime_error("pp-stage-skip-theta must be >= 0");
    if (args.ppStageSkipAlpha < 0.0f || args.ppStageSkipAlpha > 1.0f)
        throw std::runtime_error("pp-stage-skip-alpha must be in [0,1]");
    if (args.ppStageSkipMaxConsecutive < 1)
        throw std::runtime_error("pp-stage-skip-max-consecutive must be >= 1");
    if (args.ppStageSkipMaxRejectStreak < 1)
        throw std::runtime_error("pp-stage-skip-max-reject-streak must be >= 1");
    return args;
}

NnUint resolvePrefillChunkBatchSize(const AppCliArgs *args, NnUint nPrefillTokens) {
    if (args->nBatches < 1)
        return 1;

    if (args->ppSize <= 1)
        return args->nBatches;

    if (nPrefillTokens < args->prefillChunkThreshold)
        return args->nBatches;

    if (args->prefillChunkSize > 0)
        return std::min(args->nBatches, args->prefillChunkSize);

    NnUint autoChunk = args->nBatches / args->ppSize;
    if (autoChunk < 1)
        autoChunk = 1;

    if (args->ppSize >= 4)
        autoChunk = std::max<NnUint>(1, autoChunk / 2);

    NnUint pressureDivisor = args->prefillChunkThreshold > 0 ? args->prefillChunkThreshold : 1;
    NnUint pressure = nPrefillTokens / pressureDivisor;
    if (pressure >= 16)
        autoChunk = std::max<NnUint>(1, autoChunk / 4);
    else if (pressure >= 8)
        autoChunk = std::max<NnUint>(1, autoChunk / 2);

    return autoChunk;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
        NnCpuDevice *cpuDevice = new NnCpuDevice(netConfig, nodeConfig, netExecution);
        netExecution->nodeBuffers = cpuDevice->buffers;
        devices.push_back(NnExecutorDevice(cpuDevice, -1, -1));
    }
    return devices;
}

RootLlmInference::RootLlmInference(
    LlmNet *net,
    NnNetExecution *execution,
    NnExecutor *executor,
    NnNetwork *network,
    const NnParallelTopology *topology,
    NnNodeConfig *nodeConfig,
    NnUint decodeLogitsMode,
    NnUint topKCount,
    NnFloatType pipelineActivationType,
    bool stageTiming,
    bool pipelineDelta,
    NnUint pipelineDeltaMinBytes,
    NnUint pipelineChunkBytes,
    bool stageSkipShadow,
    bool stageSkipEnabled,
    NnUint stageSkipTarget,
    float stageSkipTheta,
    float stageSkipAlpha,
    NnUint stageSkipMaxConsecutive,
    NnUint stageSkipMaxRejectStreak,
    bool stageSkipLog,
    const char *stageSkipLogFilePrefix
) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[nodeConfig->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network; // May be nullptr!
    this->nodeConfig = nodeConfig;
    this->topology = topology;
    this->xPipe = execution->pipes[nodeConfig->xPipeIndex];
    this->xPipeRowBytes = net->netConfig.pipes[nodeConfig->xPipeIndex].size.nBytes / net->netConfig.nBatches;
    this->logitsPipeRowBytes = net->netConfig.pipes[net->logitsPipeIndex].size.nBytes / net->netConfig.nBatches;
    this->logitsPipeMaxBytes = net->netConfig.pipes[net->logitsPipeIndex].size.nBytes;
    this->decodeLogitsMode = decodeLogitsMode;
    this->pipelineActivationType = pipelineActivationType;
    this->stageTiming = stageTiming;
    this->tokenFromWorker = 0.0f;
    this->topKIndices.resize(topKCount);
    this->topKLogits.resize(topKCount);
    this->topKPacked.resize((size_t)topKCount * 2u);
    if (network != nullptr && topology->ppSize > 1) {
        this->pipeline.reset(new NnPipelineCommunicator(
            network,
            topology,
            nodeConfig->nodeIndex,
            pipelineDelta,
            pipelineDeltaMinBytes,
            pipelineChunkBytes
        ));
    }
    std::memset(&controlPacket, 0, sizeof(controlPacket));
    controlPacket.tokenOnlyMode = decodeLogitsMode;
    controlPacket.topKCount = topKCount;
    controlPacket.activationType = (NnUint)pipelineActivationType;
    controlPacket.pipelineDelta = pipelineDelta ? 1u : 0u;
    controlPacket.pipelineDeltaMinBytes = pipelineDeltaMinBytes;
    controlPacket.pipelineChunkBytes = pipelineChunkBytes;
    controlPacket.stageSkipShadow = stageSkipShadow ? 1u : 0u;
    controlPacket.stageSkipEnabled = stageSkipEnabled ? 1u : 0u;
    controlPacket.stageSkipTarget = stageSkipTarget;
    controlPacket.stageSkipTheta = stageSkipTheta;
    controlPacket.stageSkipAlpha = stageSkipAlpha;
    controlPacket.stageSkipMaxConsecutive = stageSkipMaxConsecutive;
    controlPacket.stageSkipMaxRejectStreak = stageSkipMaxRejectStreak;
    controlPacket.stageSkipLog = stageSkipLog ? 1u : 0u;
    controlPacket.stageSkipLogFilePrefix[0] = '\0';
    if (stageSkipLogFilePrefix != nullptr && stageSkipLogFilePrefix[0] != '\0') {
        std::snprintf(
            controlPacket.stageSkipLogFilePrefix,
            sizeof(controlPacket.stageSkipLogFilePrefix),
            "%s",
            stageSkipLogFilePrefix
        );
    }
    controlPacket.positionMode = 0u;
    setDecodePhase(false);
}

void RootLlmInference::setDecodePhase(bool isDecodePhase) {
    controlPacket.phase = isDecodePhase ? 1u : 0u;
    executor->setDecodePhase(isDecodePhase);
}

bool RootLlmInference::isDecodePhase() const {
    return controlPacket.phase == 1u;
}

bool RootLlmInference::isDecodeRemoteTokenMode() const {
    return decodeLogitsMode != 0u;
}

NnUint RootLlmInference::getTokenFromWorker() const {
    return (NnUint)tokenFromWorker;
}

int RootLlmInference::sampleToken(Sampler *sampler) {
    if (sampler == nullptr)
        throw std::runtime_error("Sampler is null");
    if (decodeLogitsMode == 1u && controlPacket.phase == 1u && controlPacket.batchSize == 1u)
        return (int)tokenFromWorker;
    if (decodeLogitsMode == 2u && controlPacket.phase == 1u && controlPacket.batchSize == 1u)
        return sampler->sampleTopK(topKIndices.data(), topKLogits.data(), (int)controlPacket.topKCount);
    return sampler->sample(logitsPipe);
}

int RootLlmInference::sampleTokenAtBatch(Sampler *sampler, NnUint batchIndex) {
    if (sampler == nullptr)
        throw std::runtime_error("Sampler is null");
    if (batchIndex >= controlPacket.batchSize)
        throw std::runtime_error("sampleTokenAtBatch batchIndex out of range");
    if (decodeLogitsMode != 0u && controlPacket.phase == 1u && controlPacket.batchSize != 1u) {
        // token-only/top-k modes are single-batch decode fast paths.
        // Fallback to full logits sampling when batch > 1.
    }
    const NnUint stride = logitsPipeRowBytes / sizeof(float);
    return sampler->sample(logitsPipe + (size_t)batchIndex * stride);
}

bool RootLlmInference::getDecodeRecvWaitStats(float *p50Ms, float *p95Ms) const {
    if (decodeRecvWaitUs.empty())
        return false;
    if (p50Ms != nullptr)
        *p50Ms = percentileMs(decodeRecvWaitUs, 0.50f);
    if (p95Ms != nullptr)
        *p95Ms = percentileMs(decodeRecvWaitUs, 0.95f);
    return true;
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    if (batchSize > MAX_CONTROL_BATCH_POS)
        throw std::runtime_error("batchSize exceeds MAX_CONTROL_BATCH_POS");
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    controlPacket.positionMode = 0u;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setBatchPositions(const NnUint *positions, NnUint batchSize) {
    if (positions == nullptr)
        throw std::runtime_error("positions is null");
    if (batchSize < 1 || batchSize > MAX_CONTROL_BATCH_POS)
        throw std::runtime_error("invalid batchSize in setBatchPositions");
    setBatchSize(batchSize);
    controlPacket.positionMode = 1u;
    controlPacket.position = positions[0];
    for (NnUint i = 0; i < batchSize; i++) {
        if (positions[i] >= header->seqLen)
            throw std::runtime_error("position out of range in setBatchPositions");
        controlPacket.batchPositions[i] = positions[i];
        positionPipe[i] = (float)positions[i];
    }
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::setSpGroupPosition(NnUint spGroupId, NnUint position, NnUint batchSize) {
    assert(spGroupId < MAX_SP_GROUPS);
    controlPacket.spGroups[spGroupId].position = position;
    controlPacket.spGroups[spGroupId].batchSize = batchSize;
    // Root is always SP group 0; update root's own positionPipe when setting group 0
    if (spGroupId == 0) {
        execution->setBatchSize(batchSize);
        controlPacket.batchSize = batchSize;
        controlPacket.position = position;
        controlPacket.positionMode = 0u;
        for (NnUint i = 0; i < batchSize; i++)
            positionPipe[i] = (float)(position + i);
    }
}

void RootLlmInference::clearSpGroupOverrides() {
    for (NnUint i = 0; i < MAX_SP_GROUPS; i++) {
        controlPacket.spGroups[i].position = 0;
        controlPacket.spGroups[i].batchSize = 0;
    }
}

void RootLlmInference::forward() {
    const unsigned long long t0 = stageTiming ? nowUs() : 0;
    if (network != nullptr) {
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
        if (shouldLogControlPackets()) {
            printf("📨 [CTRL][root->workers] position=%u batchSize=%u phase=%u payload=%zuB\n",
                controlPacket.position,
                controlPacket.batchSize,
                controlPacket.phase,
                sizeof(LlmControlPacket));
        }
    }
    executor->forward();
    const unsigned long long tExecDone = stageTiming ? nowUs() : 0;
    if (pipeline.get() != nullptr && pipeline->shouldSendActivations()) {
        const NnSize payloadBytes = xPipeRowBytes * controlPacket.batchSize;
        if (!pipeline->sendActivation(
            pipeline->getTargetPpRank(),
            controlPacket.position,
            0,
            nodeConfig->tpRank,
            xPipe,
            payloadBytes,
            pipelineActivationType
        )) {
            throw std::runtime_error("Failed to send pipeline activation from root stage");
        }
    }
    // In PP mode, receive logits from last PP stage
    if (pipeline.get() != nullptr && topology->ppSize > 1) {
        const unsigned long long tRecvStart = nowUs();
        NnSize bufferSize = logitsPipeRowBytes * controlPacket.batchSize;
        NnByte *bufferPtr = (NnByte *)logitsPipe;
        if (decodeLogitsMode == 1u && controlPacket.phase == 1u && controlPacket.batchSize == 1u) {
            bufferSize = sizeof(float);
            bufferPtr = (NnByte *)&tokenFromWorker;
        } else if (decodeLogitsMode == 2u && controlPacket.phase == 1u && controlPacket.batchSize == 1u) {
            const NnSize topKBytes = (NnSize)controlPacket.topKCount * sizeof(float) * 2u;
            if (topKLogits.size() != controlPacket.topKCount) {
                topKLogits.resize(controlPacket.topKCount);
                topKIndices.resize(controlPacket.topKCount);
                topKPacked.resize((size_t)controlPacket.topKCount * 2u);
            }
            bufferSize = topKBytes;
            bufferPtr = (NnByte *)topKPacked.data();
        }
        NnPipelineActivationHeader header;
        if (!pipeline->recvActivation(topology->ppSize - 1, &header, bufferPtr, bufferSize))
            throw std::runtime_error("Failed to receive pipeline logits from last stage");
        if (decodeLogitsMode == 2u && controlPacket.phase == 1u && controlPacket.batchSize == 1u) {
            for (NnUint i = 0; i < controlPacket.topKCount; i++) {
                topKIndices[i] = (int)topKPacked[i * 2u];
                topKLogits[i] = topKPacked[i * 2u + 1u];
            }
        }
        if (controlPacket.phase == 1u && controlPacket.batchSize == 1u) {
            const unsigned long long tRecvDone = nowUs();
            const NnUint recvWaitUs = (NnUint)(tRecvDone - tRecvStart);
            decodeRecvWaitUs.push_back(recvWaitUs);
            static const size_t kDecodeRecvWindow = 128;
            if (decodeRecvWaitUs.size() > kDecodeRecvWindow)
                decodeRecvWaitUs.erase(decodeRecvWaitUs.begin());
        }
        if (stageTiming && controlPacket.batchSize == 1u) {
            const unsigned long long tDone = nowUs();
            const char *phase = controlPacket.phase == 1u ? "decode" : "prefill";
            NnPipelineTransferStats recvStats = pipeline->getLastRecvStats();
            printf("🧭 [ROOT_STAGE] phase=%s pos=%u exec=%lluus recv_wait=%lluus recv_wire=%uus recv_decode=%uus total=%lluus wire=%zuB payload=%zuB\n",
                phase,
                controlPacket.position,
                (unsigned long long)(tExecDone - t0),
                (unsigned long long)(tDone - tRecvStart),
                recvStats.wireUs,
                recvStats.decodeUs,
                (unsigned long long)(tDone - t0),
                recvStats.wireBytes,
                recvStats.payloadBytes);
        }
    }
}

void RootLlmInference::forwardPrefillNoWait() {
    // Wave pipelining용: stage0 처리 후 activation 전송, logits recv 생략
    // Workers의 기존 루프(ctrl recv → activation recv → forward → activation send)와 호환됨
    if (network != nullptr) {
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
        if (shouldLogControlPackets()) {
            printf("📨 [WAVE][root->workers] position=%u batchSize=%u phase=%u\n",
                controlPacket.position, controlPacket.batchSize, controlPacket.phase);
        }
    }
    executor->forward();
    if (pipeline.get() != nullptr && pipeline->shouldSendActivations()) {
        const NnSize payloadBytes = xPipeRowBytes * controlPacket.batchSize;
        if (!pipeline->sendActivation(
            pipeline->getTargetPpRank(),
            controlPacket.position,
            0,
            nodeConfig->tpRank,
            xPipe,
            payloadBytes,
            pipelineActivationType
        )) {
            throw std::runtime_error("Failed to send pipeline activation (wave mode)");
        }
    }
    // PP=1이면 logitsPipe가 로컬에 이미 있으므로 별도 처리 불필요
}

void RootLlmInference::drainPrefillLogits(NnUint nChunks) {
    // wave pipelining으로 쌓인 logits를 수거
    // 마지막 chunk의 logits만 logitsPipe에 보존하고 나머지는 버림
    if (pipeline.get() == nullptr || topology->ppSize <= 1 || nChunks == 0)
        return;

    for (NnUint i = 0; i < nChunks; i++) {
        NnPipelineActivationHeader hdr;
        // 마지막 chunk만 logitsPipe에 보존, 나머지는 덮어써도 무방
        if (!pipeline->recvActivation(
            topology->ppSize - 1,
            &hdr,
            (NnByte *)logitsPipe,
            logitsPipeMaxBytes
        )) {
            throw std::runtime_error("Failed to drain prefill logits from last PP stage");
        }
    }
    // loop 종료 시 logitsPipe = 마지막 chunk의 logits
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
        if (shouldLogControlPackets()) {
            printf("🛑 [CTRL][root->workers] stop position=%u batchSize=%u phase=%u payload=%zuB\n",
                controlPacket.position,
                controlPacket.batchSize,
                controlPacket.phase,
                sizeof(LlmControlPacket));
        }
    }
}

WorkerLlmInference::WorkerLlmInference(
    NnNetExecution *execution,
    NnNetwork *network,
    const NnParallelTopology *topology,
    NnNodeConfig *nodeConfig,
    NnNetConfig *netConfig,
    NnFloatType pipelineActivationType,
    NnFloatType xPipeBufferType,
    bool pipelineDelta,
    NnUint pipelineDeltaMinBytes,
    NnUint pipelineChunkBytes,
    const char *ppStageSkipLogFilePath
) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->nodeConfig = nodeConfig;
    this->maxBatchSize = netConfig->nBatches;
    this->positionPipe = (float *)execution->pipes[nodeConfig->positionPipeIndex];
    this->xPipe = execution->pipes[nodeConfig->xPipeIndex];
    this->xPipeRowBytes = netConfig->pipes[nodeConfig->xPipeIndex].size.nBytes / netConfig->nBatches;
    this->logitsPipe = execution->pipes[nodeConfig->logitsPipeIndex];
    this->logitsPipeRowBytes = netConfig->pipes[nodeConfig->logitsPipeIndex].size.nBytes / netConfig->nBatches;
    this->logitsDim = logitsPipeRowBytes / sizeof(float);
    this->decodePhase = false;
    this->pipelineActivationType = pipelineActivationType;
    this->xPipeBufferType = xPipeBufferType;
    this->fusedLmHeadArgmax = false;
    this->topKPacked.clear();
    this->stageRecvUs.clear();
    this->stageFwdUs.clear();
    this->stageSendUs.clear();
    this->stageTotalUs.clear();
    this->stageDeltaNorm.clear();
    this->stageCosine.clear();
    this->stageNormRatio.clear();
    this->opSyncUs.clear();
    this->opNormUs.clear();
    this->opAttnUs.clear();
    this->opFfnUs.clear();
    this->opLmHeadUs.clear();
    this->opOtherUs.clear();
    this->prevXPipeRow.resize(xPipeRowBytes);
    this->hasPrevXPipeRow = false;
    this->stageSkipScore.clear();
    this->stageSkipAccept.clear();
    this->stageVerifierUs.clear();
    this->skipForwardThisToken = false;
    this->hasLastSkipDecision = false;
    this->lastSkipDecisionScore = 0.0f;
    this->lastSkipDecisionAccept = 0u;
    this->hasLastDeltaNorm = false;
    this->lastDeltaNorm = 0.0f;
    this->hasLastActivationMetrics = false;
    this->lastActivationCosine = 0.0f;
    this->lastActivationNormRatio = 0.0f;
    this->lastActivationCurrNorm = 0.0f;
    this->lastActivationPrevNorm = 0.0f;
    this->skipConsecutiveAccepts = 0u;
    this->skipRejectStreak = 0u;
    this->skipForcedFullByRejectStreak = 0u;
    this->skipLogFile = nullptr;
    this->skipLogFilePrefix.clear();
    this->skipLogRunId = (unsigned long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    this->skipLogHeaderWritten = false;
    for (NnUint s = 0; s < nodeConfig->nSegments; s++) {
        NnSegmentConfig *seg = &nodeConfig->segments[s];
        for (NnUint i = 0; i < seg->nOps; i++) {
            if (seg->ops[i].code == OP_MATMUL_ARGMAX &&
                std::strcmp(seg->ops[i].name, "final_matmul_logits") == 0) {
                this->fusedLmHeadArgmax = true;
                break;
            }
        }
        if (this->fusedLmHeadArgmax)
            break;
    }
    std::memset(&controlPacket, 0, sizeof(controlPacket));
    if (topology->ppSize > 1) {
        this->pipeline.reset(new NnPipelineCommunicator(
            network,
            topology,
            nodeConfig->nodeIndex,
            pipelineDelta,
            pipelineDeltaMinBytes,
            pipelineChunkBytes
        ));
    }
    configureStageSkipLogFile(ppStageSkipLogFilePath);
}

void WorkerLlmInference::configureStageSkipLogFile(const char *pathPrefix) {
    if (pathPrefix == nullptr || pathPrefix[0] == '\0')
        return;
    if (skipLogFile != nullptr && skipLogFilePrefix == pathPrefix)
        return;
    if (skipLogFile != nullptr) {
        std::fclose(skipLogFile);
        skipLogFile = nullptr;
        skipLogHeaderWritten = false;
    }

    skipLogFilePrefix = pathPrefix;
    {
        char path[1024];
        std::snprintf(path, sizeof(path), "%s.node%u.tsv", pathPrefix, nodeConfig->nodeIndex);
        if (!ensureParentDirectoryForFile(path)) {
            printf("⚠️  Unable to create parent directory for pp-stage-skip log file: %s (errno=%d: %s)\n",
                path, errno, std::strerror(errno));
        }
        this->skipLogFile = std::fopen(path, "a");
        if (this->skipLogFile != nullptr) {
            std::fprintf(this->skipLogFile,
                "run_id\tnode\tpp\tpos\ttarget_stage\tdelta_norm\tgate_pass\tverifier_score\tverifier_pass\twas_skip\tfallback\tconsecutive_skip\treject_streak\tforced_full_total\trecv_wait_ms\twall_step_ms\talpha\tcosine\tnorm_ratio\tcurr_norm\tprev_norm\ttoken_id\tis_special\ttoken_text\n");
            std::fflush(this->skipLogFile);
            this->skipLogHeaderWritten = true;
            printf("🧪 Opened pp-stage-skip log file: %s\n", path);
        } else {
            printf("⚠️  Unable to open pp-stage-skip log file: %s (errno=%d: %s)\n",
                path, errno, std::strerror(errno));
        }
    }
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    if (shouldLogControlPackets()) {
        printf("📩 [CTRL][worker<-root] position=%u batchSize=%u phase=%u payload=%zuB\n",
            controlPacket.position,
            controlPacket.batchSize,
            controlPacket.phase,
            sizeof(LlmControlPacket));
    }
    if (controlPacket.batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    if (controlPacket.stageSkipLog == 1u)
        configureStageSkipLogFile(controlPacket.stageSkipLogFilePrefix);

    if (controlPacket.batchSize > maxBatchSize) {
        throw NnExecutorException(
            "Invalid control packet: batchSize=" + std::to_string(controlPacket.batchSize) +
            " exceeds maxBatchSize=" + std::to_string(maxBatchSize) +
            " (possible root/worker binary mismatch)"
        );
    }

    // Concurrent P/D: check if this SP group has a per-group override
    NnUint mySpRank = nodeConfig->spRank;
    NnUint tpSize = nodeConfig->tpGroupEnd - nodeConfig->tpGroupStart;
    NnUint actualSpSize = (nodeConfig->spGroupEnd - nodeConfig->spGroupStart) / tpSize;
    NnUint activePos = controlPacket.position;
    NnUint activeBatchSize = controlPacket.batchSize;
    if (actualSpSize > 1 && mySpRank < MAX_SP_GROUPS && controlPacket.spGroups[mySpRank].batchSize > 0) {
        activePos = controlPacket.spGroups[mySpRank].position;
        activeBatchSize = controlPacket.spGroups[mySpRank].batchSize;
    }
    if (activeBatchSize == 0) {
        throw NnExecutorException("Invalid control packet: activeBatchSize is zero");
    }
    if (activeBatchSize > maxBatchSize) {
        throw NnExecutorException(
            "Invalid control packet: activeBatchSize=" + std::to_string(activeBatchSize) +
            " exceeds maxBatchSize=" + std::to_string(maxBatchSize)
        );
    }
    const bool prevDecodePhase = decodePhase;
    decodePhase = controlPacket.phase == 1u;
    if (!prevDecodePhase && decodePhase) {
        hasPrevXPipeRow = false;
        skipConsecutiveAccepts = 0u;
        skipRejectStreak = 0u;
        skipForcedFullByRejectStreak = 0u;
    }
    if (prevDecodePhase && !decodePhase) {
        hasPrevXPipeRow = false;
        skipConsecutiveAccepts = 0u;
        skipRejectStreak = 0u;
        skipForcedFullByRejectStreak = 0u;
    }
    if (controlPacket.activationType <= (NnUint)F_Q80)
        pipelineActivationType = (NnFloatType)controlPacket.activationType;
    if (pipeline.get() != nullptr) {
        pipeline->setTransportOptions(
            controlPacket.pipelineDelta == 1u,
            controlPacket.pipelineDeltaMinBytes,
            controlPacket.pipelineChunkBytes
        );
    }

    if (controlPacket.positionMode == 1u) {
        if (activeBatchSize > MAX_CONTROL_BATCH_POS)
            throw NnExecutorException("Invalid control packet: activeBatchSize exceeds MAX_CONTROL_BATCH_POS");
        for (NnUint i = 0; i < activeBatchSize; i++)
            positionPipe[i] = (float)controlPacket.batchPositions[i];
    } else {
        for (NnUint i = 0; i < activeBatchSize; i++)
            positionPipe[i] = (float)(activePos + i);
    }
    execution->setBatchSize(activeBatchSize);
    return true;
}

bool WorkerLlmInference::isDecodePhase() const {
    return decodePhase;
}

NnUint WorkerLlmInference::getPosition() const {
    return controlPacket.position;
}

NnUint WorkerLlmInference::getBatchSize() const {
    return execution->batchSize;
}

bool WorkerLlmInference::shouldSkipForward() const {
    return skipForwardThisToken;
}

bool WorkerLlmInference::shouldRecordSkipLog() const {
    return controlPacket.stageSkipLog == 1u;
}

bool WorkerLlmInference::isStageSkipDecisionStage() const {
    if (controlPacket.stageSkipTarget == 0u)
        return false;
    return nodeConfig->ppRank + 1u == controlPacket.stageSkipTarget;
}

bool WorkerLlmInference::isStageSkipTargetStage() const {
    return nodeConfig->ppRank == controlPacket.stageSkipTarget;
}

bool WorkerLlmInference::isStageSkipPostTargetStage() const {
    return nodeConfig->ppRank == controlPacket.stageSkipTarget + 1u;
}

void WorkerLlmInference::evaluateStageSkipDecision() {
    hasLastSkipDecision = false;
    lastSkipDecisionScore = 0.0f;
    lastSkipDecisionAccept = 0u;

    const bool stageSkipTrackingEnabled =
        (controlPacket.stageSkipShadow == 1u || controlPacket.stageSkipEnabled == 1u);
    if (!stageSkipTrackingEnabled || !isStageSkipDecisionStage())
        return;
    if (!decodePhase || execution->batchSize != 1u)
        return;

    if (!hasLastDeltaNorm)
        return;

    const unsigned long long tVerifier0 = nowUs();
    hasLastSkipDecision = true;
    lastSkipDecisionScore = lastDeltaNorm;
    bool accept = lastDeltaNorm <= controlPacket.stageSkipTheta;
    if (accept) {
        if (controlPacket.stageSkipAlpha <= 0.0f) {
            accept = false;
        } else if (controlPacket.stageSkipAlpha < 1.0f) {
            accept = deterministicStageSkipUnit(
                controlPacket.position,
                controlPacket.stageSkipTarget
            ) < controlPacket.stageSkipAlpha;
        }
    }
    bool forceFullByRejectStreak = false;
    if (accept && skipConsecutiveAccepts >= controlPacket.stageSkipMaxConsecutive)
        accept = false;
    if (skipRejectStreak >= controlPacket.stageSkipMaxRejectStreak) {
        // Dampening: force one full-path token after a long reject streak.
        accept = false;
        forceFullByRejectStreak = true;
    }
    lastSkipDecisionAccept = accept ? 1u : 0u;
    if (accept) {
        skipConsecutiveAccepts++;
        skipRejectStreak = 0u;
    } else {
        skipConsecutiveAccepts = 0u;
        if (forceFullByRejectStreak) {
            skipRejectStreak = 0u;
            skipForcedFullByRejectStreak++;
        } else {
            skipRejectStreak++;
        }
    }
    const unsigned long long tVerifier1 = nowUs();
    stageVerifierUs.push_back((NnUint)(tVerifier1 - tVerifier0));
}

void WorkerLlmInference::recordStageTiming(NnUint recvUs, NnUint fwdUs, NnUint sendUs, NnUint totalUs) {
    if (!decodePhase || execution->batchSize != 1u)
        return;
    stageRecvUs.push_back(recvUs);
    stageFwdUs.push_back(fwdUs);
    stageSendUs.push_back(sendUs);
    stageTotalUs.push_back(totalUs);
    if (pipeline.get() != nullptr) {
        const NnPipelineTransferStats &recvStats = pipeline->getLastRecvStats();
        const NnPipelineTransferStats &sendStats = pipeline->getLastSendStats();
        stageRecvWireUs.push_back(recvStats.wireUs);
        stageRecvDecodeUs.push_back(recvStats.decodeUs);
        stageSendWireUs.push_back(sendStats.wireUs);
    }

    const bool stageSkipTrackingEnabled =
        (controlPacket.stageSkipShadow == 1u || controlPacket.stageSkipEnabled == 1u);
    if (!stageSkipTrackingEnabled)
        return;
    if (hasLastDeltaNorm)
        stageDeltaNorm.push_back(lastDeltaNorm);
    if (hasLastActivationMetrics) {
        stageCosine.push_back(lastActivationCosine);
        stageNormRatio.push_back(lastActivationNormRatio);
    }
    if (!isStageSkipDecisionStage())
        return;
    if (hasLastSkipDecision) {
        stageSkipScore.push_back(lastSkipDecisionScore);
        stageSkipAccept.push_back(lastSkipDecisionAccept);
        const NnUint fallback = lastSkipDecisionAccept == 1u ? 0u : 1u;
        if (controlPacket.stageSkipLog == 1u) {
            const NnUint rejectStreakNow = skipRejectStreak;
            const NnUint consecNow = skipConsecutiveAccepts;
            printf("🧪 [WORKER_SKIP_TOKEN] node=%u pp=%u pos=%u delta_norm=%.6f cosine=%.6f norm_ratio=%.6f score=%.6f theta=%.6f alpha=%.3f accept=%u consecutive_skip=%u reject_streak=%u forced_full_total=%u mode=%s\n",
                nodeConfig->nodeIndex,
                nodeConfig->ppRank,
                controlPacket.position,
                lastSkipDecisionScore,
                lastActivationCosine,
                lastActivationNormRatio,
                lastSkipDecisionScore,
                controlPacket.stageSkipTheta,
                controlPacket.stageSkipAlpha,
                lastSkipDecisionAccept,
                consecNow,
                rejectStreakNow,
                skipForcedFullByRejectStreak,
                controlPacket.stageSkipEnabled == 1u ? "execute" : "shadow");
        }
        if (skipLogFile != nullptr) {
            std::fprintf(skipLogFile,
                "%llu\t%u\t%u\t%u\t%u\t%.6f\t%u\t%.6f\t%u\t%u\t%u\t%u\t%u\t%u\t%.3f\t%.3f\t%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t-1\t-1\t\n",
                skipLogRunId,
                nodeConfig->nodeIndex,
                nodeConfig->ppRank,
                controlPacket.position,
                controlPacket.stageSkipTarget,
                lastSkipDecisionScore,
                lastSkipDecisionAccept,
                lastSkipDecisionScore,
                lastSkipDecisionAccept,
                lastSkipDecisionAccept,
                fallback,
                skipConsecutiveAccepts,
                skipRejectStreak,
                skipForcedFullByRejectStreak,
                recvUs / 1000.0f,
                totalUs / 1000.0f,
                controlPacket.stageSkipAlpha,
                lastActivationCosine,
                lastActivationNormRatio,
                lastActivationCurrNorm,
                lastActivationPrevNorm);
            std::fflush(skipLogFile);
        }
    }
}

void WorkerLlmInference::recordOpTiming(const NnExecutorOpBreakdown &opBreakdown) {
    if (!decodePhase || execution->batchSize != 1u)
        return;
    opSyncUs.push_back(opBreakdown.syncUs);
    opNormUs.push_back(opBreakdown.normUs);
    opAttnUs.push_back(opBreakdown.attnUs);
    opFfnUs.push_back(opBreakdown.ffnUs);
    opLmHeadUs.push_back(opBreakdown.lmHeadUs);
    opOtherUs.push_back(opBreakdown.otherUs);
}

void WorkerLlmInference::printStageTimingSummary() const {
    if (stageTotalUs.empty())
        return;
    const float recvP50 = percentileMs(stageRecvUs, 0.50f);
    const float recvP95 = percentileMs(stageRecvUs, 0.95f);
    const float fwdP50 = percentileMs(stageFwdUs, 0.50f);
    const float fwdP95 = percentileMs(stageFwdUs, 0.95f);
    const float sendP50 = percentileMs(stageSendUs, 0.50f);
    const float sendP95 = percentileMs(stageSendUs, 0.95f);
    const float totalP50 = percentileMs(stageTotalUs, 0.50f);
    const float totalP95 = percentileMs(stageTotalUs, 0.95f);
    printf("🧭 [WORKER_STAGE_SUMMARY] node=%u pp=%u sp=%u n=%zu recv_ms(p50/p95)=%.2f/%.2f fwd_ms(p50/p95)=%.2f/%.2f send_ms(p50/p95)=%.2f/%.2f total_ms(p50/p95)=%.2f/%.2f\n",
        nodeConfig->nodeIndex,
        nodeConfig->ppRank,
        nodeConfig->spRank,
        stageTotalUs.size(),
        recvP50, recvP95,
        fwdP50, fwdP95,
        sendP50, sendP95,
        totalP50, totalP95);
    if (!stageRecvWireUs.empty()) {
        const float recvWireP50 = percentileMs(stageRecvWireUs, 0.50f);
        const float recvWireP95 = percentileMs(stageRecvWireUs, 0.95f);
        const float recvDecodeP50 = percentileMs(stageRecvDecodeUs, 0.50f);
        const float recvDecodeP95 = percentileMs(stageRecvDecodeUs, 0.95f);
        const float sendWireP50 = percentileMs(stageSendWireUs, 0.50f);
        const float sendWireP95 = percentileMs(stageSendWireUs, 0.95f);
        printf("🧭 [WORKER_XFER_SUMMARY] node=%u pp=%u sp=%u n=%zu recv_wire_ms(p50/p95)=%.2f/%.2f recv_decode_ms(p50/p95)=%.2f/%.2f send_wire_ms(p50/p95)=%.2f/%.2f\n",
            nodeConfig->nodeIndex,
            nodeConfig->ppRank,
            nodeConfig->spRank,
            stageRecvWireUs.size(),
            recvWireP50, recvWireP95,
            recvDecodeP50, recvDecodeP95,
            sendWireP50, sendWireP95);
    }
    if (!opAttnUs.empty()) {
        const float syncP50 = percentileMs(opSyncUs, 0.50f);
        const float syncP95 = percentileMs(opSyncUs, 0.95f);
        const float normP50 = percentileMs(opNormUs, 0.50f);
        const float normP95 = percentileMs(opNormUs, 0.95f);
        const float attnP50 = percentileMs(opAttnUs, 0.50f);
        const float attnP95 = percentileMs(opAttnUs, 0.95f);
        const float ffnP50 = percentileMs(opFfnUs, 0.50f);
        const float ffnP95 = percentileMs(opFfnUs, 0.95f);
        const float lmHeadP50 = percentileMs(opLmHeadUs, 0.50f);
        const float lmHeadP95 = percentileMs(opLmHeadUs, 0.95f);
        const float otherP50 = percentileMs(opOtherUs, 0.50f);
        const float otherP95 = percentileMs(opOtherUs, 0.95f);
        printf("🧭 [WORKER_OP_SUMMARY] node=%u pp=%u sp=%u n=%zu sync_ms(p50/p95)=%.2f/%.2f norm_ms(p50/p95)=%.2f/%.2f attn_ms(p50/p95)=%.2f/%.2f ffn_ms(p50/p95)=%.2f/%.2f lm_head_ms(p50/p95)=%.2f/%.2f other_ms(p50/p95)=%.2f/%.2f\n",
            nodeConfig->nodeIndex,
            nodeConfig->ppRank,
            nodeConfig->spRank,
            opAttnUs.size(),
            syncP50, syncP95,
            normP50, normP95,
            attnP50, attnP95,
            ffnP50, ffnP95,
            lmHeadP50, lmHeadP95,
            otherP50, otherP95);
    }
    if (!stageDeltaNorm.empty()) {
        const float deltaP50 = percentileValue(stageDeltaNorm, 0.50f);
        const float deltaP95 = percentileValue(stageDeltaNorm, 0.95f);
        const float cosineP50 = stageCosine.empty() ? 0.0f : percentileValue(stageCosine, 0.50f);
        const float cosineP05 = stageCosine.empty() ? 0.0f : percentileValue(stageCosine, 0.05f);
        const float normRatioP50 = stageNormRatio.empty() ? 0.0f : percentileValue(stageNormRatio, 0.50f);
        const float normRatioP95 = stageNormRatio.empty() ? 0.0f : percentileValue(stageNormRatio, 0.95f);
        float acceptRate = 0.0f;
        float rejectRate = 0.0f;
        float fallbackRate = 0.0f;
        if (!stageSkipAccept.empty()) {
            NnUint accepts = 0;
            for (NnUint v : stageSkipAccept)
                accepts += v;
            acceptRate = (float)accepts / (float)stageSkipAccept.size();
            rejectRate = 1.0f - acceptRate;
            fallbackRate = rejectRate;
        }
        const float verifierAvgMs = stageVerifierUs.empty()
            ? 0.0f
            : (float)std::accumulate(stageVerifierUs.begin(), stageVerifierUs.end(), 0.0) /
                (1000.0f * (float)stageVerifierUs.size());
        printf("🧪 [WORKER_SKIP_SUMMARY] node=%u pp=%u sp=%u n=%zu delta_norm(p50/p95)=%.6f/%.6f cosine(p05/p50)=%.6f/%.6f norm_ratio(p50/p95)=%.6f/%.6f theta=%.6f alpha=%.3f accept_rate=%.4f reject_rate=%.4f fallback_rate=%.4f verifier_avg_ms=%.4f forced_full=%u target=%u mode=%s\n",
            nodeConfig->nodeIndex,
            nodeConfig->ppRank,
            nodeConfig->spRank,
            stageDeltaNorm.size(),
            deltaP50,
            deltaP95,
            cosineP05,
            cosineP50,
            normRatioP50,
            normRatioP95,
            controlPacket.stageSkipTheta,
            controlPacket.stageSkipAlpha,
            acceptRate,
            rejectRate,
            fallbackRate,
            verifierAvgMs,
            skipForcedFullByRejectStreak,
            controlPacket.stageSkipTarget,
            controlPacket.stageSkipEnabled == 1u ? "execute" : "shadow");
    }
}

bool WorkerLlmInference::getLastPipelineRecvStats(NnPipelineTransferStats *stats) const {
    if (stats == nullptr || pipeline.get() == nullptr)
        return false;
    *stats = pipeline->getLastRecvStats();
    return true;
}

bool WorkerLlmInference::getLastPipelineSendStats(NnPipelineTransferStats *stats) const {
    if (stats == nullptr || pipeline.get() == nullptr)
        return false;
    *stats = pipeline->getLastSendStats();
    return true;
}

void WorkerLlmInference::beforeForward() {
    skipForwardThisToken = false;
    hasLastDeltaNorm = false;
    hasLastActivationMetrics = false;
    if (pipeline.get() == nullptr || !pipeline->shouldRecvActivations())
        return;

    NnPipelineActivationHeader header;
    const NnSize bufferSize = xPipeRowBytes * execution->batchSize;
    const bool stageSkipExecute = controlPacket.stageSkipEnabled == 1u;
    const bool decodeSingle = decodePhase && execution->batchSize == 1u;

    if (stageSkipExecute && decodeSingle && isStageSkipPostTargetStage()) {
        const NnUint bypassSource = controlPacket.stageSkipTarget - 1u;
        const NnUint normalSource = controlPacket.stageSkipTarget;
        bool received = false;
        const unsigned long long tPollStart = nowUs();
        const unsigned long long kPollTimeoutUs = 30ull * 1000ull * 1000ull;
        while (!received) {
            if (pipeline->recvActivation(bypassSource, &header, xPipe, bufferSize, 1ul)) {
                received = true;
                break;
            }
            if (pipeline->recvActivation(normalSource, &header, xPipe, bufferSize, 1ul)) {
                received = true;
                break;
            }
            if (nowUs() - tPollStart >= kPollTimeoutUs) {
                throw NnExecutorException("Stage-skip polling timeout waiting for pp3/pp4 activation");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    } else {
        if (!pipeline->recvActivation(pipeline->getSourcePpRank(), &header, xPipe, bufferSize))
            throw NnExecutorException("Failed to receive pipeline activation");
    }

    if (stageSkipExecute && decodeSingle && isStageSkipTargetStage()) {
        if ((header.flags & PIPELINE_FLAG_STAGE_SKIP_BYPASS) != 0u && header.payloadBytes == 0u) {
            skipForwardThisToken = true;
            return;
        }
    }
    if (header.seqPosition != controlPacket.position) {
        throw NnExecutorException("Pipeline activation position mismatch");
    }

    const bool stageSkipTrackingEnabled =
        (controlPacket.stageSkipShadow == 1u || controlPacket.stageSkipEnabled == 1u);
    if (stageSkipTrackingEnabled && decodeSingle) {
        if (hasPrevXPipeRow && prevXPipeRow.size() == xPipeRowBytes) {
            float deltaNorm = 0.0f;
            float cosine = 0.0f;
            float normRatio = 0.0f;
            float currNorm = 0.0f;
            float prevNorm = 0.0f;
            if (computeActivationPairMetricsEncoded(
                xPipe,
                prevXPipeRow.data(),
                xPipeRowBytes,
                xPipeBufferType,
                &deltaNorm,
                &cosine,
                &normRatio,
                &currNorm,
                &prevNorm
            )) {
                hasLastDeltaNorm = true;
                lastDeltaNorm = deltaNorm;
                hasLastActivationMetrics = true;
                lastActivationCosine = cosine;
                lastActivationNormRatio = normRatio;
                lastActivationCurrNorm = currNorm;
                lastActivationPrevNorm = prevNorm;
            }
        }
        std::memcpy(prevXPipeRow.data(), xPipe, xPipeRowBytes);
        hasPrevXPipeRow = true;
    }
}

void WorkerLlmInference::afterForward() {
    if (pipeline.get() == nullptr)
        return;
    const bool stageSkipExecute = controlPacket.stageSkipEnabled == 1u;
    const bool decodeSingle = decodePhase && execution->batchSize == 1u;
    evaluateStageSkipDecision();

    if (stageSkipExecute && decodeSingle && isStageSkipTargetStage() && skipForwardThisToken)
        return;

    if (pipeline->shouldSendActivations()) {
        if (stageSkipExecute && decodeSingle && isStageSkipDecisionStage() &&
            hasLastSkipDecision && lastSkipDecisionAccept == 1u) {
            // Notify skipped stage with header-only marker so it can skip forward() safely.
            static const float kDummy = 0.0f;
            if (!pipeline->sendActivation(
                controlPacket.stageSkipTarget,
                controlPacket.position,
                0,
                nodeConfig->tpRank,
                reinterpret_cast<const NnByte *>(&kDummy),
                0,
                F_32,
                PIPELINE_FLAG_STAGE_SKIP_BYPASS
            )) {
                throw NnExecutorException("Failed to send stage-skip marker to target stage");
            }

            // Bypass skipped stage: pp(target-1) -> pp(target+1).
            const NnUint bypassTarget = controlPacket.stageSkipTarget + 1u;
            const NnSize payloadBytes = xPipeRowBytes * execution->batchSize;
            if (!pipeline->sendActivation(
                bypassTarget,
                controlPacket.position,
                0,
                nodeConfig->tpRank,
                xPipe,
                payloadBytes,
                pipelineActivationType,
                PIPELINE_FLAG_STAGE_SKIP_BYPASS
            )) {
                throw NnExecutorException("Failed to send stage-skip bypass activation");
            }
            return;
        }

        // Forward xPipe to next PP stage
        const NnSize payloadBytes = xPipeRowBytes * execution->batchSize;
        if (!pipeline->sendActivation(
            pipeline->getTargetPpRank(),
            controlPacket.position,
            0,
            nodeConfig->tpRank,
            xPipe,
            payloadBytes,
            pipelineActivationType
        )) {
            throw NnExecutorException("Failed to send pipeline activation");
        }
    } else if (nodeConfig->ppRank > 0 && nodeConfig->spRank == 0 && nodeConfig->tpRank == 0) {
        // Last PP stage, SP rank 0 and TP rank 0 only: send logits back to root.
        // Root receives from a single upstream producer on the matching lane/rank.
        // Restricting to tpRank=0 prevents multi-sender socket backpressure/deadlock
        // when TP > 1.
        NnSize payloadBytes = logitsPipeRowBytes * execution->batchSize;
        NnByte *payloadPtr = logitsPipe;
        float sampledToken = 0.0f;
        if (controlPacket.tokenOnlyMode == 1u && decodePhase && execution->batchSize == 1u) {
            if (fusedLmHeadArgmax) {
                sampledToken = ((float *)logitsPipe)[0];
            } else {
                const float *logits = (const float *)logitsPipe;
                sampledToken = (float)sampleArgmaxToken(logits, logitsDim);
            }
            payloadBytes = sizeof(float);
            payloadPtr = (NnByte *)&sampledToken;
        } else if (controlPacket.tokenOnlyMode == 2u && decodePhase && execution->batchSize == 1u) {
            const float *logits = (const float *)logitsPipe;
            const NnUint k = std::min(controlPacket.topKCount, logitsDim);
            packTopKLogits(logits, logitsDim, k, topKPacked);
            payloadBytes = topKPacked.size() * sizeof(float);
            payloadPtr = (NnByte *)topKPacked.data();
        }
        if (!pipeline->sendActivation(
            0,
            controlPacket.position,
            0,
            nodeConfig->tpRank,
            payloadPtr,
            payloadBytes,
            F_32
        )) {
            throw NnExecutorException("Failed to send pipeline logits to root");
        }
    }
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;

    // For distributed CPU runs, default to PP-first unless user explicitly set PP/SP.
    // This avoids accidental TP-only topology (pp=1,tp=nNodes), which is very slow on WAN/VPN.
    if (nNodes > 1 &&
        args->autoPpForDistributed &&
        !args->ppSizeExplicit &&
        !args->spSizeExplicit &&
        args->ppSize == 1 &&
        args->spSize == 1) {
        args->ppSize = nNodes;
        printf("⚙️  Auto PP heuristic: using pp=%u sp=1 tp=1 (override with --pp-size/--sp-size or --auto-pp 0)\n", nNodes);
    }

    NnParallelTopology topology = createPPxSPxTPTopology(nNodes, args->ppSize, args->spSize);
    NnFloatType pipelineActivationType = args->pipelineFloatType == F_UNK
        ? args->syncType
        : args->pipelineFloatType;
    if (pipelineActivationType != F_32 &&
        pipelineActivationType != F_16 &&
        pipelineActivationType != F_Q40 &&
        pipelineActivationType != F_Q80) {
        throw std::runtime_error("Unsupported --pipeline-float-type");
    }
    printf("📦 ControlPacket size: %zu bytes\n", sizeof(LlmControlPacket));
    if (args->strictKvAffinity &&
        topology.spSize > 1 &&
        !args->prefillSpOnly &&
        !args->allowKvMigration) {
        throw std::runtime_error("--strict-kv-affinity=1 requires --prefill-sp-only=1 for sp-size>1 (or set --allow-kv-migration=1 for migration experiments)");
    }
    if ((args->ppStageSkip || args->ppStageSkipShadow) && topology.ppSize <= 2) {
        throw std::runtime_error("pp-stage-skip requires pp-size >= 3");
    }
    if ((args->ppStageSkip || args->ppStageSkipShadow) &&
        (args->ppStageSkipTarget == 0 || args->ppStageSkipTarget >= topology.ppSize - 1)) {
        throw std::runtime_error("pp-stage-skip-target must be in [1, pp-size-2]");
    }
    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    if (nNodes > header.nKvHeads)
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");
    const bool ppTokenOnlyActive = args->ppTokenOnly &&
        topology.ppSize > 1 &&
        topology.spSize == 1 &&
        topology.tpSize == 1;
    const bool ppTopKActive = args->ppTopK > 0 &&
        topology.ppSize > 1 &&
        topology.spSize == 1 &&
        topology.tpSize == 1 &&
        std::strcmp(args->mode, "perplexity") != 0;
    const NnUint decodeLogitsMode = ppTopKActive ? 2u : (ppTokenOnlyActive ? 1u : 0u);
    const NnUint topKCount = ppTopKActive ? std::min(args->ppTopK, header.vocabSize) : 0u;

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);

    LlmNet net = buildLlmNet(&header, topology, args->nBatches, nullptr, decodeLogitsMode == 1u);
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        synchronizer.reset(new NnNetworkNodeSynchronizer(
            network,
            &execution,
            &net.netConfig,
            rootNodeConfig,
            args->collectiveType,
            args->prefillSpOnly,
            args->strictKvAffinity,
            args->allowKvMigration
        ));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution);
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);
    executor.setStepProfilingEnabled(args->stageTiming);

    NnRootWeightLoader weightLoader(&executor, network, nNodes, topology.ppSize, header.nLayers);
    loadLlmNetWeight(args->modelPath, &net, &weightLoader);

    RootLlmInference inference(
        &net,
        &execution,
        &executor,
        network,
        &topology,
        rootNodeConfig,
        decodeLogitsMode,
        topKCount,
        pipelineActivationType,
        args->stageTiming,
        args->pipelineDelta,
        args->pipelineDeltaMinBytes,
        args->pipelineChunkBytes,
        args->ppStageSkipShadow,
        args->ppStageSkip,
        args->ppStageSkipTarget,
        args->ppStageSkipTheta,
        args->ppStageSkipAlpha,
        args->ppStageSkipMaxConsecutive,
        args->ppStageSkipMaxRejectStreak,
        args->ppStageSkipLog,
        args->ppStageSkipLogFile
    );

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }

        const char *collectiveName = "auto";
        if (args->collectiveType == COLLECTIVE_STAR) collectiveName = "star";
        else if (args->collectiveType == COLLECTIVE_RING) collectiveName = "ring";
        const NnNodePlacement rootPlacement = topology.getPlacement(0);
        const NnUint laneCount = topology.spSize * topology.tpSize;
        const NnUint rootLane = rootPlacement.spRank * topology.tpSize + rootPlacement.tpRank;
        printf("📡 Collective: %s (nNodes=%d)\n", collectiveName, nNodes);
        printf("🔀 Topology: pp=%u sp=%u tp=%u\n", topology.ppSize, topology.spSize, topology.tpSize);
        printf("🚚 Pipeline activation transport: %s\n", floatTypeToString(pipelineActivationType));
        printf("🛣️  Lanes: %u (lane = full PP chain, G=floor(N/P)); root lane=%u\n", laneCount, rootLane);
        if (args->ppTopK > 0 && !ppTopKActive)
            printf("⚠️  PP top-k requested but unsupported for this mode/topology (requires mode!=perplexity, pp>1, sp=1, tp=1)\n");
        if (args->ppTokenOnly && !ppTokenOnlyActive)
            printf("⚠️  PP token-only requested but unsupported for this topology (requires pp>1, sp=1, tp=1)\n");
        else if (ppTopKActive)
            printf("⚡ PP top-k decode fast path: ENABLED (k=%u from last PP stage)\n", topKCount);
        else if (ppTokenOnlyActive)
            printf("⚡ PP token-only decode fast path: ENABLED (argmax token from last PP stage)\n");
        printf("🔷 Prefill-only SP: %s (threshold=%u)\n",
            args->prefillSpOnly ? "on" : "off",
            args->spPrefillThreshold);
        printf("🧷 KV affinity: strict=%s allowMigration=%s\n",
            args->strictKvAffinity ? "on" : "off",
            args->allowKvMigration ? "on" : "off");
        printf("🫧 Prefill interleave: %s (quota=%u decode steps)\n",
            args->prefillInterleave ? "on" : "off",
            args->prefillQuota);
        if (args->pipelineChunkBytes > 0)
            printf("📦 Pipeline chunk transport: ENABLED (chunkBytes=%u)\n", args->pipelineChunkBytes);
        if (args->pipelineDelta)
            printf("🧮 Pipeline delta transport: ENABLED (minBytes=%u, wireDtype=%s)\n",
                args->pipelineDeltaMinBytes, floatTypeToString(pipelineActivationType));
        if (args->ppStageSkipShadow || args->ppStageSkip) {
            printf("🧪 PP stage-skip: mode=%s target=%u theta=%.6f alpha=%.3f maxConsecutive=%u maxRejectStreak=%u log=%s\n",
                args->ppStageSkip ? "execute" : "shadow",
                args->ppStageSkipTarget,
                args->ppStageSkipTheta,
                args->ppStageSkipAlpha,
                args->ppStageSkipMaxConsecutive,
                args->ppStageSkipMaxRejectStreak,
                args->ppStageSkipLog ? "on" : "off");
        }
        if (args->concurrentPrefillDecode) {
            if (topology.spSize < 2)
                throw std::runtime_error("--concurrent-pd requires --sp-size >= 2");
            printf("⚡ Concurrent Prefill/Decode: ENABLED (spSize=%u)\n", topology.spSize);
        }
        if (args->stageTiming)
            printf("🧭 Stage timing: ENABLED (root/worker per-step recv/forward/send timing)\n");

        if (args->netMonitor)
            network->enablePerformanceMonitoring(true);
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;

    handler(&context);

    inference.finish();
    
    // Print network performance report and bottleneck analysis
    if (network != nullptr) {
        printf("📦 Root link traffic summary\n");
        // 경계를 지역변수로 고정한다.
        // 매 반복마다 network->nSockets 를 다시 읽으면, 종료 경로에서 이 객체가
        // 이미 정리된 뒤라 값이 불안정해져 루프가 폭주한다(로그가 18.9 GB 까지 자란 사례).
        const NnUint nSocketsSnapshot = network->nSockets;
        for (NnUint i = 0; i < nSocketsSnapshot; i++) {
            std::string label = "root<->worker[" + std::to_string(i + 1) + "]";
            network->printSocketTrafficSummary(i, label.c_str());
        }

        if (args->netMonitor) {
            network->printPerformanceReport();
            network->printBottleneckAnalysis();
        }
    }
}

void runWorkerApp(AppCliArgs *args) {
    printf("📦 ControlPacket size: %zu bytes\n", sizeof(LlmControlPacket));
    while (true) {
        try {
            std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
            NnNetwork *network = networkPtr.get();

            NnWorkerConfigReader configReader(network);
            NnNetConfig netConfig = configReader.readNet();
            NnNodeConfig nodeConfig = configReader.readNode();
            std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
            std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

            printNodeRequiredMemory(&netConfig, &nodeConfig);

            NnNetExecution execution(args->nThreads, &netConfig);

            std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution);
            NnNetworkNodeSynchronizer synchronizer(
                network,
                &execution,
                &netConfig,
                &nodeConfig,
                args->collectiveType,
                args->prefillSpOnly,
                args->strictKvAffinity,
                args->allowKvMigration
            );
            NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
            executor.setStepProfilingEnabled(args->stageTiming);

            NnWorkerWeightReader weightReader(&executor, network);
            weightReader.read();
            network->resetStats();

            const NnUint inferredTpSize = nodeConfig.tpGroupEnd - nodeConfig.tpGroupStart;
            const NnUint inferredSpSize = inferredTpSize > 0
                ? (nodeConfig.spGroupEnd - nodeConfig.spGroupStart) / inferredTpSize
                : 1;
            const NnUint inferredPpSize = (inferredTpSize > 0 && inferredSpSize > 0)
                ? netConfig.nNodes / (inferredSpSize * inferredTpSize)
                : 1;
            NnParallelTopology topology = createPPxSPxTPTopology(netConfig.nNodes, inferredPpSize, inferredSpSize);
            NnFloatType workerPipelineActivationType = args->pipelineFloatType == F_UNK
                ? args->syncType
                : args->pipelineFloatType;
            WorkerLlmInference inference(
                &execution,
                network,
                &topology,
                &nodeConfig,
                &netConfig,
                workerPipelineActivationType,
                args->syncType,
                args->pipelineDelta,
                args->pipelineDeltaMinBytes,
                args->pipelineChunkBytes,
                args->ppStageSkipLogFile
            );
            bool isFirstAttempt = true;
            bool isTurboEnabled = false;

            if (args->collectiveType == COLLECTIVE_RING && !args->netTurbo) {
                network->setTurbo(true);
                isTurboEnabled = true;
                printf("🚁 Network is in non-blocking mode (ring)\n");
            }

            clock_t startTime;
            while (true) {
                try {
                    if (isFirstAttempt)
                        startTime = clock();

                    if (!inference.tryReadControlPacket()) {
                        if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                            network->setTurbo(false);
                            isTurboEnabled = false;
                            printf("🚁 Network is in blocking mode\n");
                        }
                        isFirstAttempt = false;
                        continue;
                    }
                    if (inference.isFinished)
                        break;

                    if (args->netTurbo && !isTurboEnabled) {
                        network->setTurbo(true);
                        isTurboEnabled = true;
                        printf("🚁 Network is in non-blocking mode\n");
                    }
                    const bool needsWorkerTokenTiming = args->stageTiming || inference.shouldRecordSkipLog();
                    const unsigned long long tRecv0 = needsWorkerTokenTiming ? nowUs() : 0;
                    inference.beforeForward();
                    const unsigned long long tRecv1 = needsWorkerTokenTiming ? nowUs() : 0;
                    executor.setDecodePhase(inference.isDecodePhase());
                    const unsigned long long tFwd0 = needsWorkerTokenTiming ? nowUs() : 0;
                    if (!inference.shouldSkipForward())
                        executor.forward();
                    const unsigned long long tFwd1 = needsWorkerTokenTiming ? nowUs() : 0;
                    if (args->stageTiming && !inference.shouldSkipForward()) {
                        NnExecutorOpBreakdown opBreakdown;
                        if (executor.getLastForwardOpBreakdown(&opBreakdown))
                            inference.recordOpTiming(opBreakdown);
                    }
                    inference.afterForward();
                    const unsigned long long tSend1 = needsWorkerTokenTiming ? nowUs() : 0;
                    if (needsWorkerTokenTiming) {
                        inference.recordStageTiming(
                            (NnUint)(tRecv1 - tRecv0),
                            (NnUint)(tFwd1 - tFwd0),
                            (NnUint)(tSend1 - tFwd1),
                            (NnUint)(tSend1 - tRecv0));
                    }
                    if (args->stageTiming && inference.getBatchSize() == 1u) {
                        NnPipelineTransferStats recvStats;
                        NnPipelineTransferStats sendStats;
                        const bool hasRecvStats = inference.getLastPipelineRecvStats(&recvStats);
                        const bool hasSendStats = inference.getLastPipelineSendStats(&sendStats);
                        const unsigned int recvWireUs = hasRecvStats ? recvStats.wireUs : 0u;
                        const unsigned int recvDecodeUs = hasRecvStats ? recvStats.decodeUs : 0u;
                        const unsigned int sendWireUs = hasSendStats ? sendStats.wireUs : 0u;
                        printf("🧭 [WORKER_STAGE] node=%u pp=%u sp=%u phase=%s pos=%u recv=%lluus recv_wire=%uus recv_decode=%uus fwd=%lluus send=%lluus send_wire=%uus total=%lluus\n",
                            nodeConfig.nodeIndex,
                            nodeConfig.ppRank,
                            nodeConfig.spRank,
                            inference.isDecodePhase() ? "decode" : "prefill",
                            inference.getPosition(),
                            (unsigned long long)(tRecv1 - tRecv0),
                            recvWireUs,
                            recvDecodeUs,
                            (unsigned long long)(tFwd1 - tFwd0),
                            (unsigned long long)(tSend1 - tFwd1),
                            sendWireUs,
                            (unsigned long long)(tSend1 - tRecv0));
                    }
                    isFirstAttempt = true;
                } catch (const NnTransferSocketException &e) {
                    printf("🚨 Network error: %s\n", e.what());
                    break;
                } catch (const NnExecutorException &e) {
                    printf("🚨 Inference error: %s\n", e.what());
                    break;
                }
            }

            if (network->nSockets > 0) {
                if (args->stageTiming)
                    inference.printStageTimingSummary();
                printf("📦 Worker[%u] link traffic summary\n", nodeConfig.nodeIndex);
                network->printSocketTrafficSummary(ROOT_SOCKET_INDEX, "worker<->root");
                const NnUint nSocketsSnapshot = network->nSockets;   // 위와 동일한 이유
                for (NnUint socketIndex = 1; socketIndex < nSocketsSnapshot; socketIndex++) {
                    std::string label = "worker-peer-socket[" + std::to_string(socketIndex) + "]";
                    network->printSocketTrafficSummary(socketIndex, label.c_str());
                }
            }
        } catch (const NnTransferSocketException &e) {
            printf("⚠️  Worker session ended (%s). Re-listening on port %u...\n", e.what(), args->port);
            continue;
        } catch (const std::exception &e) {
            printf("⚠️  Worker loop error (%s). Re-listening on port %u...\n", e.what(), args->port);
            continue;
        }
    }
}
