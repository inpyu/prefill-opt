#ifndef APP_HPP
#define APP_HPP

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-pipeline.hpp"
#include "nn/nn-topology.hpp"
#include "nn/nn-executor.hpp"
#include "tokenizer.hpp"
#include "llm.hpp"

// SP group role assignment for concurrent Prefill/Decode
enum class SpGroupRole {
    IDLE,    // no active request
    PREFILL, // currently processing prefill tokens
    DECODE,  // currently generating decode tokens
};

// Tracks per-SP-group role assignment and request state
class SpGroupScheduler {
public:
    NnUint nSpGroups;
    std::vector<SpGroupRole> groupRoles;
    std::vector<NnUint> groupRequestIds; // which request is in each group
    NnUint nextRequestId;

    SpGroupScheduler(NnUint nSpGroups);

    // Assign the next IDLE (or oldest DECODE) group to a new prefill request.
    // Returns the assigned spGroupId, or UINT32_MAX if no group available.
    NnUint assignPrefill();

    // Mark a group as transitioned from Prefill to Decode
    void transitionToDecode(NnUint spGroupId);

    // Release a group back to IDLE (decode complete)
    void releaseGroup(NnUint spGroupId);

    // Find the group currently doing Decode, or UINT32_MAX if none
    NnUint findDecodeGroup() const;

    // Find the group currently doing Prefill, or UINT32_MAX if none
    NnUint findPrefillGroup() const;

    bool hasIdle() const;
    bool hasDecode() const;
    bool hasPrefill() const;
};

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    char *promptsFile;
    char *cleanOutputPrefix;
    NnFloatType syncType;
    NnUint nWorkers;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    CollectiveType collectiveType;
    NnUint ppSize;
    NnUint spSize;
    bool ppSizeExplicit;
    bool spSizeExplicit;
    bool autoPpForDistributed;
    bool netMonitor;
    NnUint prefillChunkSize;
    NnUint prefillChunkThreshold;
    bool wavePipeline;
    bool concurrentPrefillDecode;
    bool prefillInterleave;
    NnUint prefillQuota;
    bool spPrefillOnly;
    bool prefillSpOnly;
    NnUint spPrefillThreshold;
    // 블록 병렬 시뮬레이션 (research/07 Phase A1). 0 이면 비활성.
    NnUint simBlockSize;
    NnUint simAnchorLen;
    bool strictKvAffinity;
    bool allowKvMigration;
    NnFloatType pipelineFloatType;
    bool ppTokenOnly;
    NnUint ppTopK;
    bool stageTiming;
    bool wallMetrics;
    NnUint decodeLogInterval;
    NnUint decodeCbMaxActive;
    NnUint decodeCbMaxNewTokens;
    bool pipelineDelta;
    NnUint pipelineDeltaMinBytes;
    NnUint pipelineChunkBytes;
    bool ppStageSkipShadow;
    bool ppStageSkip;
    NnUint ppStageSkipTarget;
    float ppStageSkipTheta;
    float ppStageSkipAlpha;
    bool ppStageSkipVerifierDelta;
    NnUint ppStageSkipMaxConsecutive;
    NnUint ppStageSkipMaxRejectStreak;
    bool ppStageSkipLog;
    char *ppStageSkipLogFile;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    // worker
    NnUint port;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    ~AppCliArgs();
};

#define MAX_SP_GROUPS 8
#define MAX_CONTROL_BATCH_POS 64
#define MAX_STAGE_SKIP_LOG_PREFIX 256

typedef struct {
    NnUint position;
    NnUint batchSize; // 0 = stop signal
    NnUint phase;     // 0 = prefill/eval, 1 = decode/prediction
    NnUint positionMode; // 0 = contiguous (position+i), 1 = explicit batchPositions[i]
    NnUint tokenOnlyMode; // 0 = full logits, 1 = sampled token id, 2 = top-k logits
    NnUint topKCount; // active only when tokenOnlyMode == 2
    NnUint activationType; // NnFloatType used for stage-to-stage activation transport
    NnUint pipelineDelta; // 0 = off, 1 = on
    NnUint pipelineDeltaMinBytes;
    NnUint pipelineChunkBytes; // 0 = off
    NnUint stageSkipShadow; // 0 = off, 1 = shadow-only
    NnUint stageSkipEnabled; // 0 = off, 1 = execute
    NnUint stageSkipTarget; // pp rank to bypass
    float stageSkipTheta; // gate+verifier threshold (delta_norm)
    float stageSkipAlpha; // deterministic acceptance-rate scaler in [0,1]
    NnUint stageSkipMaxConsecutive;
    NnUint stageSkipMaxRejectStreak;
    NnUint stageSkipLog; // 0 = off, 1 = token log on
    char stageSkipLogFilePrefix[MAX_STAGE_SKIP_LOG_PREFIX];
    NnUint batchPositions[MAX_CONTROL_BATCH_POS];
    // Per-SP-group overrides (used when spSize > 1 for concurrent P/D)
    // spGroups[i].batchSize == 0 means SP group i is idle
    struct {
        NnUint position;
        NnUint batchSize;
    } spGroups[MAX_SP_GROUPS];
} LlmControlPacket;

class RootLlmInference {
public:
    float *logitsPipe;
private:
    float *tokenPipe;
    float *positionPipe;
    LlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    NnNetwork *network;
    NnNodeConfig *nodeConfig;
    const NnParallelTopology *topology;
    std::unique_ptr<NnPipelineCommunicator> pipeline;
    NnByte *xPipe;
    NnSize xPipeRowBytes;
    NnSize logitsPipeRowBytes;
    NnSize logitsPipeMaxBytes;
    LlmControlPacket controlPacket;
    NnUint decodeLogitsMode;
    NnFloatType pipelineActivationType;
    bool stageTiming;
    float tokenFromWorker;
    std::vector<int> topKIndices;
    std::vector<float> topKLogits;
    std::vector<float> topKPacked;
    std::vector<NnUint> decodeRecvWaitUs;
public:
    RootLlmInference(
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
    );
    void setDecodePhase(bool isDecodePhase);
    bool isDecodePhase() const;
    bool isDecodeRemoteTokenMode() const;
    NnUint getTokenFromWorker() const;
    int sampleToken(Sampler *sampler);
    int sampleTokenAtBatch(Sampler *sampler, NnUint batchIndex);
    bool getDecodeRecvWaitStats(float *p50Ms, float *p95Ms) const;
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setBatchPositions(const NnUint *positions, NnUint batchSize);
    void setToken(NnUint batchIndex, NnUint token);
    // Concurrent P/D: set per-SP-group position override
    // spGroupId == 0 also updates the root's own positionPipe (root is always SP group 0)
    void setSpGroupPosition(NnUint spGroupId, NnUint position, NnUint batchSize);
    // Clear all SP group overrides (for single-group operation)
    void clearSpGroupOverrides();
    void forward();
    void forwardPrefillNoWait();
    void drainPrefillLogits(NnUint nChunks);
    void finish();
};

class WorkerLlmInference {
public:
    bool isFinished;
private:
    float *positionPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    NnNodeConfig *nodeConfig;
    NnUint maxBatchSize;
    std::unique_ptr<NnPipelineCommunicator> pipeline;
    NnByte *xPipe;
    NnSize xPipeRowBytes;
    NnByte *logitsPipe;
    NnSize logitsPipeRowBytes;
    LlmControlPacket controlPacket;
    bool decodePhase;
    NnUint logitsDim;
    NnFloatType pipelineActivationType;
    bool fusedLmHeadArgmax;
    std::vector<float> topKPacked;
    std::vector<NnUint> stageRecvUs;
    std::vector<NnUint> stageFwdUs;
    std::vector<NnUint> stageSendUs;
    std::vector<NnUint> stageTotalUs;
    std::vector<NnUint> stageRecvWireUs;
    std::vector<NnUint> stageRecvDecodeUs;
    std::vector<NnUint> stageSendWireUs;
    std::vector<NnUint> opSyncUs;
    std::vector<NnUint> opNormUs;
    std::vector<NnUint> opAttnUs;
    std::vector<NnUint> opFfnUs;
    std::vector<NnUint> opLmHeadUs;
    std::vector<NnUint> opOtherUs;
    NnFloatType xPipeBufferType;
    std::vector<NnByte> prevXPipeRow;
    bool hasPrevXPipeRow;
    std::vector<float> stageDeltaNorm;
    std::vector<float> stageCosine;
    std::vector<float> stageNormRatio;
    std::vector<float> stageSkipScore;
    std::vector<NnUint> stageSkipAccept;
    std::vector<NnUint> stageVerifierUs;
    bool skipForwardThisToken;
    bool hasLastSkipDecision;
    float lastSkipDecisionScore;
    NnUint lastSkipDecisionAccept;
    bool hasLastDeltaNorm;
    float lastDeltaNorm;
    bool hasLastActivationMetrics;
    float lastActivationCosine;
    float lastActivationNormRatio;
    float lastActivationCurrNorm;
    float lastActivationPrevNorm;
    NnUint skipConsecutiveAccepts;
    NnUint skipRejectStreak;
    NnUint skipForcedFullByRejectStreak;
    FILE *skipLogFile;
    std::string skipLogFilePrefix;
    unsigned long long skipLogRunId;
    bool skipLogHeaderWritten;
    bool isStageSkipDecisionStage() const;
    bool isStageSkipTargetStage() const;
    bool isStageSkipPostTargetStage() const;
    void configureStageSkipLogFile(const char *pathPrefix);
    void evaluateStageSkipDecision();
public:
    WorkerLlmInference(
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
    );
    bool tryReadControlPacket();
    bool isDecodePhase() const;
    NnUint getPosition() const;
    NnUint getBatchSize() const;
    bool shouldSkipForward() const;
    bool shouldRecordSkipLog() const;
    void recordStageTiming(NnUint recvUs, NnUint fwdUs, NnUint sendUs, NnUint totalUs);
    void recordOpTiming(const NnExecutorOpBreakdown &opBreakdown);
    void printStageTimingSummary() const;
    bool getLastPipelineRecvStats(NnPipelineTransferStats *stats) const;
    bool getLastPipelineSendStats(NnPipelineTransferStats *stats) const;
    void beforeForward();
    void afterForward();
};

typedef struct {
    AppCliArgs *args;
    LlmHeader *header;
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    NnExecutor *executor;
} AppInferenceContext;

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context));
void runWorkerApp(AppCliArgs *args);
NnUint resolvePrefillChunkBatchSize(const AppCliArgs *args, NnUint nPrefillTokens);

#endif
