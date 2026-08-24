#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-pipeline.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <stdexcept>

// 서브블록 오프셋. 레이어 하나를 att / ff 두 서브블록으로 나눈다.
//
// 왜 필요한가 (research/06 §4.16): 레이어 단위 PP 는 N 이 커지면 조정 입자도가
// 너무 거칠어진다. N=8 이면 레이어 1개가 목표 스테이지 시간의 22 % 라, 느린 노드를
// 한 레이어 줄이면 다른 노드가 새 최대가 되어 균형화가 실패한다.
// FFN 이 레이어 FLOPs 의 81 % 이므로 att|ff 경계로 자르면 입자도가 대략 절반이 된다.
//
// 경계 자체는 기존 브리지가 그대로 처리한다 — MERGE_ADD(zqPipe→x) 후 CAST(x→xPipe)
// 는 att 뒤에서도 ff 뒤에서도 같은 의미다.
static std::vector<NnUint> buildSubBlockOffsets(NnUint nLayers, NnUint ppSize,
                                                const std::vector<NnUint> *ppStageSubCounts) {
    const NnUint nSub = nLayers * 2u;
    std::vector<NnUint> offsets(ppSize + 1, 0u);
    if (ppStageSubCounts != nullptr && !ppStageSubCounts->empty()) {
        if (ppStageSubCounts->size() != ppSize)
            throw std::runtime_error("ppStageSubCounts size mismatch with ppSize");
        NnUint sum = 0;
        for (NnUint i = 0; i < ppSize; i++) {
            if ((*ppStageSubCounts)[i] == 0u)
                throw std::runtime_error("ppStageSubCounts must be > 0 for all stages");
            offsets[i] = sum;
            sum += (*ppStageSubCounts)[i];
        }
        offsets[ppSize] = sum;
        if (sum != nSub)
            throw std::runtime_error("ppStageSubCounts sum must match nLayers*2");
        return offsets;
    }
    // 균등 분할.
    const NnUint per = nSub / ppSize;
    const NnUint rem = nSub % ppSize;
    NnUint acc = 0;
    for (NnUint i = 0; i < ppSize; i++) {
        offsets[i] = acc;
        acc += per + (i < rem ? 1u : 0u);
    }
    offsets[ppSize] = acc;
    return offsets;
}

static std::vector<NnUint> buildLayerStartOffsets(NnUint nLayers, NnUint ppSize, const std::vector<NnUint> *ppStageLayerCounts) {
    std::vector<NnUint> offsets(ppSize + 1, 0u);
    if (ppStageLayerCounts != nullptr && !ppStageLayerCounts->empty()) {
        if (ppStageLayerCounts->size() != ppSize)
            throw std::runtime_error("ppStageLayerCounts size mismatch with ppSize");
        NnUint sum = 0;
        for (NnUint i = 0; i < ppSize; i++) {
            if ((*ppStageLayerCounts)[i] == 0u)
                throw std::runtime_error("ppStageLayerCounts must be > 0 for all stages");
            offsets[i] = sum;
            sum += (*ppStageLayerCounts)[i];
        }
        offsets[ppSize] = sum;
        if (sum != nLayers)
            throw std::runtime_error("ppStageLayerCounts sum must match nLayers");
        return offsets;
    }

    // Legacy equal partition fallback.
    NnUint layersPerStage = nLayers / ppSize;
    for (NnUint i = 0; i < ppSize; i++)
        offsets[i] = i * layersPerStage;
    offsets[ppSize] = nLayers;
    return offsets;
}

static const char *hiddenActToString(LlmHiddenAct act) {
    if (act == HIDDEN_ACT_GELU) return "Gelu";
    if (act == HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    if (type == ROPE_FALCON) return "Falcon";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(LlmArchType type) {
    if (type == LLAMA) return "Llama";
    if (type == QWEN3) return "Qwen3";
    if (type == QWEN3_MOE) return "Qwen3 MoE";
    throw std::runtime_error("Unsupported architecture");
}

static float convertNormEpsilon(int value) {
    if (value == 5) return 1e-05f;
    if (value == 6) return 1e-06f;
    throw std::runtime_error("Unsupported norm epsilon");
}

LlmHeader loadLlmHeader(const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.ropeScalingFactor = 1.0f;
    header.normEpsilon = 1e-5f;
    header.moeHiddenDim = 0u;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read magic value");

    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    if (magic != 0xA00ABCD)
        throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize);
    int *buffer = &bufferPtr[0];
    if (fread(buffer, header.headerSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read header values");

    int nKv = (header.headerSize - 2 * sizeof(int)) / sizeof(int);

    for (int i = 0; i < nKv; i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == VERSION) header.version = value;
        else if (key == ARCH_TYPE) header.archType = (LlmArchType)value;
        else if (key == DIM) header.dim = value;
        else if (key == HIDDEN_DIM) header.hiddenDim = value;
        else if (key == N_LAYERS) header.nLayers = value;
        else if (key == N_HEADS) header.nHeads = value;
        else if (key == N_KV_HEADS) header.nKvHeads = value;
        else if (key == N_EXPERTS) header.nExperts = value;
        else if (key == N_ACTIVE_EXPERTS) header.nActiveExperts = value;
        else if (key == VOCAB_SIZE) header.vocabSize = value;
        else if (key == SEQ_LEN) header.seqLen = value;
        else if (key == HIDDEN_ACT) header.hiddenAct = (LlmHiddenAct)value;
        else if (key == ROPE_THETA) header.ropeTheta = (float)value;
        else if (key == WEIGHT_FLOAT_TYPE) header.weightType = (NnFloatType)value;
        else if (key == ROPE_SCALING_FACTOR) header.ropeScalingFactor = (float)value;
        else if (key == ROPE_SCALING_LOW_FREQ_FACTOR) header.ropeScalingLowFreqFactor = (float)value;
        else if (key == ROPE_SCALING_HIGH_FREQ_FACTORY) header.ropeScalingHighFreqFactory = (float)value;
        else if (key == ROPE_SCALING_ORIG_MAX_SEQ_LEN) header.ropeScalingOrigMaxSeqLen = value;
        else if (key == ROPE_TYPE) header.ropeType = (NnRopeType)value;
        else if (key == HEAD_DIM) header.headDim = value;
        else if (key == NORM_EPSILON) header.normEpsilon = convertNormEpsilon(value);
        else if (key == MOE_HIDDEN_DIM) header.moeHiddenDim = value;
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    if (header.headDim == 0)
        header.headDim = header.dim / header.nHeads;
    header.qDim = header.headDim * header.nHeads;
    header.kvDim = header.headDim * header.nKvHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);

    if (header.archType == QWEN3 || header.archType == QWEN3_MOE)
        header.ropeType = ROPE_FALCON;
    return header;
}

void printLlmHeader(LlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 QDim: %u\n", header->qDim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    }
    if (header->nExperts > 0) {
        printf("💡 nExperts: %u\n", header->nExperts);
        printf("💡 nActiveExperts: %u\n", header->nActiveExperts);
        printf("💡 MoeHiddenDim: %u\n", header->moeHiddenDim);
    }
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("💡 RopeScaling: f=%.1f, l=%.1f, h=%.1f, o=%d\n",
            header->ropeScalingFactor,
            header->ropeScalingLowFreqFactor,
            header->ropeScalingHighFreqFactory,
            header->ropeScalingOrigMaxSeqLen);
    }
}

LlmNet buildLlmNet(
    LlmHeader *h,
    const NnParallelTopology &topology,
    NnUint nBatches,
    const std::vector<NnUint> *ppStageLayerCounts,
    NnUint pruneLayer,
    bool fusedLmHeadArgmax
) {
    NnUint nNodes = topology.nNodes;
    NnUint tpSize = topology.tpSize;
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);
    NnUint ffDim = h->hiddenDim;

    if (h->archType == QWEN3_MOE)
        ffDim = h->moeHiddenDim;

    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);
    n.moeGateSize = size2D(F_32, h->dim, h->nExperts);

    NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, tpSize, nBatches);

    n.qSlice = sliceRowMatmul(h->weightType, tpSize, h->dim, h->qDim);
    n.kSlice = sliceRowMatmul(h->weightType, tpSize, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, tpSize, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, tpSize, h->qDim, h->dim);

    n.w1Slice = sliceRowMatmul(h->weightType, tpSize, h->dim, ffDim);
    n.w2Slice = sliceColMatmul(h->weightType, tpSize, ffDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, tpSize, h->dim, ffDim);
    n.wclsSlice = sliceColMatmul(h->weightType, tpSize, h->dim, h->vocabSize);
 
    NnUint nQNormColumns = 1;
    NnUint nKNormColumns = 1;
    NnUint nInvBufferColumns = 1;
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        ASSERT_EQ(n.qSlice.d0 % h->headDim, 0);
        ASSERT_EQ(n.kSlice.d0 % h->headDim, 0);
        nQNormColumns = n.qSlice.d0 / h->headDim;
        nKNormColumns = n.kSlice.d0 / h->headDim;
        nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);
    }

    NnNetConfigBuilder netBuilder(nNodes, nBatches);

    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(F_32, nBatches, h->dim));

    netBuilder.addPreSync(n.positionPipeIndex);

    n.header = h;
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes];
    // 서브블록(레이어당 att/ff 2개) 단위 분할. --pp-layers 가 서브블록 수를 주면
    // 그대로 쓰고, 없으면 균등 분할한다.
    const bool subMode = ppStageLayerCounts != nullptr && !ppStageLayerCounts->empty()
        && ppStageLayerCounts->size() == topology.ppSize
        && [&]{ NnUint sm = 0; for (NnUint c : *ppStageLayerCounts) sm += c; return sm == h->nLayers * 2u; }();
    const std::vector<NnUint> subOffsets = subMode
        ? buildSubBlockOffsets(h->nLayers, topology.ppSize, ppStageLayerCounts)
        : std::vector<NnUint>();
    const std::vector<NnUint> layerOffsets = subMode
        ? std::vector<NnUint>()
        : buildLayerStartOffsets(h->nLayers, topology.ppSize, ppStageLayerCounts);
    // 가중치 로더가 같은 분할을 쓰도록 알려준다. 이걸 빠뜨리면 불균등 분할에서
    // 가중치가 엉뚱한 노드로 간다.
    // 로더가 같은 분할을 쓰도록 **서브블록 단위**로 알려준다.
    // 레이어 단위로 주면 att/ff 가 다른 노드로 갈릴 때 가중치가 엉뚱한 곳으로 간다.
    {
        std::vector<NnUint> subForLoader;
        if (subMode) {
            subForLoader = subOffsets;
        } else {
            subForLoader.resize(layerOffsets.size());
            for (size_t i = 0; i < layerOffsets.size(); i++)
                subForLoader[i] = layerOffsets[i] * 2u;
        }
        nnNetworkSetPpLayerOffsets(subForLoader);
    }

    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        NnNodePlacement nodePlacement = topology.getPlacement(nodeIndex);
        // 진단용: DLLAMA_SP_NOSHARD=1 이면 KV 를 샤딩하지 않는다(각 노드가 전체 KV 를 쓴다).
    // spSize>1 에서 출력이 깨지는 원인이 샤딩/allgather 인지 가르기 위한 스위치다.
    const bool spNoShard = std::getenv("DLLAMA_SP_NOSHARD") != nullptr;
    NnKvCacheSlice kvCacheSlice = spNoShard
        ? sliceKvCache(h->kvDim, h->seqLen, tpSize, 1u, 0u)
        : sliceKvCache(h->kvDim, h->seqLen, tpSize, topology.spSize, nodePlacement.spRank);
        NnRopeSlice ropeSlice = sliceRope(h->ropeType, h->qDim, h->kvDim, h->nKvHeads, tpSize, h->seqLen, h->headDim, h->ropeTheta, nodePlacement.tpRank);

        // Calculate layer range for this PP stage
        // 이 노드가 맡은 서브블록 구간 [subStart, subEnd). 서브블록 s 는
        // layer = s/2, part = s%2 (0=att, 1=ff) 다.
        const NnUint subStart = subMode ? subOffsets[nodePlacement.ppRank]
                                        : layerOffsets[nodePlacement.ppRank] * 2u;
        const NnUint subEnd = subMode ? subOffsets[nodePlacement.ppRank + 1]
                                      : layerOffsets[nodePlacement.ppRank + 1] * 2u;
        NnUint layerStart = subStart / 2u;
        NnUint layerEnd = (subEnd + 1u) / 2u;
        if (topology.ppSize > 1)
            printf("🧱 node=%u ppRank=%u sub=[%u,%u) layers=[%u,%u)\n",
                nodeIndex, nodePlacement.ppRank, subStart, subEnd, layerStart, layerEnd);
        NnNodeConfigBuilder nodeBuilder(nodeIndex);

        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));

        const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, nBatches, h->qDim));
        const NnUint zqSliceBufferIndex = nodeBuilder.addBuffer("q_z_slice", size2D(h->syncType, nBatches, h->qDim / tpSize));

        const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, n.qSlice.d0));
        const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, n.kSlice.d0));
        const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, n.vSlice.d0));

        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));

        const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
        const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, h->vocabSize));
        const NnUint argmaxPartialBufferIndex = nodeBuilder.addBuffer("argmax_partial", size2D(F_32, nBatches, 128));

        // not moe
        const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint dqBufferIndex = h->syncType == F_32
            ? dBufferIndex
            : nodeBuilder.addBuffer("q_d", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, n.w3Slice.d0));

        // SharedPack-SDOT (research/19): Down 입력의 공유 block_q8_0x4 버퍼.
        //
        // 기존에는 matmul 안에서 **스레드마다** activation 전체를 중복 변환했다.
        // K 가 긴 Down(14336)에서 네 개의 독립 cache footprint 가 생겨 4스레드 확장이
        // 무너진다 — 실측 198.7 vs 공유 333.8 GOPS (1.68x, research/18 §6c).
        //
        // 크기: 총 (nBatches/4) 그룹 x kBlocks x sizeof(block_q8_0x4)
        //   block_q8_0x4 = 4 x (32 int8 + fp16 scale) = 136 B
        //
        // ⚠️ PNTR_BATCH 는 버퍼의 행 수가 nBatches 와 같기를 요구한다
        //    (nn-cpu.cpp:196 ASSERT_EQ(sourceSize->y, nBatches)).
        //    그래서 nBatches 행으로 잡고 **행당 1/4 그룹** 분량을 둔다.
        //    총량은 동일하고, pack op 는 base 포인터에서 전체를 선형 인덱싱한다.
        //      행당 float 수 = kBlocks * 34 / 4  (34 float = 136 B = block_q8_0x4)
        // yq 는 Q/K/V 와 Gate/Up 이 **모두** 입력으로 쓴다 (K = dim).
        // pack 하나로 다섯 projection 을 커버한다 — inter-projection sharing.
        const bool yPackOn = (h->syncType == F_Q80) &&
                             (h->dim % Q40_BLOCK_SIZE == 0u) &&
                             ((h->dim / Q40_BLOCK_SIZE) * 34u % 4u == 0u);
        const NnUint yPackBufferIndex = yPackOn
            ? nodeBuilder.addBuffer("y_pack", size2D(F_32, nBatches,
                  (h->dim / Q40_BLOCK_SIZE) * 34u / 4u))
            : 0u;

        const bool sharedPackOn = (h->syncType == F_Q80) &&
                                  (n.w2Slice.d % Q40_BLOCK_SIZE == 0u) &&
                                  ((n.w2Slice.d / Q40_BLOCK_SIZE) * 34u % 4u == 0u);
        const NnUint dPackBufferIndex = sharedPackOn
            ? nodeBuilder.addBuffer("d_pack", size2D(F_32, nBatches,
                  (n.w2Slice.d / Q40_BLOCK_SIZE) * 34u / 4u))
            : 0u;

        // moe
        const NnUint moeGtBufferIndex = nodeBuilder.addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
        const NnUint moeExpertIndexesBufferIndex = nodeBuilder.addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
        const NnUint moeYBufferIndex = nodeBuilder.addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeYqBufferIndex = h->syncType == F_32
            ? moeYBufferIndex
            : nodeBuilder.addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeDBufferIndex = nodeBuilder.addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeDQBufferIndex = h->syncType == F_32
            ? moeDBufferIndex
            : nodeBuilder.addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeLBufferIndex = nodeBuilder.addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, n.w3Slice.d0));
        const NnUint moeSBufferIndex = nodeBuilder.addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

        NnSegmentConfigBuilder start;
        if (nodePlacement.ppRank == 0) {
            if (nodeIndex == 0) {
                start.addOp(
                    OP_EMBEDDING, "embedding", 0,
                    pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    n.tokenEmbeddingSize,
                    NnEmbeddingOpConfig{});
            }
            if (topology.ppSize == 1)
                start.addSync(n.xPipeIndex, SYNC_WITH_ROOT);
        }
        nodeBuilder.addSegment(start.build());

        for (NnUint layerIndex = layerStart; layerIndex < layerEnd; layerIndex++) {
            const NnUint kBufferIndex = nodeBuilder.addBuffer("k", kvCacheSlice.keySize);
            const NnUint vBufferIndex = nodeBuilder.addBuffer("v", kvCacheSlice.valueSize);

            NnSegmentConfigBuilder att;
            NnSegmentConfigBuilder ff;

            const NnUint subAtt = layerIndex * 2u;
            const NnUint subFf = subAtt + 1u;
            const bool ownAtt = subAtt >= subStart && subAtt < subEnd;
            const bool ownFf = subFf >= subStart && subFf < subEnd;

            // 토큰 가지치기 (research/10). 지정된 레이어 진입 시점에 살아남은 행을
            // 앞으로 압축하고, 이후 모든 op 가 줄어든 행 수만 돈다.
            // 잔존 수가 데이터에 따라 달라지므로 스테이지 부하도 달라진다 —
            // 이것이 온라인 재분할이 필요해지는 이유다.
            if (pruneLayer != UINT32_MAX && layerIndex == pruneLayer) {
                att.addOp(
                    OP_PRUNE_TOKENS, "block_prune_tokens", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnPruneTokensOpCodeConfig{n.positionPipeIndex});
            }

            // att. 스테이지의 첫 서브블록이면 상류에서 받은 xPipe 를 x 로 옮긴다.
            if (subAtt == subStart) {
                att.addOp(
                    OP_CAST, "block_cast_x", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            } else {
                att.addOp(
                    OP_MERGE_ADD, "block_merge_add", layerIndex,
                    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnMergeAddOpCodeConfig{});
            }

            att.addOp(
                OP_INV_RMS, "block_norm_pre_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            att.addOp(
                OP_RMS_NORM, "block_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                att.addOp(
                    OP_CAST, "block_cast_y", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            // SharedPack: yq 를 한 번만 재배치해 Q·K·V 가 공유한다.
            if (yPackOn) {
                att.addOp(
                    OP_PACK_Q80X4, "block_pack_yq", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yPackBufferIndex),
                    size0(),
                    NnPackQ80x4OpCodeConfig{});
            }
            att.addOp(
                OP_MATMUL, "block_matmul_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex, yPackOn ? yPackBufferIndex : NN_NO_PREPACK});
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex, yPackOn ? yPackBufferIndex : NN_NO_PREPACK});
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex, yPackOn ? yPackBufferIndex : NN_NO_PREPACK});

            if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
                att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});

                att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
            }

            att.addOp(
                OP_ROPE, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 1, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_ROPE, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 0, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_SHIFT, "block_shift_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, kBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex, kvCacheSlice.localSeqStart, kvCacheSlice.localSeqLen});
            att.addOp(
                OP_SHIFT, "block_shift_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, vBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex, kvCacheSlice.localSeqStart, kvCacheSlice.localSeqLen});
            if (topology.spSize > 1 && !spNoShard) {
                att.addSpKvSync(n.positionPipeIndex, kBufferIndex, vBufferIndex,
                    kvCacheSlice.localSeqStart, kvCacheSlice.localSeqLen, kvCacheSlice.kvDim0);
            }
            att.addOp(
                OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                size0(),
                NnMultiHeadAttOpConfig{
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    kvCacheSlice.localSeqLen, kvCacheSlice.localSeqStart, topology.spSize,
                    n.positionPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_y2", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_wo", layerIndex,
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex, NN_NO_PREPACK});
            att.addOp(
                OP_CAST, "block_cast_d", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            // ff
            ff.addOp(
                OP_MERGE_ADD, "block_merge_add2", layerIndex,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            if (subFf == subStart) {
                // ff 가 스테이지의 첫 서브블록이다. 위 MERGE_ADD 는 이 노드가
                // 계산하지 않은 attention 결과를 더하려는 것이므로 의미가 없다.
                // 대신 상류가 보낸 x 를 받는다(브리지가 이미 잔차를 합쳐서 보낸다).
                ff = NnSegmentConfigBuilder();
                ff.addOp(
                    OP_CAST, "block_cast_x_ff", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            ff.addOp(
                OP_INV_RMS, "block_norm_pre_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            ff.addOp(
                OP_RMS_NORM, "block_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});

            if (h->archType == QWEN3_MOE) {
                ff.addOp(
                    OP_REPEAT_Z, "block_moe_y_repeat", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    size0(),
                    NnRepeatZOpCodeConfig{});
                ff.addOp(
                    OP_MATMUL, "block_moe_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    n.moeGateSize,
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex, NN_NO_PREPACK});
                ff.addOp(
                    OP_SOFTMAX, "block_moe_softmax", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    size0(),
                    NnSoftmaxOpCodeConfig{});
                ff.addOp(
                    OP_MOE_GATE, "block_moe_gate2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeSBufferIndex),
                    size0(),
                    NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, NN_NO_PREPACK});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeLBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, NN_NO_PREPACK});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{moeLBufferIndex});
                if (moeDBufferIndex != moeDQBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex, NN_NO_PREPACK});
                ff.addOp(
                    OP_SCALE, "block_moe_scale", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size0(),
                    NnScaleOpCodeConfig{moeSBufferIndex});
                ff.addOp(
                    OP_MERGE_SUM, "block_moe_merge_sum", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size0(),
                    NnMergeSumOpCodeConfig{});
            } else {
                if (yBufferIndex != yqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_y3", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                // SharedPack: FFN 입력 yq 를 한 번만 재배치해 Gate·Up 이 공유한다.
                // (attention 의 pack_yq 와 같은 버퍼를 쓰지 않는다 — 그 사이에
                //  norm_1 이 yq 를 덮어쓰므로 레이어 안에서 값이 다르다.)
                if (yPackOn) {
                    ff.addOp(
                        OP_PACK_Q80X4, "block_pack_yq2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, yPackBufferIndex),
                        size0(),
                        NnPackQ80x4OpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size2D(h->weightType, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex,
                        yPackOn ? yPackBufferIndex : NN_NO_PREPACK});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, lBufferIndex),
                    size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex,
                        yPackOn ? yPackBufferIndex : NN_NO_PREPACK});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{lBufferIndex});
                if (dBufferIndex != dqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                // SharedPack-SDOT: Q80 활성화를 block_q8_0x4 로 **한 번만** 재배치한다.
                // 네 스레드가 batch-row group 을 나눠 공유 버퍼에 병렬로 쓰고,
                // executor 의 op 경계가 barrier 를 제공한다.
                if (sharedPackOn) {
                    ff.addOp(
                        OP_PACK_Q80X4, "block_pack_dq", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dPackBufferIndex),
                        size0(),
                        NnPackQ80x4OpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex,
                        sharedPackOn ? dPackBufferIndex : NN_NO_PREPACK});
            }
            ff.addOp(
                OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);
            // NOTE: 여기에 SYNC_CP_LOGITS 를 걸어 마지막 행 은닉 상태를 옮기려 했으나
            // 되돌렸다. 이 지점의 xPipe 는 살아 있는 잔차가 아니다 — 송신/수신 양쪽에서
            // maxabs=0 이었고, CP 를 끈 SP2 까지 깨졌다(원래는 정상 동작).
            // 잔차는 레이어 사이에 노드 로컬 버퍼에 있고 xPipe 는 그 시점에 비어 있다.
            // 로짓 회수는 실제 잔차가 있는 버퍼를 찾아 다시 설계해야 한다.

            if (ownAtt)
                nodeBuilder.addSegment(att.build());
            if (ownFf)
                nodeBuilder.addSegment(ff.build());
        }

        if (nodePlacement.ppRank < topology.ppSize - 1) {
            NnSegmentConfigBuilder bridge;
            bridge.addOp(
                OP_MERGE_ADD, "stage_bridge_merge_add", 0,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            bridge.addOp(
                OP_CAST, "stage_bridge_cast_x", 0,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            nodeBuilder.addSegment(bridge.build());
        }

        // Final segment: only for the last PP stage
        if (nodePlacement.ppRank == topology.ppSize - 1) {
            NnSegmentConfigBuilder end;
            end.addOp(
                OP_MERGE_ADD, "final_merge_add", 0,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            end.addOp(
                OP_INV_RMS, "final_norm_pre", 0,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            end.addOp(
                OP_RMS_NORM, "final_norm", 0,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                end.addOp(
                    OP_CAST, "final_cast_y", 0,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            if (fusedLmHeadArgmax) {
                end.addOp(
                    OP_MATMUL_ARGMAX, "final_matmul_logits", 0,
                    pointerBatchedSliceConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, argmaxPartialBufferIndex),
                    size2D(h->weightType, n.wclsSlice.n0, n.wclsSlice.d),
                    NnMatmulOpConfig{0u, 0u, 0u, NN_NO_PREPACK});
                end.addOp(
                    OP_ARGMAX_REDUCE, "final_reduce_argmax", 0,
                    pointerBatchConfig(SRC_BUFFER, argmaxPartialBufferIndex),
                    pointerBatchConfig(SRC_PIPE, n.logitsPipeIndex),
                    size0(),
                    NnArgmaxReduceOpCodeConfig{64});
            } else {
                end.addOp(
                    OP_MATMUL, "final_matmul_logits", 0,
                    pointerBatchedSliceConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_PIPE, n.logitsPipeIndex),
                    size2D(h->weightType, n.wclsSlice.n0, n.wclsSlice.d),
                    NnMatmulOpConfig{0u, 0u, 0u, NN_NO_PREPACK});
            }
            end.addSync(n.logitsPipeIndex, SYNC_NODE_SLICES);

            nodeBuilder.addSegment(end.build());
        }
        NnNodeConfig nodeConfig = nodeBuilder.build();
        nodeConfig.ppRank = nodePlacement.ppRank;
        nodeConfig.tpRank = nodePlacement.tpRank;
        nodeConfig.tpGroupStart = nodePlacement.tpGroupStart;
        nodeConfig.tpGroupEnd = nodePlacement.tpGroupEnd;
        nodeConfig.spRank = nodePlacement.spRank;
        nodeConfig.spGroupStart = nodePlacement.spGroupStart;
        nodeConfig.spGroupEnd = nodePlacement.spGroupEnd;
        nodeConfig.positionPipeIndex = n.positionPipeIndex;
        nodeConfig.tokenPipeIndex = n.tokenPipeIndex;
        nodeConfig.xPipeIndex = n.xPipeIndex;
        nodeConfig.logitsPipeIndex = n.logitsPipeIndex;
        n.nodeConfigs[nodeIndex] = nodeConfig;
    }
    return n;
}

void releaseLlmNet(LlmNet *net) {
    for (NnUint nodeIndex = 0u; nodeIndex < net->netConfig.nNodes; nodeIndex++)
        releaseNodeConfig(&net->nodeConfigs[nodeIndex]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1u);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
#endif

    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint layerIndex = 0u; layerIndex < net->header->nLayers; layerIndex++) {
        b += loader->loadRowMatmulSlices("block_matmul_q", layerIndex, 0u, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", layerIndex, 0u, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", layerIndex, 0u, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", layerIndex, 0u, &net->woSlice, b);

        if (net->header->nExperts > 0u) {
            b += loader->loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
            for (NnUint expertIndex = 0u; expertIndex < net->header->nExperts; expertIndex++) {
                b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, expertIndex, &net->w1Slice, b);
                b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, expertIndex, &net->w2Slice, b);
                b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, expertIndex, &net->w3Slice, b);
            }
        } else {
            b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, 0u, &net->w1Slice, b);
            b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, 0u, &net->w2Slice, b);
            b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, 0u, &net->w3Slice, b);
        }

        if (net->header->archType == QWEN3 || net->header->archType == QWEN3_MOE) {
            b += loader->loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += loader->loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
        }

        b += loader->loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += loader->loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);

        if (timer.elapsedMiliseconds() > 10000)
            printf("💿 Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
    }

    b += loader->loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
    // final_matmul_logits belongs to the last PP stage (ppSize - 1), not layer 0
    NnUint lastPpStage = net->netConfig.nNodes > 1 ? (net->netConfig.nNodes / net->nodeConfigs[0].tpGroupEnd) - 1 : 0;
    // tpGroupEnd for root is tpSize, so lastPpStage = ppSize - 1
    b += loader->loadColMatmulSlices("final_matmul_logits", 0u, 0u, &net->wclsSlice, b, lastPpStage > 0 ? lastPpStage : UINT_MAX);

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0u)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");

    loader->finish();
}
