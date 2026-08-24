#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include "nn-quants.hpp"

// primitives

typedef struct {
    NnFloatType floatType;
    NnUint z;
    NnUint y;
    NnUint x;
    NnSize length;
    NnSize nBytes;
    NnSize nBytesXY;
} NnSize3D;

// slices

typedef struct {
    NnUint kvDim0;
    NnUint localSeqLen;  // 이 SP rank가 담당하는 시퀀스 길이 (seqLen / spSize)
    NnUint localSeqStart; // 이 SP rank의 시퀀스 시작 위치 (spRank * localSeqLen)
    NnSize3D keySize;
    NnSize3D valueSize;
} NnKvCacheSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint d0;
    NnUint n;
    NnSize3D size;
    NnSize3D sliceSize;
} NnRowMatmulSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint n;
    NnUint n0;
    NnUint d;
    NnSize3D size;
    NnSize3D sliceSize;
} NnColMatmulSlice;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimEnd;
    NnUint qShift;
    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float ropeTheta;
    NnSize3D cacheSize;
} NnRopeSlice;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSlice;

// base enums

enum NnOpCode {
    OP_MERGE_ADD,
    OP_MERGE_SUM,
    OP_EMBEDDING,
    OP_INV_RMS,
    OP_RMS_NORM,
    OP_MATMUL,
    OP_ROPE,
    OP_MULTIHEAD_ATT,
    OP_GELU,
    OP_SILU,
    OP_MUL,
    OP_SCALE,
    OP_CAST,
    OP_REPEAT_Z,
    OP_SHIFT,
    OP_SOFTMAX,
    OP_MATMUL_ARGMAX,
    OP_ARGMAX_REDUCE,
    OP_MOE_GATE,
    // 토큰 가지치기 + 압축. enum 끝에 추가해야 기존 op 코드 번호가 밀리지 않는다
    // (워커로 직렬화되는 값이다).
    OP_PRUNE_TOKENS,
    // SharedPack-SDOT (research/18 §6d). Q80 활성화를 block_q8_0x4 로 **한 번만**
    // 재배치해 여러 스레드가 공유한다. 기존에는 matmul 안에서 스레드마다 전체를
    // 중복 변환했고, K 가 길면(Down, K=14336) 4스레드 확장이 1.68배 무너졌다.
    OP_PACK_Q80X4,
};

enum NnOpQuantType {
    // <input>_<weight>_<output>
    F32_F32_F32,
    F32_Q40_F32,
    F32_Q40_Q80,
    F32_F32_Q80,
    Q80_Q80_Q80,
    Q80_Q80_F32,
    Q80_Q40_F32,
    Q80_F32_F32,
};

#define N_OP_CODES (OP_MOE_GATE + 1)
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerSource {
    SRC_PIPE,
    SRC_BUFFER,
};

enum NnPointerType {
    PNTR_RAW,
    PNTR_BATCH,
    PNTR_BATCHED_SLICE
};

enum NnSyncType {
    SYNC_WITH_ROOT, // whole pipe to all nodes
    SYNC_NODE_SLICES, // all-reduce (sum) over full pipe buffer
    SYNC_NODE_SLICES_EXCEPT_ROOT, // only workers send slices to root, root does not send
    SYNC_SP_KV, // allgather K and V buffers across SP group (ring attention)
    // Ring CP: 마지막 토큰 행의 로짓을 그 행을 계산한 노드(spRank N-1)에서 root 로 보낸다.
    // CP 에서는 노드가 자기 토큰 블록만 계산하므로 마지막 행이 root 에 없다.
    SYNC_CP_LOGITS,
};

enum NnRopeType {
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
};

// base configs

typedef struct {
    char *name;
    NnSize3D size;
} NnPipeConfig;

typedef struct {
    char *name;
    NnSize3D size;
} NnBufferConfig;

typedef struct {
    NnPointerSource source;
    NnUint pointerIndex;
    NnPointerType type;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnUint index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize3D weightSize;
    NnByte *config;
    NnUint configSize;
} NnOpConfig;

typedef struct {
    NnUint pipeIndex;
} NnPreSyncConfig;

typedef struct {
    NnUint pipeIndex;
    NnSyncType syncType;
    // For SYNC_SP_KV only (zeros for other sync types):
    NnUint spKvKBufferIndex;
    NnUint spKvVBufferIndex;
    NnUint spKvLocalSeqStart;
    NnUint spKvLocalSeqLen;
    NnUint spKvDim0;
    NnUint spKvPositionPipeIdx;
} NnSyncConfig;

typedef struct  {
    NnUint nOps;
    NnOpConfig *ops;
    NnUint nSyncs;
    NnSyncConfig *syncs;
} NnSegmentConfig;

typedef struct {
    NnUint nBatches;
    NnUint nNodes;
    NnUint nPipes;
    NnPipeConfig *pipes;
    NnUint nPreSyncs;
    NnPreSyncConfig *preSyncs;
} NnNetConfig;

typedef struct {
    NnUint nodeIndex;
    NnUint ppRank;
    NnUint tpRank;
    NnUint tpGroupStart;
    NnUint tpGroupEnd;
    NnUint spRank;
    NnUint spGroupStart;
    NnUint spGroupEnd;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnUint nBuffers;
    NnBufferConfig *buffers;
    NnUint nSegments;
    NnSegmentConfig *segments;
} NnNodeConfig;

// op configs

typedef struct {
    // empty
} NnEmbeddingOpConfig;

typedef struct {
    float epsilon;
    NnUint nColumns;
} NnInvRmsOpConfig;

typedef struct {
    NnUint invRmsBufferIndex;
    NnUint nColumns;
} NnRmsNormOpConfig;

typedef struct {
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint activeExpertIndexesBufferIndex;
    // SharedPack-SDOT (research/18 §6d).
    // OP_PACK_Q80X4 가 만든 공유 block_q8_0x4 버퍼의 인덱스.
    // NN_NO_PREPACK 이면 기존처럼 스레드마다 중복 변환한다(fallback 유지).
    NnUint prepackedBufferIndex;
} NnMatmulOpConfig;

// prepackedBufferIndex 가 이 값이면 공유 버퍼를 쓰지 않는다.
#define NN_NO_PREPACK 0xFFFFFFFFu

typedef struct {
    NnRopeType type;
    NnUint isQ; // Cannot use `bool` here due to GPU memory alignment
    NnUint positionPipeIndex;
    NnUint ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    NnRopeSlice slice;
} NnRopeOpConfig;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnUint nKvHeads;
    NnUint headDim;
    NnUint seqLen;
    NnUint qSliceD0;
    NnUint kvDim0;
    NnUint localSeqLen;   // SP: 이 노드가 담당하는 로컬 시퀀스 길이
    NnUint localSeqStart; // SP: 로컬 시퀀스 시작 위치
    NnUint spSize;        // SP 크기 (1 = SP 비활성화)
    NnUint positionPipeIndex;
    NnUint queryBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
    NnUint attBufferIndex;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    // empty
} NnMergeSumOpCodeConfig;

typedef struct {
    // empty
} NnSiluOpCodeConfig;

typedef struct {
    NnUint multiplierBufferIndex;
} NnMulOpCodeConfig;

typedef struct {
    NnUint scaleBufferIndex;
} NnScaleOpCodeConfig;

typedef struct {
    // empty
} NnCastOpCodeConfig;

// SharedPack-SDOT (research/18 §6d).
// Q80 활성화를 block_q8_0x4 로 재배치한다. 네 스레드가 batch-row group 을 나눠
// **하나의 공유 버퍼**에 쓰고, executor 의 op 경계가 barrier 역할을 한다.
typedef struct {
    // empty
} NnPackQ80x4OpCodeConfig;

typedef struct {
    // empty
} NnRepeatZOpCodeConfig;

typedef struct {
    NnUint indexPipeIndex;
    NnUint localSeqStart; // SP: only write pos if >= localSeqStart (0 = no guard)
    NnUint localSeqLen;   // SP: only write pos if < localSeqStart+localSeqLen (0 = write all)
} NnShiftOpCodeConfig;

typedef struct {
    NnUint positionPipeIndex;
} NnPruneTokensOpCodeConfig;

typedef struct {
    // empty
} NnSoftmaxOpCodeConfig;

typedef struct {
    NnUint k;
    NnUint normTopk;
    NnUint indexesBufferIndex;
} NnMoeGateOpCodeConfig;

typedef struct {
    NnUint maxThreads;
} NnArgmaxReduceOpCodeConfig;

// utility functions

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize3D size0();
NnSize3D size1D(NnFloatType floatType, NnUint x);
NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x);
NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x);
NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    void reset();
    NnUint elapsedMiliseconds();
    NnUint elapsedMicroseconds();
};

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes, NnUint spSize = 1, NnUint spRank = 0);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches);

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);

// rope

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache);

#endif
