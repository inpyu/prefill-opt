#ifndef NN_CPU_OPS_H
#define NN_CPU_OPS_H

#include "nn-core.hpp"

#define ASSERT_EQ(a, b) \
    if (a != b) { \
        printf("Assertion failed: %d != %d (%s:%d)\n", a, b, __FILE__, __LINE__); \
        exit(-1); \
    }

typedef struct {
    const char *name;
    // NnByte(uint8_t) 였다. 이 구조체에서 NnByte 는 바이트 포인터용 타입인데
    // 개수 필드에 잘못 쓰여, nBatches 가 256 에서 0 으로 돌았다:
    //   --n-batches 448  -> 448 & 255 = 192  -> "Assertion failed: 448 != 192"
    //   --n-batches 1024 -> 1024 & 255 = 0   -> "Assertion failed: 1024 != 0"
    // 즉 prefill 청크 폭이 255 에서 조용히 무너지고 있었다.
    NnUint nBatches;
    NnByte *bufferFlags;
    NnByte **buffers;
    NnBufferConfig *bufferConfigs;
    NnByte **pipes;
    NnPipeConfig *pipeConfigs;
    void *opConfig;

    NnByte **input;
    NnSize3D inputSize;
    bool hasInputContinuousMemory;

    NnByte **output;
    NnSize3D outputSize;
    bool hasOutputContinuousMemory;

    NnByte *weight;
    NnSize3D weightSize;

    // Q4_0 repack 경로 (research/02, research/03).
    // 가중치는 로드가 끝난 시점에 in-place 로 재배치되므로 추가 메모리는 없다.
    // loadedBytes 로 "이 op 의 가중치가 전부 들어왔는가"를 판단한다.
    NnSize loadedBytes;
    bool isRepacked;

    // prefill 에서는 로짓이 전혀 소비되지 않는다(prefill 은 위치 0..n-2 만 처리하고,
    // decode 첫 스텝이 마지막 입력 토큰을 처리해 첫 출력 로짓을 만든다).
    // 따라서 lm_head 는 prefill 동안 마지막 행만 계산하면 된다.
    bool isLmHead;

    // 캘리브레이션 덤프용 (research/07 깊이 분해 검증).
    // opConfig->index 는 레이어 번호다.
    NnUint layerIndex;
} NnCpuOpContext;

// prefill/decode 단계를 op 계층에 알린다. NnExecutor::setDecodePhase 에서 호출된다.
void nnCpuOpsSetDecodePhase(bool isDecodePhase);
void nnCpuOpsSetAttnFused(int mode);
void nnCpuOpsSetCpRange(NnUint cpSize, NnUint cpRank); // Ring CP: 노드별 토큰 행 윈도우
void nnReportVerifyPackSummary();                     // DLLAMA_VERIFY_PACK 요약
void nnCpuOpsReportAttPhase();                        // DLLAMA_ATT_PHASE — attention 내부 분해
void nnCpuOpsSetPrune(NnUint layerIndex, float keepRatio); // 토큰 가지치기 (research/10)
void nnCpuOpsResetActiveRows();
NnUint nnCpuOpsGetActiveRows(); // -1=auto(배치 폭으로), 0=off, 1=on
bool nnCpuOpsIsDecodePhase();

// 블록 병렬 시뮬레이션 (research/07 Phase A1).
//
// 단일 노드에서 attention 마스크만 블록 병렬과 동일하게 만들어, 분산 구현 전에
// 근사 정확도를 먼저 측정한다. prefill 에서만 적용되고 decode 는 전체 KV 를 본다
// (Star Attention 의 query phase 에 해당).
//
//   blockSize : 블록 하나의 토큰 수. 0 이면 비활성
//   anchorLen : 모든 블록이 함께 보는 앞부분 길이. 0 이면 anchor 없음
void nnCpuOpsSetBlockMask(NnUint blockSize, NnUint anchorLen);

typedef void (*NnCpuOpForwardInit)(NnCpuOpContext *context);
typedef void (*NnCpuOpForward)(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context);

void printCpuInstructionSet();
NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType);
NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType);

void softmax_F32(float *x, const NnUint size);

#endif

void nnCpuOpsReportAttSkipProbe();
