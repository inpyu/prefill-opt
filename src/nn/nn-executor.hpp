#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <vector>
#include <stdexcept>
#include <condition_variable>
#include <mutex>
#include <thread>

class NnDeviceSegment {
public:
    virtual ~NnDeviceSegment() {};
    virtual void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) = 0;
    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
};

class NnDevice {
public:
    virtual NnUint maxNThreads() = 0;
    virtual ~NnDevice() {}
    virtual NnDeviceSegment *createSegment(NnUint segmentIndex) = 0;
};

class NnNodeSynchronizer {
public:
    virtual ~NnNodeSynchronizer() {};
    virtual void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) = 0;
    virtual void setDecodePhase(bool isDecodePhase);
};

class NnFakeNodeSynchronizer : public NnNodeSynchronizer {
public:
    ~NnFakeNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnNetExecution {
public:
    NnUint nThreads;
    NnUint nPipes;
    NnByte **pipes;
    NnByte **nodeBuffers; // CPU node buffers (set externally after CpuDevice init)
    NnUint batchSize;
    NnUint nBatches;
    NnNetExecution(NnUint nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnUint batchSize);
};

enum NnExecutorStepType {
    STEP_EXECUTE_OP,
    STEP_SYNC_NODES,
};

#define N_STEP_TYPES STEP_SYNC_NODES + 1

class NnExecutorDevice {
public:
    std::unique_ptr<NnDevice> device;
    int segmentFrom;
    int segmentTo;
    NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo);
};

typedef struct {
    NnExecutorStepType type;
    NnDeviceSegment *segment;
    NnUint arg0;
    NnOpConfig *opConfig;
} NnExecutorStep;

typedef struct {
    NnUint syncUs;
    NnUint normUs;
    NnUint attnUs;      // attention 계열 합계 (attnCoreUs 포함) — 기존 호출부 호환용
    NnUint attnCoreUs;  // EXP-1/H2: multihead_att·softmax 등 O(S^2) 항만
    NnUint ffnUs;
    NnUint lmHeadUs;
    NnUint otherUs;
} NnExecutorOpBreakdown;

typedef struct {
    NnUint nThreads;
    NnUint nSteps;
    NnExecutorStep *steps;
    NnNodeSynchronizer *synchronizer;
    std::atomic_uint epoch;
    std::atomic_uint currentStepIndex;
    std::atomic_uint doneThreadCount;
    std::atomic_uint doneRunThreadCount;
    std::atomic_bool isAlive;
    std::atomic_bool isShutdown;
    std::atomic_bool isRunDone;
    NnUint batchSize;
    Timer *timer;
    NnUint totalTime[N_STEP_TYPES];
    std::atomic_bool stepProfilingEnabled;
    NnUint *stepTimeUs;
    unsigned long long stepTimerStartUs;
    std::mutex mutex;
    std::condition_variable cv;
} NnExecutorContext;

typedef struct {
    NnUint threadIndex;
    NnExecutorContext *context;
} NnExecutorThread;

class NnExecutorException : public std::runtime_error {
public:
    NnExecutorException(const std::string message);
};

class NnExecutor {
private:
    NnNetExecution *netExecution;
    NnNodeConfig *nodeConfig;
    std::vector<std::unique_ptr<NnDeviceSegment>> segments;
    std::vector<NnExecutorStep> steps;
    std::vector<NnUint> stepTimeUs;
    std::vector<NnExecutorThread> threads;
    std::vector<std::thread> threadHandles;
    NnExecutorContext context;
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    void forward();
    void setDecodePhase(bool isDecodePhase);
    NnUint getTotalTime(NnExecutorStepType type);
    void setStepProfilingEnabled(bool enabled);
    bool getLastForwardOpBreakdown(NnExecutorOpBreakdown *out) const;

    // 직전 forward 의 **step 별** 시간을 그대로 내준다.
    //
    // 캘리브레이션에서 레이어 수 회귀를 없애기 위해 필요하다(research/16 §7.6).
    // 회귀는 L=1..4 를 서로 다른 시각·온도에서 재므로 시간 드리프트를 레이어 효과로
    // 오인한다. 한 번의 forward 에서 레이어별 op 시간을 직접 읽으면
    // 모든 레이어가 **같은 열·메모리 상태**에서 수집되어 그 교란이 사라진다.
    //
    // outName/outLayer/outUs 는 nSteps 길이로 채워진다. 반환은 채운 개수.
    // step profiling 이 꺼져 있으면 0.
    NnUint getLastForwardStepTimes(const char **outName, NnUint *outLayer,
                                   NnUint *outUs, NnUint maxOut) const;
};

#endif
