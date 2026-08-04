#include <cassert>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include "nn-executor.hpp"

static inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context);

#define DEFAULT_EXEC_STALL_LOG_MS 2000ul
#define DEFAULT_EXEC_STALL_TIMEOUT_MS 180000ul

static unsigned long readExecutorTimeoutEnvMs(const char *name, unsigned long fallbackMs) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallbackMs;

    char *endPtr = nullptr;
    unsigned long parsed = std::strtoul(value, &endPtr, 10);
    if (endPtr == nullptr || *endPtr != '\0' || parsed == 0)
        return fallbackMs;
    return parsed;
}

static inline unsigned long getExecutorStallLogMs() {
    static const unsigned long value = readExecutorTimeoutEnvMs("DLLAMA_EXEC_STALL_LOG_MS", DEFAULT_EXEC_STALL_LOG_MS);
    return value;
}

static inline unsigned long getExecutorStallTimeoutMs() {
    static const unsigned long raw = readExecutorTimeoutEnvMs("DLLAMA_EXEC_STALL_TIMEOUT_MS", DEFAULT_EXEC_STALL_TIMEOUT_MS);
    static const unsigned long minimum = getExecutorStallLogMs();
    return raw < minimum ? minimum : raw;
}

static inline const char *executorStepTypeToString(NnExecutorStepType type) {
    if (type == STEP_EXECUTE_OP) return "EXECUTE_OP";
    if (type == STEP_SYNC_NODES) return "SYNC_NODES";
    return "UNKNOWN";
}

static inline unsigned long long nowUsExec() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline bool opNameContains(const char *name, const char *needle) {
    return name != nullptr && std::strstr(name, needle) != nullptr;
}

void NnNodeSynchronizer::setDecodePhase(bool isDecodePhase) {
    (void)isDecodePhase;
}

void NnFakeNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    // Nothing
}

NnNetExecution::NnNetExecution(NnUint nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward
    this->nodeBuffers = nullptr; // Set externally after CpuDevice initialization

    pipes = new NnByte *[netConfig->nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = new NnByte[pipeConfig->size.nBytes];
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }
}

NnNetExecution::~NnNetExecution() {
    for (NnUint pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnUint batchSize) {
    assert(batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutorDevice::NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo) {
    this->device = std::unique_ptr<NnDevice>(device);
    this->segmentFrom = segmentFrom;
    this->segmentTo = segmentTo;
}

NnExecutorException::NnExecutorException(const std::string message)
    : std::runtime_error(message) 
{}

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *devices, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark)
    : segments(nodeConfig->nSegments), steps()
{
    NnUint maxNThreads = 0;
    for (NnExecutorDevice &d : *devices) {
        if (d.device->maxNThreads() > maxNThreads)
            maxNThreads = d.device->maxNThreads();
    }
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This configuration supports max " + std::to_string(maxNThreads) + " threads");

    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDevice *device = nullptr;
        for (NnExecutorDevice &d : *devices) {
            if (
                (d.segmentFrom == -1 && d.segmentTo == -1) ||
                (segmentIndex >= d.segmentFrom && segmentIndex <= d.segmentTo)
            ) {
                device = d.device.get();
                break;
            }
        }
        if (device == nullptr)
            throw std::invalid_argument("Cannot locate device for segment " + std::to_string(segmentIndex));

        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex] });
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr });
    }

    steps.shrink_to_fit();
    stepTimeUs.resize(steps.size(), 0u);

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.nSteps = (NnUint)steps.size();
    context.steps = steps.data();
    context.epoch.store(0);
    context.currentStepIndex.store(0);
    context.doneThreadCount.store(0);
    context.doneRunThreadCount.store(0);
    context.isAlive.store(true);
    context.isShutdown.store(false);
    context.isRunDone.store(true);
    if (benchmark)
        context.timer = new Timer();
    else
        context.timer = nullptr;
    context.stepProfilingEnabled.store(false);
    context.stepTimeUs = stepTimeUs.data();
    context.stepTimerStartUs = 0ull;

    threads.resize(netExecution->nThreads);
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        threads[threadIndex].threadIndex = threadIndex;
        threads[threadIndex].context = &context;
    }

    threadHandles.reserve(netExecution->nThreads);
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        threadHandles.emplace_back([this, threadIndex]() {
            NnExecutorThread *thread = &this->threads[threadIndex];
            NnExecutorContext *ctx = thread->context;
            NnUint localEpoch = 0;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(ctx->mutex);
                    ctx->cv.wait(lock, [ctx, localEpoch]() {
                        return ctx->isShutdown.load() || ctx->epoch.load() != localEpoch;
                    });
                }

                if (ctx->isShutdown.load())
                    break;

                localEpoch = ctx->epoch.load();

                NnUint nThreads = ctx->nThreads;
                NnUint doneCount = nThreads - 1;

                while (ctx->isAlive.load()) {
                    const NnUint currentStepIndex = ctx->currentStepIndex.load();
                    if (currentStepIndex == ctx->nSteps)
                        break;

                    NnExecutorStep *step = &ctx->steps[currentStepIndex];
                    try {
                        executeStep(step, nThreads, thread, ctx);
                    } catch (const std::runtime_error &e) {
                        ctx->isAlive.store(false);
                        printf("🚨 Execution error: %s\n", e.what());
                        ctx->cv.notify_all();
                        break;
                    }

                    NnUint currentCount = ctx->doneThreadCount.fetch_add(1);
                    if (currentCount == doneCount) {
                        const NnUint finishedStepIndex = currentStepIndex;
                        if (ctx->stepProfilingEnabled.load()) {
                            const unsigned long long tNow = nowUsExec();
                            const unsigned long long tStart = ctx->stepTimerStartUs;
                            ctx->stepTimeUs[finishedStepIndex] = tStart > 0ull && tNow >= tStart
                                ? (NnUint)(tNow - tStart)
                                : 0u;
                            ctx->stepTimerStartUs = tNow;
                        }
                        if (ctx->timer != nullptr) {
                            NnUint time = ctx->timer->elapsedMicroseconds();
                            ctx->totalTime[step->type] += time;
                            ctx->timer->reset();
                        }

                        ctx->doneThreadCount.store(0);
                        ctx->currentStepIndex.fetch_add(1);
                    } else {
                        while (
                            ctx->currentStepIndex.load() == currentStepIndex &&
                            ctx->isAlive.load() &&
                            !ctx->isShutdown.load() &&
                            ctx->epoch.load() == localEpoch
                        ) {
                            std::this_thread::yield();
                        }
                    }
                }

                NnUint runCount = ctx->doneRunThreadCount.fetch_add(1);
                if (runCount == doneCount) {
                    ctx->isRunDone.store(true);
                    ctx->cv.notify_all();
                }
            }
        });
    }
}

NnExecutor::~NnExecutor() {
    if (context.timer != nullptr)
        delete context.timer;

    context.isShutdown.store(true);
    context.epoch.fetch_add(1);
    context.cv.notify_all();
    for (std::thread &t : threadHandles) {
        if (t.joinable())
            t.join();
    }
}

void NnExecutor::loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnUint i = 0; i < segmentConfig->nOps; i++) {
            NnOpConfig *opConfig = &segmentConfig->ops[i];
            if (opConfig->index == opIndex && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(i, offset, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

static inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else {
        throw std::invalid_argument("Unsupported step type");
    }
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    {
        std::lock_guard<std::mutex> lock(context.mutex);
        context.isAlive.store(true);
        context.currentStepIndex.store(0);
        context.doneThreadCount.store(0);
        context.doneRunThreadCount.store(0);
        context.isRunDone.store(false);
        context.batchSize = netExecution->batchSize;

        if (context.timer != nullptr) {
            std::memset(context.totalTime, 0, sizeof(context.totalTime));
            context.timer->reset();
        }
        if (context.stepProfilingEnabled.load()) {
            std::memset(stepTimeUs.data(), 0, sizeof(NnUint) * stepTimeUs.size());
            context.stepTimerStartUs = nowUsExec();
        }

        context.epoch.fetch_add(1);
    }
    context.cv.notify_all();

    const unsigned long stallLogMs = getExecutorStallLogMs();
    const unsigned long stallTimeoutMs = getExecutorStallTimeoutMs();
    auto lastProgressTime = std::chrono::steady_clock::now();
    auto lastLogTime = lastProgressTime;

    std::unique_lock<std::mutex> lock(context.mutex);
    NnUint observedStepIndex = context.currentStepIndex.load();
    while (
        !context.isShutdown.load() &&
        !context.isRunDone.load() &&
        context.isAlive.load()
    ) {
        bool isDone = context.cv.wait_for(lock, std::chrono::milliseconds(50), [this]() {
            return this->context.isShutdown.load() ||
                   this->context.isRunDone.load() ||
                   !this->context.isAlive.load();
        });
        if (isDone)
            break;

        NnUint currentStepIndex = context.currentStepIndex.load();
        if (currentStepIndex != observedStepIndex) {
            observedStepIndex = currentStepIndex;
            lastProgressTime = std::chrono::steady_clock::now();
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        long long stallMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime).count();
        long long sinceLogMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLogTime).count();

        if ((unsigned long)stallMs >= stallLogMs && (unsigned long)sinceLogMs >= stallLogMs) {
            const char *stepType = "DONE";
            const char *stepName = "-";
            if (currentStepIndex < context.nSteps) {
                NnExecutorStep *step = &context.steps[currentStepIndex];
                stepType = executorStepTypeToString(step->type);
                if (step->type == STEP_EXECUTE_OP && step->opConfig != nullptr && step->opConfig->name != nullptr)
                    stepName = step->opConfig->name;
            }

            printf("⏳ [EXEC_STALL] step=%u/%u type=%s op=%s stalled=%lldms doneThreads=%u/%u\n",
                currentStepIndex,
                context.nSteps,
                stepType,
                stepName,
                stallMs,
                context.doneThreadCount.load(),
                context.nThreads);
            fflush(stdout);
            lastLogTime = now;
        }

        if ((unsigned long)stallMs >= stallTimeoutMs) {
            const char *stepType = "DONE";
            const char *stepName = "-";
            if (currentStepIndex < context.nSteps) {
                NnExecutorStep *step = &context.steps[currentStepIndex];
                stepType = executorStepTypeToString(step->type);
                if (step->type == STEP_EXECUTE_OP && step->opConfig != nullptr && step->opConfig->name != nullptr)
                    stepName = step->opConfig->name;
            }

            printf("🚨 [EXEC_TIMEOUT] step=%u/%u type=%s op=%s stalled=%lldms (timeout=%lums)\n",
                currentStepIndex,
                context.nSteps,
                stepType,
                stepName,
                stallMs,
                stallTimeoutMs);
            fflush(stdout);

            context.isAlive.store(false);
            context.isRunDone.store(true);
            context.epoch.fetch_add(1);
            context.cv.notify_all();
            break;
        }
    }
    lock.unlock();

    if (!context.isAlive.load())
        throw NnExecutorException("Execution failed in one of the threads");
}

void NnExecutor::setDecodePhase(bool isDecodePhase) {
    context.synchronizer->setDecodePhase(isDecodePhase);
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}

void NnExecutor::setStepProfilingEnabled(bool enabled) {
    context.stepProfilingEnabled.store(enabled);
}

bool NnExecutor::getLastForwardOpBreakdown(NnExecutorOpBreakdown *out) const {
    if (out == nullptr || !context.stepProfilingEnabled.load())
        return false;
    std::memset(out, 0, sizeof(NnExecutorOpBreakdown));
    for (NnUint i = 0; i < context.nSteps; i++) {
        const NnExecutorStep &step = steps[i];
        const NnUint us = stepTimeUs[i];
        if (us == 0u)
            continue;
        if (step.type == STEP_SYNC_NODES) {
            out->syncUs += us;
            continue;
        }
        const char *name = (step.opConfig != nullptr) ? step.opConfig->name : nullptr;
        if (opNameContains(name, "final_matmul_logits")) {
            out->lmHeadUs += us;
            continue;
        }
        if (opNameContains(name, "norm")) {
            out->normUs += us;
            continue;
        }
        if (
            opNameContains(name, "moe") ||
            opNameContains(name, "_w1") ||
            opNameContains(name, "_w2") ||
            opNameContains(name, "_w3") ||
            opNameContains(name, "silu")
        ) {
            out->ffnUs += us;
            continue;
        }
        // EXP-1/H2: attention score 경로(O(S^2))를 projection 경로(O(S·h^2))와 분리한다.
        // 짧은 프롬프트에서 O(S^2) 항이 지배적이지 않다는 가설을 이 값으로 판정한다.
        if (
            opNameContains(name, "multihead_att") ||
            opNameContains(name, "softmax")
        ) {
            out->attnCoreUs += us;
            out->attnUs += us;
            continue;
        }
        if (
            opNameContains(name, "_q") ||
            opNameContains(name, "_k") ||
            opNameContains(name, "_v") ||
            opNameContains(name, "rope") ||
            opNameContains(name, "att") ||
            opNameContains(name, "_wo")
        ) {
            out->attnUs += us;
            continue;
        }
        out->otherUs += us;
    }
    return true;
}
