// DerivePP step tax 측정 (research/16-derivepp.md §7.6)
//
// 목적: production 대비 22 % 잔차 중 **executor step 경계**가 얼마인지 분리한다.
//
// 반드시 실제 `NnExecutor` 의 `STEP_EXECUTE_OP` 경로를 쓴다. 별도 barrier loop 를
// 재면 executor 의 atomic/yield 구조를 재현하지 못해 답이 달라진다.
//
//   T(K) = a + h·K       a = 고정 오버헤드, h = step 당 비용
//
// 잔차 계산(§7.6):
//   FFN 잔차 189 ms/chunk,  FFN 전용 step = 12 × 32 = 384
//   → 잔차 전체를 step 으로 설명하려면 492 us/step 이 필요하다.
//   h 가 그보다 훨씬 작으면 "순수 barrier 가 주원인" 가설은 기각된다.
//
// no-op step 은 1x1 CAST 다. 실제 연산량은 무시할 수준이므로 측정되는 것은
// step 진입/이탈 비용이다.
//
// build:
//   g++ -std=c++11 -O3 -mcpu=native prefill_bench/bench_steps.cpp \
//       nn-quants.o nn-core.o nn-executor.o nn-cpu.o nn-cpu-ops.o \
//       llamafile-sgemm.o nn-repack.o -o bench_steps -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include "../src/nn/nn-core.hpp"
#include "../src/nn/nn-config-builder.hpp"
#include "../src/nn/nn-executor.hpp"
#include "../src/nn/nn-cpu.hpp"

static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1] + v[v.size()/2]);
}

// K 개의 no-op step 을 가진 그래프를 만들어 실행 시간을 잰다.
static double measure(NnUint K, NnUint nThreads, NnUint nBatches, int reps, int inner) {
    NnNetConfigBuilder netB(1, nBatches);
    const NnUint pipeIndex = netB.addPipe("P", size2D(F_32, nBatches, 1));
    NnNetConfig netConfig = netB.build();

    NnNodeConfigBuilder nodeB(0);
    const NnUint bufIndex = nodeB.addBuffer("b", size2D(F_32, nBatches, 1));
    NnSegmentConfigBuilder segB;
    // 1 열짜리 CAST 를 K 개. 연산량은 무시할 수준이므로 step 진입/이탈 비용만 남는다.
    // pipe <-> buffer 를 번갈아 써서 의존성이 실제 그래프와 비슷하게 유지되도록 한다.
    for (NnUint i = 0; i < K; i++) {
        char name[32];
        std::snprintf(name, sizeof(name), "noop%u", i);
        if (i % 2u == 0u)
            segB.addOp(OP_CAST, name, 0,
                pointerBatchConfig(SRC_PIPE, pipeIndex),
                pointerBatchConfig(SRC_BUFFER, bufIndex),
                size0(), NnCastOpCodeConfig{});
        else
            segB.addOp(OP_CAST, name, 0,
                pointerBatchConfig(SRC_BUFFER, bufIndex),
                pointerBatchConfig(SRC_PIPE, pipeIndex),
                size0(), NnCastOpCodeConfig{});
    }
    if (K == 0u)   // 세그먼트에 op 가 최소 1개는 있어야 한다
        segB.addOp(OP_CAST, "noop0", 0,
            pointerBatchConfig(SRC_PIPE, pipeIndex),
            pointerBatchConfig(SRC_BUFFER, bufIndex),
            size0(), NnCastOpCodeConfig{});
    nodeB.addSegment(segB.build());
    NnNodeConfig nodeConfig = nodeB.build();

    std::vector<double> ts;
    {   // executor/device 가 config 를 참조하므로 release 보다 먼저 소멸해야 한다.
        NnNetExecution execution(nThreads, &netConfig);
        // NnExecutorDevice 는 unique_ptr<NnDevice> 로 **소유권을 가져간다**.
        // 스택 객체 주소를 넘기면 소멸 시 delete 를 시도해 죽는다.
        NnCpuDevice *device = new NnCpuDevice(&netConfig, &nodeConfig, &execution);
        std::vector<NnExecutorDevice> devices;
        devices.push_back(NnExecutorDevice(device, 0, -1));
        NnFakeNodeSynchronizer sync;
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &sync, false);

        execution.setBatchSize(nBatches);
        for (int r = 0; r < reps + 1; r++) {
            const double t0 = nowSec();
            for (int it = 0; it < inner; it++)
                executor.forward();
            const double dt = (nowSec() - t0) / inner;
            if (r > 0) ts.push_back(dt);
        }
    }
    // config 는 일부러 해제하지 않는다. executor/device 와 소유권이 얽혀 있어
    // 해제 순서를 맞춰도 double free 가 난다. 호출이 9회뿐이고 버퍼가 작아
    // 누수는 무시할 수준이다(벤치 전용).
    return median(ts);
}

int main(int argc, char **argv) {
    const NnUint nThreads = argc > 1 ? (NnUint)atoi(argv[1]) : 4;
    const NnUint nBatches = argc > 2 ? (NnUint)atoi(argv[2]) : 16;
    const int reps        = argc > 3 ? atoi(argv[3]) : 5;
    char host[256] = {0}; gethostname(host, sizeof(host)-1);
    printf("# host=%s threads=%u nBatches=%u\n", host, nThreads, nBatches);
    printf("# K\tsec\tus_per_step\n");

    // 실측 step 수에 맞춘 후보 (research/16 §7.6):
    //   384 = FFN 전용(12 op x 32 layer),  832 = 전체(26 x 32),  838 = 실측 total
    static const NnUint KS[] = {0, 2, 8, 32, 128, 384, 448, 832, 838};
    std::vector<std::pair<double,double>> pts;
    for (NnUint K : KS) {
        // 구간을 충분히 길게 유지한다. K 가 작을수록 반복을 늘린다.
        const int inner = K == 0 ? 20000 : std::max(200, (int)(400000 / (K + 1)));
        const double s = measure(K, nThreads, nBatches, reps, inner);
        printf("%u\t%.9f\t%.3f\n", K, s, K ? s * 1e6 / K : 0.0);
        fflush(stdout);
        pts.push_back({(double)K, s});
    }

    // T(K) = a + h*K 최소제곱
    double n = pts.size(), sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (auto &p : pts) { sx += p.first; sy += p.second; sxx += p.first*p.first; sxy += p.first*p.second; }
    const double h = (n*sxy - sx*sy) / (n*sxx - sx*sx);
    const double a = (sy - h*sx) / n;
    printf("\n# T(K) = a + h*K :  a = %.6f ms,  h = %.3f us/step\n", a*1e3, h*1e6);
    printf("# FFN 384 step -> %.2f ms/chunk   (잔차 189 ms 의 %.1f %%)\n",
           h*384*1e3, h*384*1e3/189.0*100.0);
    printf("# 전체 832 step -> %.2f ms/chunk  (잔차의 %.1f %%)\n",
           h*832*1e3, h*832*1e3/189.0*100.0);
    printf("# 잔차 전체 설명에 필요한 h = %.1f us/step (FFN 기준)\n", 189000.0/384.0);
    return 0;
}
