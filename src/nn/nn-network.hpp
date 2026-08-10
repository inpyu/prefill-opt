#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"
#include <climits>
#include <chrono>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

#define ROOT_SOCKET_INDEX 0

enum CollectiveType {
    COLLECTIVE_AUTO,
    COLLECTIVE_STAR,
    COLLECTIVE_RING,
};

// Network performance monitoring structures
struct NnNetworkMetrics {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    NnSize bytesTransferred;
    NnUint operationCount;
    std::string operationType;
    NnUint socketIndex;
    
    NnNetworkMetrics() : bytesTransferred(0), operationCount(0), socketIndex(0) {}
};

struct NnSocketPerformanceStats {
    double avgLatencyMs;
    double maxLatencyMs;
    double minLatencyMs;
    NnSize totalBytes;
    NnUint totalOperations;
    double bandwidthMbps;
    std::vector<double> recentLatencies; // Last 100 operations
    
    NnSocketPerformanceStats() : avgLatencyMs(0), maxLatencyMs(0), minLatencyMs(0), 
                                  totalBytes(0), totalOperations(0), bandwidthMbps(0) {}
};

struct NnTrafficBreakdown {
    NnSize kvTxBytes;
    NnSize kvRxBytes;
    NnSize activationTxBytes;
    NnSize activationRxBytes;
    NnUint kvMigrationViolations;

    NnTrafficBreakdown()
        : kvTxBytes(0), kvRxBytes(0), activationTxBytes(0), activationRxBytes(0), kvMigrationViolations(0) {}
};

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, NnSize size);
void readSocket(int socket, void* data, NnSize size);
int createServerSocket(int port);
void destroySocket(int serverSocket);

class NnConnectionSocketException : public std::runtime_error {
public:
    NnConnectionSocketException(const std::string message);
};

class NnTransferSocketException : public std::runtime_error {
public:
    int code;
    NnTransferSocketException(int code, const std::string message);
};

class NnSocket {
public:
    int fd;
    NnSocket();
    NnSocket(int fd);
    ~NnSocket();
    void assign(int fd);
    int release();
};

struct NnSocketIo {
    NnUint socketIndex;
    const void *data;
    NnSize size;
};

void nnNetworkSetCpSplit(bool enabled);
void nnNetworkSetPpLayerOffsets(const std::vector<NnUint> &offsets);

class NnNetwork {
private:
    int *sockets;
    NnSize *sentBytes;
    NnSize *recvBytes;
    NnSize *sentOperations;
    NnSize *recvOperations;
    NnSocketPerformanceStats *socketStats;
    std::vector<NnNetworkMetrics> recentMetrics;
    std::mutex metricsMutex;
    std::atomic<NnSize> kvTxBytes;
    std::atomic<NnSize> kvRxBytes;
    std::atomic<NnSize> activationTxBytes;
    std::atomic<NnSize> activationRxBytes;
    std::atomic<NnUint> kvMigrationViolations;

    void updateSocketStats(NnUint socketIndex, double latencyMs, NnSize bytes);

public:
    static std::unique_ptr<NnNetwork> serve(int port);
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);

    NnUint nSockets;

    NnNetwork(std::vector<NnSocket> *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);
    void writeMany(NnUint n, NnSocketIo *ios);
    void writeAll(void *data, NnSize size);
    void readMany(NnUint n, NnSocketIo *ios);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void addTaggedTraffic(bool isKv, NnSize txBytes, NnSize rxBytes);
    void noteKvMigrationViolation();
    void getTrafficBreakdown(NnTrafficBreakdown *stats);
    // EXP-1: 누적 wait/xfer 마이크로초를 읽고, 구간별 측정을 위해 리셋한다.
    void getSyncTimeBreakdown(unsigned long long *waitUs, unsigned long long *xferUs);
    void resetSyncTimeBreakdown();
    void resetStats();
    void printSocketTrafficSummary(NnUint socketIndex, const char *label) const;
    
    // Performance monitoring functions
    void enablePerformanceMonitoring(bool enabled);
    void printPerformanceReport();
    void printBottleneckAnalysis();
    NnSocketPerformanceStats* getSocketStats(NnUint socketIndex);
    bool isPerformanceMonitoringEnabled() const;
    void recordOperation(const std::string& operationType, NnUint socketIndex, NnSize bytes, 
                        std::chrono::high_resolution_clock::time_point start, 
                        std::chrono::high_resolution_clock::time_point end);
};

class NnNetworkNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    CollectiveType collectiveType;
    bool spPrefillOnly;
    std::atomic_bool decodePhase;
    bool strictKvAffinity;
    bool allowKvMigration;
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, CollectiveType collectiveType, bool spPrefillOnly, bool strictKvAffinity, bool allowKvMigration);
    ~NnNetworkNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
    void setDecodePhase(bool isDecodePhase) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnUint socketIndex, NnNetConfig *config);
    void writeNode(NnUint socketIndex, NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

class NnWorkerConfigReader {
private:
    NnNetwork *network;
public:
    NnWorkerConfigReader(NnNetwork *network);
    NnNetConfig readNet();
    NnNodeConfig readNode();
};

class NnRootWeightLoader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnUint nNodes;
    NnUint ppSize;
    NnUint nLayers;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes, NnUint ppSize = 1, NnUint nLayers = 0);
    ~NnRootWeightLoader();
    void writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight, NnUint overridePpRank = UINT_MAX);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnUint tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnUint size);
};

#endif
