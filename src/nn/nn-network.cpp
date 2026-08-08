#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h> // For inet_addr and other functions
#include <windows.h>  // For SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>  // for getaddrinfo
#endif
#include "nn-network.hpp"
#include <cassert>
#ifndef _WIN32
#include <poll.h>
#endif
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <climits>
#include <atomic>

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571114
#define DEFAULT_NET_IO_CHUNK_SIZE 65536

#if defined(MSG_NOSIGNAL)
#define NN_SEND_FLAGS MSG_NOSIGNAL
#else
#define NN_SEND_FLAGS 0
#endif

#define DEFAULT_NET_STALL_LOG_MS 2000ul
#define DEFAULT_NET_STALL_TIMEOUT_MS 60000ul

static unsigned long readTimeoutEnvMs(const char *name, unsigned long fallbackMs) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallbackMs;

    char *endPtr = nullptr;
    unsigned long parsed = std::strtoul(value, &endPtr, 10);
    if (endPtr == nullptr || *endPtr != '\0' || parsed == 0)
        return fallbackMs;
    return parsed;
}

static inline unsigned long getNetStallLogMs() {
    static const unsigned long value = readTimeoutEnvMs("DLLAMA_NET_STALL_LOG_MS", DEFAULT_NET_STALL_LOG_MS);
    return value;
}

static inline unsigned long getNetStallTimeoutMs() {
    static const unsigned long raw = readTimeoutEnvMs("DLLAMA_NET_STALL_TIMEOUT_MS", DEFAULT_NET_STALL_TIMEOUT_MS);
    static const unsigned long minimum = getNetStallLogMs();
    return raw < minimum ? minimum : raw;
}

static inline NnSize getNetIoChunkSize() {
    static const NnSize value = []() -> NnSize {
        const char *env = std::getenv("DLLAMA_NET_IO_CHUNK_BYTES");
        if (env == nullptr || env[0] == '\0')
            return (NnSize)DEFAULT_NET_IO_CHUNK_SIZE;

        char *endPtr = nullptr;
        unsigned long parsed = std::strtoul(env, &endPtr, 10);
        if (endPtr == nullptr || *endPtr != '\0' || parsed == 0)
            return (NnSize)DEFAULT_NET_IO_CHUNK_SIZE;

        // Avoid tiny values that inflate syscall count, and cap to 1 MiB.
        const unsigned long minBytes = 4096ul;
        const unsigned long maxBytes = 1024ul * 1024ul;
        if (parsed < minBytes)
            parsed = minBytes;
        if (parsed > maxBytes)
            parsed = maxBytes;
        return (NnSize)parsed;
    }();
    return value;
}

// 논블로킹 소켓이 EAGAIN 을 반환했을 때의 대기.
//
// 원래는 무조건 1ms 를 잤다. 리눅스 타이머 해상도상 실제로는 1~2ms 이고,
// 하나의 집합통신에서 재시도가 수십 회 발생하므로 배리어 1회가 수십 ms 로 부풀었다.
// 실측(TP 2노드, S=447): 배리어 896회에 syncWait 39.3초 = 배리어당 43.9ms.
// 그중 회선 시간은 6.9ms 뿐이고 나머지 37ms 가 이 sleep 이었다.
//
// poll() 은 데이터가 준비되면 즉시 깨어나므로(마이크로초 단위) sleep 해상도에
// 묶이지 않는다. 동시에 스핀이 아니라 커널 대기라 CPU 코어를 뺏지 않는다 —
// 코어가 4개뿐인 SBC 에서는 이 점이 중요하다.
static inline void waitOnSocketRetry(int socket, bool forWrite) {
#ifndef _WIN32
    struct pollfd pfd;
    pfd.fd = socket;
    pfd.events = forWrite ? POLLOUT : POLLIN;
    pfd.revents = 0;
    // 타임아웃은 상위 루프의 stall 감지가 동작하도록 짧게 유지한다.
    poll(&pfd, 1, 1);
#else
    (void)socket; (void)forWrite;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
}

// 소켓을 특정할 수 없는 호출부(readMany/writeMany 는 여러 소켓을 동시에 본다)를 위한 폴백.
static inline void sleepOnSocketRetry() {
#ifndef _WIN32
    // 1ms 통째 sleep 대신 즉시 양보. 상위 루프가 곧바로 재시도한다.
    std::this_thread::yield();
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
}

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
    #endif
}

static inline bool shouldSkipTp1NoopSyncs() {
    const char *env = std::getenv("DLLAMA_KEEP_TP1_SYNCS");
    // Default: skip TP-size=1 collectives/syncs (no-op for PP-only runs).
    // Set DLLAMA_KEEP_TP1_SYNCS=1 to keep legacy behavior.
    return !(env != nullptr && std::strcmp(env, "1") == 0);
}

static inline void setNonBlocking(int socket, bool enabled) {
#ifdef _WIN32
    u_long mode = enabled ? 1 : 0;
    if (ioctlsocket(socket, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Error setting socket to non-blocking");
    }
#else
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
#endif
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void setQuickAck(int socket) {
#ifndef _WIN32
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
#endif
}

void setReuseAddr(int socket) {
    int opt = 1;
    #ifdef _WIN32
    int iresult = setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    if (iresult == SOCKET_ERROR) {
        closesocket(socket);
        throw std::runtime_error("setsockopt failed: " + std::to_string(WSAGetLastError()));
    }
    #else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(socket);
        throw std::runtime_error("setsockopt failed: " + std::string(strerror(errno)));
    }
    #endif
}

// EXP-1: sync 시간을 wait(peer 대기)과 xfer(실제 바이트 이동)로 나눈다.
//
// 두 계열의 전송 경로가 있고 둘 다 덮어야 한다:
//   - 집합통신(TP): NnNetwork::readMany/writeMany 가 recv/send 를 직접 호출
//   - 파이프라인(PP): NnNetwork::read/write -> readSocket/writeSocket (자유 함수)
// 후자는 인스턴스에 접근할 수 없으므로 파일 스코프 원자 카운터를 공유한다.
// (추론 프로세스당 NnNetwork 는 사실상 하나다)
static std::atomic<unsigned long long> gSyncWaitUs{0};
static std::atomic<unsigned long long> gSyncXferUs{0};

static inline unsigned long long nowUsNet() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void writeSocket(int socket, const void *data, NnSize size) {
    const unsigned long long tEnter = nowUsNet();
    unsigned long long waitUs = 0ull;
    auto lastProgressTime = std::chrono::steady_clock::now();
    auto lastLogTime = lastProgressTime;
    const unsigned long logTimeoutMs = getNetStallLogMs();
    const unsigned long hardTimeoutMs = getNetStallTimeoutMs();

    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, NN_SEND_FLAGS);
        if (s < 0) {
            if (isEagainError()) {
                auto now = std::chrono::steady_clock::now();
                long long stallMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime).count();
                long long sinceLogMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLogTime).count();

                if ((unsigned long)stallMs >= hardTimeoutMs) {
                    printf("🚨 [NET_TIMEOUT] writeSocket fd=%d stalled=%lldms pending=%zuB (timeout=%lums)\n",
                        socket,
                        stallMs,
                        (size_t)size,
                        hardTimeoutMs);
                    fflush(stdout);
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE,
                        "writeSocket timeout after " + std::to_string(stallMs) + "ms");
                }

                if ((unsigned long)stallMs >= logTimeoutMs && (unsigned long)sinceLogMs >= logTimeoutMs) {
                    printf("⏳ [NET_STALL] writeSocket fd=%d stalled=%lldms pending=%zuB\n",
                        socket,
                        stallMs,
                        (size_t)size);
                    fflush(stdout);
                    lastLogTime = now;
                }
                {
                    const unsigned long long tSleep0 = nowUsNet();
                    waitOnSocketRetry(socket, true);
                    waitUs += nowUsNet() - tSleep0;
                }
                continue;
            }
            const int err = SOCKET_LAST_ERRCODE;
            std::string msg = "Error writing to socket fd=" + std::to_string(socket) +
                " pending=" + std::to_string((size_t)size) +
                " err=" + std::to_string(err) +
                " (" + std::string(strerror(err)) + ")";
            throw NnTransferSocketException(err, msg);
        } else if (s == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
        lastProgressTime = std::chrono::steady_clock::now();
    }
    {
        const unsigned long long total = nowUsNet() - tEnter;
        gSyncWaitUs.fetch_add(waitUs, std::memory_order_relaxed);
        gSyncXferUs.fetch_add(total > waitUs ? total - waitUs : 0ull, std::memory_order_relaxed);
    }
}

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    const unsigned long long tEnter = nowUsNet();
    unsigned long long waitUs = 0ull;
    auto lastProgressTime = std::chrono::steady_clock::now();
    auto lastLogTime = lastProgressTime;
    const unsigned long logTimeoutMs = getNetStallLogMs();
    const unsigned long hardTimeoutMs = getNetStallTimeoutMs();

    NnSize s = size;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
                auto now = std::chrono::steady_clock::now();
                long long stallMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime).count();
                long long sinceLogMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLogTime).count();

                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }

                if ((unsigned long)stallMs >= hardTimeoutMs) {
                    printf("🚨 [NET_TIMEOUT] tryReadSocket fd=%d stalled=%lldms pending=%zuB (timeout=%lums)\n",
                        socket,
                        stallMs,
                        (size_t)s,
                        hardTimeoutMs);
                    fflush(stdout);
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE,
                        "tryReadSocket timeout after " + std::to_string(stallMs) + "ms");
                }

                if ((unsigned long)stallMs >= logTimeoutMs && (unsigned long)sinceLogMs >= logTimeoutMs) {
                    printf("⏳ [NET_STALL] tryReadSocket fd=%d stalled=%lldms pending=%zuB\n",
                        socket,
                        stallMs,
                        (size_t)s);
                    fflush(stdout);
                    lastLogTime = now;
                }
                {
                    const unsigned long long tSleep0 = nowUsNet();
                    waitOnSocketRetry(socket, false);
                    waitUs += nowUsNet() - tSleep0;
                }
                continue;
            }
            throw NnTransferSocketException(0, "Error reading from socket");
        } else if (r == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
        lastProgressTime = std::chrono::steady_clock::now();
    }
    {
        const unsigned long long total = nowUsNet() - tEnter;
        gSyncWaitUs.fetch_add(waitUs, std::memory_order_relaxed);
        gSyncXferUs.fetch_add(total > waitUs ? total - waitUs : 0ull, std::memory_order_relaxed);
    }
    return true;
}

void readSocket(int socket, void *data, NnSize size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnUint packet = ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static inline int connectSocket(char *host, int port) {
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[11];
    snprintf(portStr, sizeof(portStr), "%d", port);

    int addrinfoError = getaddrinfo(host, portStr, &hints, &addr);
    if (addrinfoError != 0 || addr == NULL) {
        printf("Cannot resolve target %s (%s)\n", host, gai_strerror(addrinfoError));
        throw NnConnectionSocketException("Cannot resolve address");
    }

    int sock = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (sock < 0)
        throw std::runtime_error("Cannot create socket");

    // Set socket to non-blocking mode for timeout support
    #ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(sock, FIONBIO, &mode);
    #else
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    #endif

    int connectResult = ::connect(sock, addr->ai_addr, addr->ai_addrlen);
    
    #ifdef _WIN32
    if (connectResult == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err != WSAEWOULDBLOCK) {
            printf("Cannot connect to %s:%d (error: %d)\n", host, port, err);
            closesocket(sock);
            throw NnConnectionSocketException("Cannot connect");
        }
    #else
    if (connectResult < 0 && errno != EINPROGRESS) {
        printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        close(sock);
        throw NnConnectionSocketException("Cannot connect");
    #endif
    } else {
        // Connection succeeded immediately (unlikely but possible)
        #ifdef _WIN32
        u_long mode = 0;
        ioctlsocket(sock, FIONBIO, &mode);
        #else
        flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
        #endif
        setNoDelay(sock);
        setQuickAck(sock);
        freeaddrinfo(addr);
        return sock;
    }

    // Wait for connection with timeout (30 seconds)
    fd_set writefds;
    fd_set exceptfds;
    struct timeval timeout;
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;

    FD_ZERO(&writefds);
    FD_ZERO(&exceptfds);
    FD_SET(sock, &writefds);
    FD_SET(sock, &exceptfds);

    int selectResult = select(sock + 1, NULL, &writefds, &exceptfds, &timeout);
    
    if (selectResult == 0) {
        printf("Connection timeout after 30 seconds connecting to %s:%d\n", host, port);
        printf("\n");
        printf("💡 Troubleshooting steps:\n");
        printf("   1. Verify worker is running: ssh %s 'ps aux | grep dllama'\n", host);
        printf("   2. Check port is listening: ssh %s 'netstat -tlnp | grep %d'\n", host, port);
        printf("   3. Test connectivity: ./diagnose_network.sh %s %d\n", host, port);
        printf("   4. Check firewall rules on both nodes\n");
        printf("\n");
        #ifdef _WIN32
        closesocket(sock);
        #else
        close(sock);
        #endif
        freeaddrinfo(addr);
        throw NnConnectionSocketException("Connection timeout");
    }
    
    if (selectResult < 0) {
        printf("Select error while connecting to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        #ifdef _WIN32
        closesocket(sock);
        #else
        close(sock);
        #endif
        freeaddrinfo(addr);
        throw NnConnectionSocketException("Select error");
    }

    // Check if connection succeeded or failed
    if (FD_ISSET(sock, &exceptfds)) {
        printf("Connection failed to %s:%d (exception on socket)\n", host, port);
        #ifdef _WIN32
        closesocket(sock);
        #else
        close(sock);
        #endif
        freeaddrinfo(addr);
        throw NnConnectionSocketException("Connection failed");
    }

    // Verify connection status
    int error = 0;
    socklen_t len = sizeof(error);
    #ifdef _WIN32
    getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &len);
    #else
    getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);
    #endif
    
    if (error != 0) {
        printf("Connection failed to %s:%d (SO_ERROR: %s)\n", host, port, strerror(error));
        #ifdef _WIN32
        closesocket(sock);
        #else
        close(sock);
        #endif
        freeaddrinfo(addr);
        throw NnConnectionSocketException("Connection failed");
    }

    // Set socket back to blocking mode
    #ifdef _WIN32
    u_long mode = 0;
    ioctlsocket(sock, FIONBIO, &mode);
    #else
    flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
    #endif

    setNoDelay(sock);
    setQuickAck(sock);
    freeaddrinfo(addr);
    return sock;
}
int createServerSocket(int port) {
    const char *host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    int serverSocket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");
    setReuseAddr(serverSocket);

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult;
    #ifdef _WIN32
    bindResult = bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (bindResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        closesocket(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::to_string(error));
    }
    #else
    bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        close(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::string(strerror(errno)));
    }
    #endif

    int listenResult = listen(serverSocket, SOMAXCONN);
    if (listenResult != 0) {
        #ifdef _WIN32
        closesocket(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::to_string(WSAGetLastError()));
        #else
        close(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::string(strerror(errno)));
        #endif
    }

    printf("Listening on %s:%d...\n", host, port);

    setNoDelay(serverSocket);
    setQuickAck(serverSocket);
    return serverSocket;
}

void destroySocket(int serverSocket) {
    shutdown(serverSocket, 2);
    #ifdef _WIN32
    closesocket(serverSocket);
    #else
    close(serverSocket);
    #endif
}

int acceptSocket(int serverSocket) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return clientSocket;
}

void initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(WSAGetLastError()));
    }
#endif
}

void cleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

NnConnectionSocketException::NnConnectionSocketException(const std::string message)
    : std::runtime_error(message)
{}

NnTransferSocketException::NnTransferSocketException(int code, const std::string message)
    : code(code), std::runtime_error(message)
{}

NnSocket::NnSocket() {
    this->fd = -1;
}

NnSocket::NnSocket(int fd) : NnSocket() {
    assign(fd);
}

NnSocket::~NnSocket() {
    if (this->fd >= 0)
        destroySocket(this->fd);
}

void NnSocket::assign(int fd) {
    assert(this->fd == -1);
    assert(fd >= 0);
    this->fd = fd;
}

int NnSocket::release() {
    assert(this->fd >= 0);
    int fd = this->fd;
    this->fd = -1;
    return fd;
}

std::unique_ptr<NnNetwork> NnNetwork::serve(int port) {
    NnSocket socketSocket(createServerSocket(port));
    printf("DEBUG: socketSocket.fd = %d\n", socketSocket.fd);

    NnUint nSockets;
    NnUint nodeIndex;
    int rootSocketFd = acceptSocket(socketSocket.fd);
    NnSocket rootSocket(rootSocketFd);
    printf("⭕ The root node has connected\n");

    NnUint nNodes;
    readSocket(rootSocketFd, &nNodes, sizeof(nNodes)); // receive total nodes
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocketFd, &nodeIndex, sizeof(nodeIndex)); // receive this worker's node index
    printf("⭕ NodeIndex: %d\n", nodeIndex);
    nSockets = nNodes - 1;

    std::vector<NnSocket> sockets(nSockets);
    sockets[0].assign(rootSocket.release());

    printf("⭕ Socket[0]: accepted root node\n");
    NnUint nWorkers = nNodes - 1; // exclude root
    NnUint nOtherWorkers = nWorkers - 1; // exclude this worker
    printf("⭕ Worker[%d]: expecting %d peer workers\n", nodeIndex, nOtherWorkers);
    std::vector<std::unique_ptr<char[]>> hosts(nOtherWorkers);
    std::vector<int> ports(nOtherWorkers);

    NnUint hostLen;
    for (NnUint i = 0; i < nOtherWorkers; i++) {
        readSocket(rootSocketFd, &hostLen, sizeof(hostLen));

        std::unique_ptr<char[]> host(new char[hostLen]);
        readSocket(rootSocketFd, host.get(), hostLen);
        hosts[i] = std::move(host);

        readSocket(rootSocketFd, &ports[i], sizeof(ports[i]));
    }

    printf("⭕ Worker[%d]: config received, sending initial ACK to root\n", nodeIndex);
    writeAckPacket(rootSocketFd);

    // We need to wait here until the root node will send a "root is ready" packet
    printf("⭕ Worker[%d]: waiting root-ready ACK\n", nodeIndex);
    readAckPacket(rootSocketFd);
    printf("⭕ Worker[%d]: root-ready ACK received\n", nodeIndex);

    for (NnUint i = 0; i < nOtherWorkers; i++) {
        char *host = hosts[i].get();
        int port = ports[i];
        // Calculate actual worker's node index from hosts array index
        NnUint otherWorkerIndex = (i < nodeIndex - 1) ? (i + 1) : (i + 2);
        NnUint socketIndex = otherWorkerIndex < nodeIndex ? otherWorkerIndex : (otherWorkerIndex - 1);
        printf("⭕ Worker[%d]: peer node=%d mapped to socketIndex=%d\n", nodeIndex, otherWorkerIndex, socketIndex);
        if (otherWorkerIndex < nodeIndex) {
            // Lower-indexed workers are already listening
            printf("⭕ Socket[%d]: wait for %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(acceptSocket(socketSocket.fd));
            printf("⭕ Socket[%d]: accepted\n", socketIndex);
        } else {
            // Higher-indexed workers: we connect to them
            printf("⭕ Socket[%d]: connecting to %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(connectSocket(host, port));
            printf("⭕ Socket[%d]: connected\n", socketIndex);
        }
    }

    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);
    NnUint nNodes = nSockets + 1; // +1 for root node

    std::set<std::string> seenEndpoints;
    for (NnUint i = 0; i < nSockets; i++) {
        std::string endpoint = std::string(hosts[i]) + ":" + std::to_string(ports[i]);
        if (!seenEndpoints.insert(endpoint).second) {
            throw std::runtime_error(
                "Duplicate worker endpoint detected: " + endpoint +
                ". Each worker must use a unique host:port (one worker process per endpoint).");
        }
    }

    printf("⭕ Root expects %d workers\n", nSockets);

    std::vector<NnSocket> sockets(nSockets);
    struct sockaddr_in addr;
    for (NnUint i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int fd = connectSocket(hosts[i], ports[i]);
        sockets[i].assign(fd);
        writeSocket(fd, &nNodes, sizeof(nNodes)); // send total nodes
        NnUint nodeIndex = i + 1; // worker node index (root is 0)
        writeSocket(fd, &nodeIndex, sizeof(nodeIndex)); // send worker's node index
        for (NnUint j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            NnUint hostLen = strlen(hosts[j]) + 1;
            writeSocket(fd, &hostLen, sizeof(hostLen));
            writeSocket(fd, hosts[j], hostLen);
            writeSocket(fd, &ports[j], sizeof(ports[j]));
        }
        printf("⭕ Socket[%d]: waiting initial ACK from worker\n", i);
        readAckPacket(fd);
        printf("⭕ Socket[%d]: connected\n", i);
    }
    printf("⭕ Root: broadcasting root-ready ACK to all workers\n");
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i].fd);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

NnNetwork::NnNetwork(std::vector<NnSocket> *sockets) {
    this->nSockets = sockets->size();
    this->sockets = new int[nSockets];
    for (NnUint i = 0; i < nSockets; i++)
        this->sockets[i] = sockets->at(i).release();
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
    this->sentOperations = new NnSize[nSockets];
    this->recvOperations = new NnSize[nSockets];
    this->socketStats = new NnSocketPerformanceStats[nSockets];
    this->kvTxBytes.store(0);
    this->kvRxBytes.store(0);
    this->activationTxBytes.store(0);
    this->activationRxBytes.store(0);
    this->kvMigrationViolations.store(0);
    for (NnUint i = 0; i < nSockets; i++) {
        this->sentBytes[i] = 0;
        this->recvBytes[i] = 0;
        this->sentOperations[i] = 0;
        this->recvOperations[i] = 0;
    }
}

NnNetwork::~NnNetwork() {
    delete[] sentBytes;
    delete[] recvBytes;
    delete[] sentOperations;
    delete[] recvOperations;
    delete[] socketStats;
    for (NnUint i = 0; i < nSockets; i++)
        destroySocket(sockets[i]);
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnUint i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(const NnUint socketIndex, const void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    const NnSize ioChunkSize = getNetIoChunkSize();
    for (NnSize chunk = 0; chunk < size; chunk += ioChunkSize) {
        NnSize chunkSize = chunk + ioChunkSize < size ? ioChunkSize : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
    sentBytes[socketIndex] += size;
    sentOperations[socketIndex]++;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("write", socketIndex, size, startTime, endTime);
}

void NnNetwork::read(const NnUint socketIndex, void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    const NnSize ioChunkSize = getNetIoChunkSize();
    for (NnSize chunk = 0; chunk < size; chunk += ioChunkSize) {
        NnSize chunkSize = chunk + ioChunkSize < size ? ioChunkSize : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
    recvBytes[socketIndex] += size;
    recvOperations[socketIndex]++;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("read", socketIndex, size, startTime, endTime);
}

void NnNetwork::writeAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

bool NnNetwork::tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes[socketIndex] += size;
        recvOperations[socketIndex]++;
        return true;
    }
    return false;
}

// readMany/writeMany 용: 아직 끝나지 않은 소켓 여러 개를 한 번에 기다린다.
// 개별 sleep/yield 대신 poll 로 커널 대기하면 데이터 도착 즉시 깨어나면서도
// 코어를 점유하지 않는다(SBC 는 코어가 4개뿐이라 스핀이 연산 스레드를 뺏는다).
static inline void waitOnSocketsRetry(const int *sockets, NnSocketIo *ios, NnUint n, bool forWrite) {
#ifndef _WIN32
    struct pollfd pfds[16];
    NnUint m = 0;
    for (NnUint i = 0; i < n && m < 16; i++) {
        if (ios[i].size == 0)
            continue;
        pfds[m].fd = sockets[ios[i].socketIndex];
        pfds[m].events = forWrite ? POLLOUT : POLLIN;
        pfds[m].revents = 0;
        m++;
    }
    if (m == 0)
        return;
    poll(pfds, m, 1);
#else
    (void)sockets; (void)ios; (void)n; (void)forWrite;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
}

void NnNetwork::writeMany(NnUint n, NnSocketIo *ios) {
    bool isWriting;
    auto lastProgressTime = std::chrono::steady_clock::now();
    auto lastLogTime = lastProgressTime;
    const unsigned long logTimeoutMs = getNetStallLogMs();
    const unsigned long hardTimeoutMs = getNetStallTimeoutMs();

    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
        sentOperations[io->socketIndex]++;
    }
    do {
        // EXP-1: readMany와 동일 규칙. 바이트가 움직인 패스는 xfer, 아니면 wait(수신측 back-pressure).
        auto passStart = std::chrono::high_resolution_clock::now();
        isWriting = false;
        bool hasProgress = false;
        NnSize pendingBytes = 0;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                pendingBytes += io->size;
                int socket = sockets[io->socketIndex];
                const NnSize ioChunkSize = getNetIoChunkSize();
                ssize_t chunkSize = io->size > ioChunkSize ? (ssize_t)ioChunkSize : (ssize_t)io->size;
                ssize_t s = send(socket, (const char*)io->data, chunkSize, NN_SEND_FLAGS);
                if (s < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
                hasProgress = true;
            }
        }

        {
            auto passEnd = std::chrono::high_resolution_clock::now();
            unsigned long long passUs = (unsigned long long)
                std::chrono::duration_cast<std::chrono::microseconds>(passEnd - passStart).count();
            if (hasProgress)
                gSyncXferUs.fetch_add(passUs, std::memory_order_relaxed);
            else if (isWriting)
                gSyncWaitUs.fetch_add(passUs, std::memory_order_relaxed);
        }

        if (isWriting && !hasProgress) {
            auto now = std::chrono::steady_clock::now();
            long long stallMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime).count();
            long long sinceLogMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLogTime).count();

            if ((unsigned long)stallMs >= hardTimeoutMs) {
                printf("🚨 [NET_TIMEOUT] writeMany stalled=%lldms pending=%zuB sockets=%u (timeout=%lums)\n",
                    stallMs,
                    (size_t)pendingBytes,
                    n,
                    hardTimeoutMs);
                fflush(stdout);
                throw NnTransferSocketException(SOCKET_LAST_ERRCODE,
                    "writeMany timeout after " + std::to_string(stallMs) + "ms");
            }

            if ((unsigned long)stallMs >= logTimeoutMs && (unsigned long)sinceLogMs >= logTimeoutMs) {
                printf("⏳ [NET_STALL] writeMany stalled=%lldms pending=%zuB sockets=%u\n",
                    stallMs,
                    (size_t)pendingBytes,
                    n);
                fflush(stdout);
                lastLogTime = now;
            }
            auto sleepStart = std::chrono::high_resolution_clock::now();
            waitOnSocketsRetry(sockets, ios, n, true);
            gSyncWaitUs.fetch_add((unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - sleepStart).count(), std::memory_order_relaxed);
        } else if (hasProgress) {
            lastProgressTime = std::chrono::steady_clock::now();
        }
    } while (isWriting);
}

void NnNetwork::writeAll(void *data, NnSize size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnUint n, NnSocketIo *ios) {
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastProgressTime = std::chrono::steady_clock::now();
    auto lastLogTime = lastProgressTime;
    const unsigned long logTimeoutMs = getNetStallLogMs();
    const unsigned long hardTimeoutMs = getNetStallTimeoutMs();
    
    bool isReading;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
        recvOperations[io->socketIndex]++;
        nBytes += io->size;
    }
    do {
        // EXP-1: 이 패스에서 바이트가 실제로 움직였는지에 따라 xfer/wait로 나눠 적산한다.
        auto passStart = std::chrono::high_resolution_clock::now();
        isReading = false;
        bool hasProgress = false;
        NnSize pendingBytes = 0;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                pendingBytes += io->size;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= r;
                io->data = (char*)io->data + r;
                hasProgress = true;
            }
        }

        {
            auto passEnd = std::chrono::high_resolution_clock::now();
            unsigned long long passUs = (unsigned long long)
                std::chrono::duration_cast<std::chrono::microseconds>(passEnd - passStart).count();
            if (hasProgress)
                gSyncXferUs.fetch_add(passUs, std::memory_order_relaxed);
            else if (isReading)
                gSyncWaitUs.fetch_add(passUs, std::memory_order_relaxed);
        }

        if (isReading && !hasProgress) {
            auto now = std::chrono::steady_clock::now();
            long long stallMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime).count();
            long long sinceLogMs = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLogTime).count();

            if ((unsigned long)stallMs >= hardTimeoutMs) {
                printf("🚨 [NET_TIMEOUT] readMany stalled=%lldms pending=%zuB sockets=%u (timeout=%lums)\n",
                    stallMs,
                    (size_t)pendingBytes,
                    n,
                    hardTimeoutMs);
                fflush(stdout);
                throw NnTransferSocketException(SOCKET_LAST_ERRCODE,
                    "readMany timeout after " + std::to_string(stallMs) + "ms");
            }

            if ((unsigned long)stallMs >= logTimeoutMs && (unsigned long)sinceLogMs >= logTimeoutMs) {
                printf("⏳ [NET_STALL] readMany stalled=%lldms pending=%zuB sockets=%u\n",
                    stallMs,
                    (size_t)pendingBytes,
                    n);
                fflush(stdout);
                lastLogTime = now;
            }
            // EXP-1: 재시도 대기도 peer 대기 시간이다.
            auto sleepStart = std::chrono::high_resolution_clock::now();
            waitOnSocketsRetry(sockets, ios, n, false);
            gSyncWaitUs.fetch_add((unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - sleepStart).count(), std::memory_order_relaxed);
        } else if (hasProgress) {
            lastProgressTime = std::chrono::steady_clock::now();
        }
    } while (isReading);

    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("readMany", 0, nBytes, startTime, endTime);
}

void NnNetwork::getStats(NnSize *sentBytes, NnSize *recvBytes) {
    *sentBytes = 0;
    *recvBytes = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        *sentBytes += this->sentBytes[i];
        *recvBytes += this->recvBytes[i];
    }
    resetStats();
}

void NnNetwork::addTaggedTraffic(bool isKv, NnSize txBytes, NnSize rxBytes) {
    if (isKv) {
        kvTxBytes.fetch_add(txBytes, std::memory_order_relaxed);
        kvRxBytes.fetch_add(rxBytes, std::memory_order_relaxed);
    } else {
        activationTxBytes.fetch_add(txBytes, std::memory_order_relaxed);
        activationRxBytes.fetch_add(rxBytes, std::memory_order_relaxed);
    }
}

void NnNetwork::noteKvMigrationViolation() {
    kvMigrationViolations.fetch_add(1, std::memory_order_relaxed);
}

void NnNetwork::getTrafficBreakdown(NnTrafficBreakdown *stats) {
    stats->kvTxBytes = kvTxBytes.load(std::memory_order_relaxed);
    stats->kvRxBytes = kvRxBytes.load(std::memory_order_relaxed);
    stats->activationTxBytes = activationTxBytes.load(std::memory_order_relaxed);
    stats->activationRxBytes = activationRxBytes.load(std::memory_order_relaxed);
    stats->kvMigrationViolations = kvMigrationViolations.load(std::memory_order_relaxed);
}

void NnNetwork::resetStats() {
    for (NnUint i = 0; i < nSockets; i++) {
        sentBytes[i] = 0;
        recvBytes[i] = 0;
        sentOperations[i] = 0;
        recvOperations[i] = 0;
    }
    kvTxBytes.store(0, std::memory_order_relaxed);
    kvRxBytes.store(0, std::memory_order_relaxed);
    activationTxBytes.store(0, std::memory_order_relaxed);
    activationRxBytes.store(0, std::memory_order_relaxed);
    kvMigrationViolations.store(0, std::memory_order_relaxed);
    // 주의: sync wait/xfer 카운터는 여기서 지우지 않는다.
    // getStats() 가 마지막에 resetStats() 를 호출하므로, 여기서 지우면
    // 호출 순서에 따라 getSyncTimeBreakdown() 이 0 을 읽게 된다(실제로 그랬다).
    // 이 카운터는 resetSyncTimeBreakdown() 으로만 초기화한다.
}

void NnNetwork::getSyncTimeBreakdown(unsigned long long *waitUs, unsigned long long *xferUs) {
    *waitUs = gSyncWaitUs.load(std::memory_order_relaxed);
    *xferUs = gSyncXferUs.load(std::memory_order_relaxed);
}

void NnNetwork::resetSyncTimeBreakdown() {
    gSyncWaitUs.store(0, std::memory_order_relaxed);
    gSyncXferUs.store(0, std::memory_order_relaxed);
}

void NnNetwork::printSocketTrafficSummary(NnUint socketIndex, const char *label) const {
    if (socketIndex >= nSockets) {
        // 종료 경로에서 이 분기가 무한 반복되어 로그가 14 GB 까지 자란 사례가 있다.
        // 원인(호출부 루프 경계)이 확정될 때까지 출력 자체를 제한한다.
        static std::atomic<int> invalidReports{0};
        const int n = invalidReports.fetch_add(1, std::memory_order_relaxed);
        if (n < 4) {
            printf("📦 [NET_TRAFFIC] invalid socket=%u for %s (nSockets=%u)\n",
                socketIndex,
                label == nullptr ? "unknown" : label,
                nSockets);
        } else if (n == 4) {
            printf("📦 [NET_TRAFFIC] (invalid socket 보고 억제됨 — 호출부 루프 경계 확인 필요)\n");
            fflush(stdout);
        }
        return;
    }

    const NnSize txBytes = sentBytes[socketIndex];
    const NnSize rxBytes = recvBytes[socketIndex];
    const NnSize txOps = sentOperations[socketIndex];
    const NnSize rxOps = recvOperations[socketIndex];
    const double avgTx = txOps > 0 ? (double)txBytes / (double)txOps : 0.0;
    const double avgRx = rxOps > 0 ? (double)rxBytes / (double)rxOps : 0.0;

    printf("📦 [NET_TRAFFIC] %s socket=%u tx_total=%zuB tx_count=%zu tx_avg=%.1fB rx_total=%zuB rx_count=%zu rx_avg=%.1fB\n",
        label == nullptr ? "socket" : label,
        socketIndex,
        (size_t)txBytes,
        (size_t)txOps,
        avgTx,
        (size_t)rxBytes,
        (size_t)rxOps,
        avgRx);
}

// Static member for performance monitoring state
static bool g_performanceMonitoringEnabled = false;

void NnNetwork::enablePerformanceMonitoring(bool enabled) {
    g_performanceMonitoringEnabled = enabled;
    if (enabled) {
        printf("📊 Network performance monitoring enabled\n");
    }
}

bool NnNetwork::isPerformanceMonitoringEnabled() const {
    return g_performanceMonitoringEnabled;
}

void NnNetwork::recordOperation(const std::string& operationType, NnUint socketIndex, NnSize bytes, 
                               std::chrono::high_resolution_clock::time_point start, 
                               std::chrono::high_resolution_clock::time_point end) {
    if (!g_performanceMonitoringEnabled) return;
    
    std::lock_guard<std::mutex> lock(metricsMutex);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double latencyMs = duration.count() / 1000.0;
    
    updateSocketStats(socketIndex, latencyMs, bytes);
    
    // Store recent metrics for analysis (thread-safe with size limit)
    if (socketIndex < nSockets) {  // Safety check
        NnNetworkMetrics metric;
        metric.startTime = start;
        metric.endTime = end;
        metric.bytesTransferred = bytes;
        metric.operationType = operationType;
        metric.socketIndex = socketIndex;
        
        // Limit metrics storage to prevent memory issues
        if (recentMetrics.size() < 500) {
            recentMetrics.push_back(metric);
        }
    }
}

void NnNetwork::updateSocketStats(NnUint socketIndex, double latencyMs, NnSize bytes) {
    if (socketIndex >= nSockets || !socketStats) return;
    
    NnSocketPerformanceStats& stats = socketStats[socketIndex];
    
    stats.totalOperations++;
    stats.totalBytes += bytes;
    
    // Update latency statistics
    if (stats.totalOperations == 1) {
        stats.minLatencyMs = stats.maxLatencyMs = latencyMs;
    } else {
        stats.minLatencyMs = std::min(stats.minLatencyMs, latencyMs);
        stats.maxLatencyMs = std::max(stats.maxLatencyMs, latencyMs);
    }
    
    // Update average latency
    stats.avgLatencyMs = (stats.avgLatencyMs * (stats.totalOperations - 1) + latencyMs) / stats.totalOperations;
    
    // Update recent latencies (limit size for memory safety)
    if (stats.recentLatencies.size() < 50) {
        stats.recentLatencies.push_back(latencyMs);
    }
    
    // Calculate bandwidth (MB/s)
    if (latencyMs > 0) {
        double bandwidthMBps = (bytes / (1024.0 * 1024.0)) / (latencyMs / 1000.0);
        stats.bandwidthMbps = bandwidthMBps * 8; // Convert to Mbps
    }
}

void NnNetwork::printPerformanceReport() {
    if (!g_performanceMonitoringEnabled) {
        printf("📊 Performance monitoring is disabled. Enable it first.\n");
        return;
    }

    std::lock_guard<std::mutex> lock(metricsMutex);
    
    printf("\n📊 === Network Performance Report ===\n");
    printf("Socket | Operations | Total MB | Avg Latency | Max Latency | Min Latency | Bandwidth\n");
    printf("-------|------------|----------|-------------|-------------|-------------|----------\n");
    
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketPerformanceStats& stats = socketStats[i];
        if (stats.totalOperations > 0) {
            printf("   %2d  |    %6d   |  %6.2f   |   %6.2f ms  |   %6.2f ms  |   %6.2f ms  |  %6.2f Mbps\n",
                   i, stats.totalOperations, stats.totalBytes / (1024.0 * 1024.0),
                   stats.avgLatencyMs, stats.maxLatencyMs, stats.minLatencyMs, stats.bandwidthMbps);
        }
    }
    printf("\n");
}

void NnNetwork::printBottleneckAnalysis() {
    if (!g_performanceMonitoringEnabled) {
        printf("📊 Performance monitoring is disabled. Enable it first.\n");
        return;
    }

    std::lock_guard<std::mutex> lock(metricsMutex);
    
    printf("\n🔍 === Network Bottleneck Analysis ===\n");
    
    // Find the slowest socket
    NnUint slowestSocket = 0;
    double maxAvgLatency = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        if (socketStats[i].totalOperations > 0 && socketStats[i].avgLatencyMs > maxAvgLatency) {
            maxAvgLatency = socketStats[i].avgLatencyMs;
            slowestSocket = i;
        }
    }
    
    printf("🐌 Slowest Socket: %d (Avg Latency: %.2f ms)\n", slowestSocket, maxAvgLatency);
    
    // Analyze recent latencies for variance
    for (NnUint i = 0; i < nSockets; i++) {
        if (!socketStats) continue;
        NnSocketPerformanceStats& stats = socketStats[i];
        if (stats.recentLatencies.size() > 10) {
            std::vector<double> latencies = stats.recentLatencies;
            std::sort(latencies.begin(), latencies.end());
            
            size_t size = latencies.size();
            if (size > 0) {
                double p50 = latencies[size / 2];
                double p95 = latencies[static_cast<size_t>(size * 0.95)];
                double p99 = latencies[static_cast<size_t>(size * 0.99)];
                
                printf("Socket %d: P50=%.2fms, P95=%.2fms, P99=%.2fms\n", i, p50, p95, p99);
                
                // Detect potential bottlenecks
                if (p95 > p50 * 2.0) {
                    printf("⚠️  Socket %d shows high latency variance (P95 >> P50) - potential network congestion\n", i);
                }
                if (stats.bandwidthMbps < 10.0 && stats.totalOperations > 100) {
                    printf("⚠️  Socket %d has low bandwidth (%.2f Mbps) - potential bandwidth limitation\n", i, stats.bandwidthMbps);
                }
            }
        }
    }
    
    // Analyze operation types
    std::map<std::string, int> operationCounts;
    std::map<std::string, NnSize> operationBytes;
    std::map<std::string, double> operationLatencies;
    
    for (const auto& metric : recentMetrics) {
        operationCounts[metric.operationType]++;
        operationBytes[metric.operationType] += metric.bytesTransferred;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(metric.endTime - metric.startTime);
        operationLatencies[metric.operationType] += duration.count() / 1000.0;
    }
    
    printf("\n📈 Operation Analysis:\n");
    for (const auto& op : operationCounts) {
        double avgLatency = operationLatencies[op.first] / op.second;
        double totalMB = operationBytes[op.first] / (1024.0 * 1024.0);
        printf("  %s: %d ops, %.2f MB, %.2f ms avg\n", op.first.c_str(), op.second, totalMB, avgLatency);
    }
    
    printf("\n");
}

NnSocketPerformanceStats* NnNetwork::getSocketStats(NnUint socketIndex) {
    if (socketIndex >= nSockets) return nullptr;
    return &socketStats[socketIndex];
}

static void syncWithRoot(NnNetwork *network, NnByte nodeIndex, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    if (nodeIndex == 0) {
        // root

        NnUint nSocketsPerThread = network->nSockets / nThreads + (network->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    } else {
        // worker

        if (threadIndex != 0) return;

        NnSocketIo ios;
        ios.data = buffer;
        ios.size = nBytes;
        ios.socketIndex = 0; // root
        network->readMany(1, &ios);
    }
}

// Original O(n^2) implementation - kept for backward compatibility
static void syncNodeSlices_alltoall(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    bool isWorker = nodeIndex != 0;
    NnUint nSockets = onlyFromWorkerToRoot && isWorker ? 1 : network->nSockets;
    NnUint nSocketsPerThread = nSockets / nThreads + (nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;
    NnSize sliceBytes = nBytes / nNodes;

    std::vector<NnSocketIo> ios(nSocketsPerThread);

    if (!onlyFromWorkerToRoot || isWorker) {
        NnByte *mySliceData = &buffer[sliceBytes * nodeIndex];

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            ios[i].socketIndex = socketIndex;
            ios[i].data = mySliceData;
            ios[i].size = sliceBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    }

    if (!onlyFromWorkerToRoot || !isWorker) {
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            NnUint sliceIndex = socketIndex >= nodeIndex ? socketIndex + 1 : socketIndex;
            NnByte *sliceData = &buffer[sliceBytes * sliceIndex];
            ios[i].socketIndex = socketIndex;
            ios[i].data = sliceData;
            ios[i].size = sliceBytes;
        }
        network->readMany(nSocketsPerThread, &ios[0]);
    }
}

static inline NnUint getSocketIndexForNode(NnUint myNodeIndex, NnUint peerNode) {
    // Root (node 0): sockets map directly to workers
    // socket[0] -> worker 1, socket[1] -> worker 2, etc.
    if (myNodeIndex == 0) {
        return peerNode - 1;
    }
    
    // Workers: socket[0] is root, socket[1..n-2] are other workers
    // When communicating with other workers, skip self
    if (peerNode == 0) {
        return 0;  // Root is always socket[0] for workers
    }
    
    // Worker to worker: map peer node to socket index
    // Socket indices for workers: 0=root, 1=node1, 2=node2, ..., but skip self
    if (peerNode < myNodeIndex) {
        return peerNode;  // Peer comes before us in socket array
    } else {
        return peerNode - 1;  // Peer comes after us, but we skip ourselves
    }
}

static void ringAllGather(NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize sliceBytes, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex != 0) return;

    NnUint sendToNode = (nodeIndex + 1) % nNodes;
    NnUint recvFromNode = (nodeIndex + nNodes - 1) % nNodes;

    NnUint sendSocketIndex = getSocketIndexForNode(nodeIndex, sendToNode);
    NnUint recvSocketIndex = getSocketIndexForNode(nodeIndex, recvFromNode);

    for (NnUint step = 0; step < nNodes - 1; step++) {
        NnUint sendSliceIndex = (nodeIndex - step + nNodes) % nNodes;
        NnUint recvSliceIndex = (nodeIndex - step - 1 + nNodes) % nNodes;

        NnSocketIo sendIo, recvIo;
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendSliceIndex];
        sendIo.size = sliceBytes;

        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = &buffer[sliceBytes * recvSliceIndex];
        recvIo.size = sliceBytes;

        // Even nodes send first, odd nodes receive first to avoid deadlock
        if (nodeIndex % 2 == 0) {
            network->writeMany(1, &sendIo);
            network->readMany(1, &recvIo);
        } else {
            network->readMany(1, &recvIo);
            network->writeMany(1, &sendIo);
        }
    }
}

static void binaryTreeBroadcast(NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex != 0) return;

    NnUint treeDepth = 0;
    NnUint temp = nNodes - 1;
    while (temp > 0) {
        treeDepth++;
        temp >>= 1;
    }

    for (NnUint level = 0; level < treeDepth; level++) {
        NnUint step = 1 << level;
        NnUint stride = step << 1;

        if (nodeIndex % stride == step && nodeIndex < nNodes) {
            NnUint parentNode = nodeIndex - step;
            if (parentNode < nNodes) {
                NnUint socketIndex = getSocketIndexForNode(nodeIndex, parentNode);
                NnSocketIo io;
                io.socketIndex = socketIndex;
                io.data = buffer;
                io.size = nBytes;
                network->readMany(1, &io);
            }
        } else if (nodeIndex % stride == 0 && nodeIndex + step < nNodes) {
            NnUint childNode = nodeIndex + step;
            if (childNode < nNodes) {
                NnUint socketIndex = getSocketIndexForNode(nodeIndex, childNode);
                NnSocketIo io;
                io.socketIndex = socketIndex;
                io.data = buffer;
                io.size = nBytes;
                network->writeMany(1, &io);
            }
        }
    }
}

static inline bool isAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

static void reduceSum(NnByte *result, const NnByte *input, NnSize nBytes, NnFloatType floatType) {
    // Safety check: ensure we have valid pointers and non-zero size
    if (!result || !input || nBytes == 0) {
        return;
    }
    
    // For maximum safety and compatibility, always use memcpy approach
    // This avoids any potential alignment issues regardless of input buffer alignment
    if (floatType == F_32 || floatType == F_Q80 || floatType == F_Q40) {
        // F32, Q80, Q40: Treat as float arrays
        // Always use memcpy to aligned stack buffers - safest approach
        NnSize nElements = nBytes / sizeof(float);
        if (nElements > 0) {
            const NnSize chunkSize = 256;  // Process 256 floats at a time (1KB chunks)
            float inputBuf[chunkSize];
            float resultBuf[chunkSize];
            
            NnSize processed = 0;
            while (processed < nElements) {
                NnSize currentChunk = (nElements - processed < chunkSize) ? 
                                     (nElements - processed) : chunkSize;
                
                // Copy to stack buffers (always aligned on stack)
                std::memcpy(inputBuf, input + processed * sizeof(float), currentChunk * sizeof(float));
                std::memcpy(resultBuf, result + processed * sizeof(float), currentChunk * sizeof(float));
                
                // Perform reduction on aligned buffers
                for (NnSize i = 0; i < currentChunk; i++) {
                    resultBuf[i] += inputBuf[i];
                }
                
                // Copy back
                std::memcpy(result + processed * sizeof(float), resultBuf, currentChunk * sizeof(float));
                processed += currentChunk;
            }
        }
        
        // Handle any remaining bytes (not multiple of float size)
        NnSize remainingBytes = nBytes % sizeof(float);
        if (remainingBytes > 0) {
            NnSize offset = nElements * sizeof(float);
            for (NnSize i = 0; i < remainingBytes; i++) {
                result[offset + i] = static_cast<NnByte>(result[offset + i] + input[offset + i]);
            }
        }
    } else if (floatType == F_16) {
        // F16: Use memcpy for each element (always safe regardless of alignment)
        NnSize nElements = nBytes / sizeof(NnFp16);
        for (NnSize i = 0; i < nElements; i++) {
            NnFp16 inputVal, resultVal;
            // Use memcpy to avoid alignment issues
            std::memcpy(&inputVal, input + i * sizeof(NnFp16), sizeof(NnFp16));
            std::memcpy(&resultVal, result + i * sizeof(NnFp16), sizeof(NnFp16));
            
            float a = convertF16toF32Impl(resultVal);
            float b = convertF16toF32Impl(inputVal);
            float sum = a + b;
            NnFp16 sumVal = convertF32ToF16Impl(sum);
            
            std::memcpy(result + i * sizeof(NnFp16), &sumVal, sizeof(NnFp16));
        }
        
        // Handle remaining bytes (if any)
        NnSize remainingBytes = nBytes % sizeof(NnFp16);
        if (remainingBytes > 0) {
            NnSize offset = nElements * sizeof(NnFp16);
            for (NnSize i = 0; i < remainingBytes; i++) {
                result[offset + i] = static_cast<NnByte>(result[offset + i] + input[offset + i]);
            }
        }
    } else {
        // Fallback: byte-wise addition (always safe for unknown types)
        for (NnSize i = 0; i < nBytes; i++) {
            result[i] = static_cast<NnByte>(result[i] + input[i]);
        }
    }
}

static void syncNodeSlices_ringAllReduce(bool onlyFromWorkerToRoot,
                                         NnNetwork *network,
                                         NnUint nodeIndex,
                                         NnUint tpGroupStart,
                                         NnUint tpGroupEnd,
                                         NnByte *buffer,
                                         NnSize nBytes,
                                         NnFloatType floatType,
                                         NnUint nThreads,
                                         NnUint threadIndex) {
    if (threadIndex != 0) return;  // Only thread 0 handles ring communication

    NnUint nNodes = tpGroupEnd - tpGroupStart;
    if (nNodes <= 1) return;
    NnUint localNodeIndex = nodeIndex - tpGroupStart;
    NnSize sliceBytes = nBytes / nNodes;

    // Ring topology: each node sends to next and receives from previous
    NnUint sendToNode = tpGroupStart + ((localNodeIndex + 1) % nNodes);
    NnUint recvFromNode = tpGroupStart + ((localNodeIndex + nNodes - 1) % nNodes);

    NnUint sendSocketIndex = getSocketIndexForNode(nodeIndex, sendToNode);
    NnUint recvSocketIndex = getSocketIndexForNode(nodeIndex, recvFromNode);

    // ========== PHASE 1: REDUCE-SCATTER ==========
    // Use thread_local vector for safe automatic memory management
    // This avoids stack overflow and manual memory management issues
    static thread_local std::vector<NnByte> tempBuffer;
    if (tempBuffer.size() < sliceBytes) {
        tempBuffer.reserve(sliceBytes);
        tempBuffer.resize(sliceBytes, 0);
    }
    NnByte* tempBuffer_ptr = tempBuffer.data();
    
    // In n-1 steps, each node accumulates one chunk through reduction
    for (NnUint step = 0; step < nNodes - 1; step++) {
        // Determine which chunk to send and where to receive
        NnUint sendChunkIndex = (localNodeIndex - step + nNodes) % nNodes;
        NnUint recvChunkIndex = (localNodeIndex - step - 1 + nNodes) % nNodes;

        NnSocketIo sendIo, recvIo;
        
        // Use pre-allocated buffer
        NnByte* alignedBuffer = tempBuffer_ptr;
        
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendChunkIndex];
        sendIo.size = sliceBytes;
        
        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = alignedBuffer;
        recvIo.size = sliceBytes;
        
        // Even nodes send first, odd nodes receive first (avoid deadlock)
        if (localNodeIndex % 2 == 0) {
            network->writeMany(1, &sendIo);
            network->readMany(1, &recvIo);
        } else {
            network->readMany(1, &recvIo);
            network->writeMany(1, &sendIo);
        }
        
        // Reduce: add received chunk to corresponding local chunk
        // Ensure we don't write beyond buffer bounds
        if (sliceBytes * recvChunkIndex + sliceBytes <= nBytes) {
            reduceSum(&buffer[sliceBytes * recvChunkIndex], alignedBuffer, sliceBytes, floatType);
        }
    }
    
    // At this point, each node has one fully reduced chunk
    
    if (onlyFromWorkerToRoot) {
        return;  // If only reduce-scatter needed, we're done
    }
    
    // ========== PHASE 2: ALL-GATHER ==========
    // In n-1 steps, collect all reduced chunks
    for (NnUint step = 0; step < nNodes - 1; step++) {
        NnUint sendChunkIndex = (localNodeIndex - step + nNodes) % nNodes;
        NnUint recvChunkIndex = (localNodeIndex - step - 1 + nNodes) % nNodes;

        NnSocketIo sendIo, recvIo;
        
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendChunkIndex];
        sendIo.size = sliceBytes;
        
        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = &buffer[sliceBytes * recvChunkIndex];
        recvIo.size = sliceBytes;
        
        // Even nodes send first, odd nodes receive first
        if (localNodeIndex % 2 == 0) {
            network->writeMany(1, &sendIo);
            network->readMany(1, &recvIo);
        } else {
            network->readMany(1, &recvIo);
            network->writeMany(1, &sendIo);
        }
    }
    
    // Now all nodes have the fully reduced result in all chunks
}

static void syncNodeSlices_starAllReduce(bool onlyFromWorkerToRoot,
                                         NnNetwork *network,
                                         NnUint nodeIndex,
                                         NnUint tpGroupStart,
                                         NnUint tpGroupEnd,
                                         NnByte *buffer,
                                         NnSize nBytes,
                                         NnFloatType floatType,
                                         NnUint nThreads,
                                         NnUint threadIndex) {
    if (threadIndex != 0) return;

    NnUint nNodes = tpGroupEnd - tpGroupStart;
    if (nNodes <= 1) return;
    NnUint rootNode = tpGroupStart;
    NnUint nWorkers = nNodes - 1;

    if (nodeIndex == rootNode) {
        // Gather all worker buffers concurrently via readMany
        static thread_local std::vector<NnByte> gatherBuffer;
        NnSize totalGatherSize = nBytes * nWorkers;
        if (gatherBuffer.size() < totalGatherSize) {
            gatherBuffer.resize(totalGatherSize, 0);
        }

        std::vector<NnSocketIo> readIos(nWorkers);
        for (NnUint w = 0; w < nWorkers; w++) {
            NnUint workerNode = tpGroupStart + 1 + w;
            readIos[w].socketIndex = getSocketIndexForNode(nodeIndex, workerNode);
            readIos[w].data = &gatherBuffer[w * nBytes];
            readIos[w].size = nBytes;
        }
        network->readMany(nWorkers, readIos.data());

        for (NnUint w = 0; w < nWorkers; w++) {
            reduceSum(buffer, &gatherBuffer[w * nBytes], nBytes, floatType);
        }

        if (onlyFromWorkerToRoot) {
            return;
        }

        std::vector<NnSocketIo> writeIos(nWorkers);
        for (NnUint w = 0; w < nWorkers; w++) {
            NnUint workerNode = tpGroupStart + 1 + w;
            writeIos[w].socketIndex = getSocketIndexForNode(nodeIndex, workerNode);
            writeIos[w].data = buffer;
            writeIos[w].size = nBytes;
        }
        network->writeMany(nWorkers, writeIos.data());
    } else {
        NnSocketIo sendIo;
        sendIo.socketIndex = getSocketIndexForNode(nodeIndex, rootNode);
        sendIo.data = buffer;
        sendIo.size = nBytes;
        network->writeMany(1, &sendIo);

        if (onlyFromWorkerToRoot) {
            return;
        }

        NnSocketIo recvIo;
        recvIo.socketIndex = getSocketIndexForNode(nodeIndex, rootNode);
        recvIo.data = buffer;
        recvIo.size = nBytes;
        network->readMany(1, &recvIo);
    }
}

static void syncNodeSlices_starGatherBroadcast(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    NnSize sliceBytes = nBytes / nNodes;
    
    // ========== PHASE 1: GATHER TO ROOT ==========
    if (nodeIndex == 0) {
        // ROOT: Collect slices from all workers
        // Distribute workers across threads for PARALLEL receive
        NnUint nWorkers = nNodes - 1;
        NnUint workersPerThread = nWorkers / nThreads + (nWorkers % nThreads > threadIndex ? 1 : 0);
        
        for (NnUint i = 0; i < workersPerThread; i++) {
            NnUint workerIdx = threadIndex + i * nThreads + 1;
            if (workerIdx < nNodes) {
                NnSocketIo io;
                io.socketIndex = workerIdx - 1;
                io.data = &buffer[sliceBytes * workerIdx];
                io.size = sliceBytes;
                network->readMany(1, &io);
            }
        }
    } else {
        // WORKER: Only thread 0 sends, but ALL threads must wait
        // This ensures thread synchronization
        if (threadIndex == 0) {
            NnSocketIo io;
            io.socketIndex = 0;
            io.data = &buffer[sliceBytes * nodeIndex];
            io.size = sliceBytes;
            network->writeMany(1, &io);
        }
        // Implicit barrier: all threads exit this block together
        // No early return - this keeps threads synchronized
    }
    
    // If only gathering to root, we're done
    if (onlyFromWorkerToRoot) {
        // All threads reach here together
        return;
    }
    
    // ========== PHASE 2: BROADCAST FROM ROOT ==========
    if (nodeIndex == 0) {
        // ROOT: Broadcast complete buffer to all workers
        // Distribute workers across threads for PARALLEL send
        NnUint nWorkers = nNodes - 1;
        NnUint workersPerThread = nWorkers / nThreads + (nWorkers % nThreads > threadIndex ? 1 : 0);
        
        for (NnUint i = 0; i < workersPerThread; i++) {
            NnUint workerIdx = threadIndex + i * nThreads + 1;
            if (workerIdx < nNodes) {
                NnSocketIo io;
                io.socketIndex = workerIdx - 1;
                io.data = buffer;
                io.size = nBytes;
                network->writeMany(1, &io);
            }
        }
    } else {
        // WORKER: Only thread 0 receives, but ALL threads must wait
        if (threadIndex == 0) {
            NnSocketIo io;
            io.socketIndex = 0;
            io.data = buffer;
            io.size = nBytes;
            network->readMany(1, &io);
        }
        // Implicit barrier: all threads exit this function together
    }
    
    // All threads reach here together - synchronized!
}

static void syncNodeSlices(bool onlyFromWorkerToRoot,
                           NnNetwork *network,
                           NnUint nodeIndex,
                           NnUint tpGroupStart,
                           NnUint tpGroupEnd,
                           NnByte *buffer,
                           NnSize nBytes,
                           NnFloatType floatType,
                           CollectiveType collectiveType,
                           NnUint nThreads,
                           NnUint threadIndex) {
    static std::atomic<bool> ringFallbackWarned(false);

    if (!(tpGroupStart <= nodeIndex && nodeIndex < tpGroupEnd)) {
        throw std::invalid_argument("node is outside its TP group");
    }
    NnUint nNodes = tpGroupEnd - tpGroupStart;
    if (nNodes <= 1 || nBytes == 0) return;

    CollectiveType effective = collectiveType;
    const bool ringTopologySupported =
        nNodes >= 8 &&
        (nBytes % (NnSize)nNodes) == 0;

    if (effective == COLLECTIVE_AUTO) {
        const bool canRing =
            !onlyFromWorkerToRoot &&
            ringTopologySupported;
        effective = canRing ? COLLECTIVE_RING : COLLECTIVE_STAR;
    }

    if (effective == COLLECTIVE_RING && (!ringTopologySupported || onlyFromWorkerToRoot)) {
        effective = COLLECTIVE_STAR;
    }

    if (collectiveType == COLLECTIVE_RING && effective != COLLECTIVE_RING && threadIndex == 0) {
        bool expected = false;
        if (ringFallbackWarned.compare_exchange_strong(expected, true)) {
            printf("⚠️  Collective ring downgraded to star (tp=%u, bytes=%zu)\n", nNodes, (size_t)nBytes);
            fflush(stdout);
        }
    }

    if (effective == COLLECTIVE_RING) {
        syncNodeSlices_ringAllReduce(onlyFromWorkerToRoot, network, nodeIndex, tpGroupStart, tpGroupEnd, buffer, nBytes, floatType, nThreads, threadIndex);
    } else {
        syncNodeSlices_starAllReduce(onlyFromWorkerToRoot, network, nodeIndex, tpGroupStart, tpGroupEnd, buffer, nBytes, floatType, nThreads, threadIndex);
    }
}

// CP 분할 여부. allgather 의 구간 계산 기준이 달라진다(위 blockRange 참조).
static std::atomic<bool> gNetworkCpSplit{false};

void nnNetworkSetCpSplit(bool enabled) {
    gNetworkCpSplit.store(enabled, std::memory_order_relaxed);
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, CollectiveType collectiveType, bool spPrefillOnly, bool strictKvAffinity, bool allowKvMigration) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->collectiveType = collectiveType;
    this->spPrefillOnly = spPrefillOnly;
    this->strictKvAffinity = strictKvAffinity;
    this->allowKvMigration = allowKvMigration;
    this->decodePhase.store(false);
}

void NnNetworkNodeSynchronizer::setDecodePhase(bool isDecodePhase) {
    decodePhase.store(isDecodePhase);
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];
            
            auto syncStartTime = std::chrono::high_resolution_clock::now();
            std::string syncTypeName;

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncTypeName = "SYNC_WITH_ROOT";
                // With TP group size 1, root sync is typically a no-op broadcast path.
                // Skip by default to reduce decode-path control-plane overhead.
                if (shouldSkipTp1NoopSyncs()) {
                    NnUint declaredTpSize = 0;
                    if (nodeConfig->tpGroupEnd > nodeConfig->tpGroupStart)
                        declaredTpSize = nodeConfig->tpGroupEnd - nodeConfig->tpGroupStart;
                    if (declaredTpSize <= 1)
                        continue;
                }
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncTypeName = "SYNC_NODE_SLICES";
                // TP group size 1 means there is nothing to all-reduce.
                // Skipping here prevents unnecessary cross-node sync traffic on PP-only runs.
                if (shouldSkipTp1NoopSyncs()) {
                    NnUint declaredTpSize = 0;
                    if (nodeConfig->tpGroupEnd > nodeConfig->tpGroupStart)
                        declaredTpSize = nodeConfig->tpGroupEnd - nodeConfig->tpGroupStart;
                    if (declaredTpSize <= 1)
                        continue;
                }
                NnUint tpGroupStart = nodeConfig->tpGroupStart;
                NnUint tpGroupEnd = nodeConfig->tpGroupEnd;
                if (tpGroupEnd <= tpGroupStart ||
                    tpGroupEnd > netConfig->nNodes ||
                    nodeConfig->nodeIndex < tpGroupStart ||
                    nodeConfig->nodeIndex >= tpGroupEnd) {
                    tpGroupStart = 0;
                    tpGroupEnd = netConfig->nNodes;
                }
                syncNodeSlices(false, network, nodeConfig->nodeIndex, tpGroupStart, tpGroupEnd, pipeBatch, batchBytes, pipeConfig->size.floatType, collectiveType, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncTypeName = "SYNC_NODE_SLICES_EXCEPT_ROOT";
                // TP group size 1 means there is nothing to gather/broadcast.
                if (shouldSkipTp1NoopSyncs()) {
                    NnUint declaredTpSize = 0;
                    if (nodeConfig->tpGroupEnd > nodeConfig->tpGroupStart)
                        declaredTpSize = nodeConfig->tpGroupEnd - nodeConfig->tpGroupStart;
                    if (declaredTpSize <= 1)
                        continue;
                }
                NnUint tpGroupStart = nodeConfig->tpGroupStart;
                NnUint tpGroupEnd = nodeConfig->tpGroupEnd;
                if (tpGroupEnd <= tpGroupStart ||
                    tpGroupEnd > netConfig->nNodes ||
                    nodeConfig->nodeIndex < tpGroupStart ||
                    nodeConfig->nodeIndex >= tpGroupEnd) {
                    tpGroupStart = 0;
                    tpGroupEnd = netConfig->nNodes;
                }
                syncNodeSlices(true, network, nodeConfig->nodeIndex, tpGroupStart, tpGroupEnd, pipeBatch, batchBytes, pipeConfig->size.floatType, collectiveType, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_SP_KV) {
                // SYNC_SP_KV is handled outside the batch loop (see below)
                break;
            } else if (syncConfig->syncType == SYNC_CP_LOGITS) {
                // 배치 루프 밖에서 처리한다 (아래)
                break;
            } else {
                throw std::invalid_argument("Unknown sync type");
            }

            auto syncEndTime = std::chrono::high_resolution_clock::now();

            // Record sync operation for monitoring
            if (network->isPerformanceMonitoringEnabled()) {
                NnSize totalBytes = batchBytes * execution->batchSize;
                network->recordOperation(syncTypeName, 0, totalBytes, syncStartTime, syncEndTime);
            }
        }

        // SYNC_CP_LOGITS: 마지막 토큰 행의 로짓을 root 로 회수한다.
        //
        // Ring CP 에서는 노드가 배치의 자기 블록만 계산하므로 마지막 행은 spRank N-1
        // 에만 있다. root 는 그 로짓으로 첫 토큰을 샘플링해야 한다.
        // 블록 배정을 회전시켜 root 에게 마지막 블록을 주는 방법도 있었지만,
        // SYNC_SP_KV 의 수신 측이 peerStart = spRank * localSeqLen 으로 구간을
        // 계산하기 때문에 회전시키면 KV 에 구멍이 난다.
        if (syncConfig->syncType == SYNC_CP_LOGITS) {
            static std::atomic<int> dbg{0};
            if (dbg.fetch_add(1) < 3)
                printf("[CP_LOGITS] node=%u thread=%u batchSize=%u\n",
                    nodeConfig->nodeIndex, threadIndex, execution->batchSize);
            if (threadIndex != 0) continue;
            if (execution->batchSize <= 1u) continue;  // decode 는 모든 노드가 같은 값을 낸다

            const NnUint tpSizeL = nodeConfig->tpGroupEnd - nodeConfig->tpGroupStart;
            if (tpSizeL == 0u) continue;
            const NnUint spSizeL = (nodeConfig->spGroupEnd - nodeConfig->spGroupStart) / tpSizeL;
            if (spSizeL <= 1u) continue;
            const NnUint mySpRank = (nodeConfig->nodeIndex - nodeConfig->spGroupStart) / tpSizeL;
            const NnUint myTpRank = nodeConfig->nodeIndex - nodeConfig->tpGroupStart;

            const NnPipeConfig *lgConfig = &netConfig->pipes[syncConfig->pipeIndex];
            const NnSize rowBytes = lgConfig->size.nBytes / netConfig->nBatches;
            NnByte *lg = execution->pipes[syncConfig->pipeIndex];
            NnByte *lastRow = lg + (NnSize)(execution->batchSize - 1u) * rowBytes;

            const NnUint senderSpRank = spSizeL - 1u;
            const NnUint rootNode = nodeConfig->spGroupStart + myTpRank;
            const NnUint senderNode = nodeConfig->spGroupStart + senderSpRank * tpSizeL + myTpRank;
            if (senderNode == rootNode)
                continue;

            static std::atomic<int> dbg2{0};
            if (dbg2.fetch_add(1) < 3)
                printf("[CP_LOGITS] node=%u mySpRank=%u sender=%u root=%u rowBytes=%zu\n",
                    nodeConfig->nodeIndex, mySpRank, senderNode, rootNode, (size_t)rowBytes);
            auto argmaxOf = [&](const NnByte *row) {
                const float *f = (const float *)row;
                const NnUint n = (NnUint)(rowBytes / sizeof(float));
                NnUint best = 0; float bv = f[0];
                for (NnUint i = 1; i < n; i++) if (f[i] > bv) { bv = f[i]; best = i; }
                printf("[CP_LOGITS] node=%u argmax=%u val=%f\n", nodeConfig->nodeIndex, best, bv);
            };
            if (mySpRank == senderSpRank) {
                argmaxOf(lastRow);
                const NnUint sock = getSocketIndexForNode(nodeConfig->nodeIndex, rootNode);
                network->write(sock, lastRow, rowBytes);
                network->addTaggedTraffic(false, rowBytes, 0);
            } else if (nodeConfig->nodeIndex == rootNode) {
                const NnUint sock = getSocketIndexForNode(nodeConfig->nodeIndex, senderNode);
                network->read(sock, lastRow, rowBytes);
                network->addTaggedTraffic(false, 0, rowBytes);
                argmaxOf(lastRow);
            }
            continue;
        }

        // SYNC_SP_KV: allgather K and V buffers across SP group
        if (syncConfig->syncType == SYNC_SP_KV) {
            const bool isDecode = decodePhase.load();
            // Prefill-only SP mode: decode path must not run KV sync.
            // This is the normal strict-KV-affinity fast path and should not be treated as violation.
            if (spPrefillOnly && isDecode) continue;

            // If decode still reaches SYNC_SP_KV, we're effectively migrating/fetching KV over network.
            if (isDecode && strictKvAffinity && !allowKvMigration) {
                network->noteKvMigrationViolation();
                throw std::runtime_error("KV_MIGRATION_VIOLATION: decode-path KV sync is disabled by --strict-kv-affinity=1");
            }
            if (isDecode && strictKvAffinity && allowKvMigration) {
                network->noteKvMigrationViolation();
            }
            if (threadIndex != 0) continue; // only thread 0 handles network I/O

            NnUint spGroupStart = nodeConfig->spGroupStart;
            NnUint spGroupEnd   = nodeConfig->spGroupEnd;
            NnUint tpGroupStart = nodeConfig->tpGroupStart;
            NnUint tpGroupEnd   = nodeConfig->tpGroupEnd;
            NnUint tpSize       = tpGroupEnd - tpGroupStart;
            NnUint actualSpSize = (spGroupEnd - spGroupStart) / tpSize;
            if (actualSpSize <= 1) continue; // nothing to do

            NnUint myNodeIndex = nodeConfig->nodeIndex;
            NnUint myTpRank    = myNodeIndex - tpGroupStart;
            NnUint mySpRank    = (myNodeIndex - spGroupStart) / tpSize;

            // Determine how many positions are active (max position across batch)
            const float *positions = (float *)execution->pipes[syncConfig->spKvPositionPipeIdx];
            NnUint maxPos = 0;
            for (NnUint b = 0; b < execution->batchSize; b++) {
                NnUint p = (NnUint)positions[b];
                if (p > maxPos) maxPos = p;
            }

            NnSize kvDim0Bytes = (NnSize)syncConfig->spKvDim0 * sizeof(float);
            NnByte *kBuf = execution->nodeBuffers[syncConfig->spKvKBufferIndex];
            NnByte *vBuf = execution->nodeBuffers[syncConfig->spKvVBufferIndex];

            // 구간 계산.
            //
            // 기본(SP)은 seqLen/N 로 자른 고정 구간이다. CP 가 켜지면 노드가 계산하는
            // 것은 "배치 행 블록"이고 그 position 은 batchSize 기준이라 기준이 다르다.
            // 그래서 CP 에서는 각 rank 의 블록 첫/마지막 행의 position 으로 구간을 잡는다.
            // (cpRowWindow 와 같은 균등 분할: 앞쪽 rem 개 블록이 1행씩 더 가진다)
            const bool cpOn = gNetworkCpSplit.load(std::memory_order_relaxed) && execution->batchSize > 1u;
            auto blockRange = [&](NnUint rank, NnUint *lo, NnUint *hi) {
                const NnUint base = execution->batchSize / actualSpSize;
                const NnUint rem = execution->batchSize % actualSpSize;
                const NnUint b0 = rank * base + (rank < rem ? rank : rem);
                const NnUint cnt = base + (rank < rem ? 1u : 0u);
                *lo = (NnUint)positions[b0];
                *hi = (NnUint)positions[b0 + cnt - 1u] + 1u;
            };

            NnUint myStart, myEnd;
            if (cpOn) {
                blockRange(mySpRank, &myStart, &myEnd);
            } else {
                myStart = syncConfig->spKvLocalSeqStart;
                myEnd   = myStart + syncConfig->spKvLocalSeqLen;
            }
            NnUint myActive = (maxPos + 1 < myEnd) ? (maxPos + 1) : myEnd;
            NnSize sendBytes = (myActive > myStart) ? (myActive - myStart) * kvDim0Bytes : 0;

            // Exchange with each other SP rank (same tpRank, different spRank)
            for (NnUint spRankPeer = 0; spRankPeer < actualSpSize; spRankPeer++) {
                if (spRankPeer == mySpRank) continue;

                NnUint peerNode   = spGroupStart + spRankPeer * tpSize + myTpRank;
                NnUint peerSocket = getSocketIndexForNode(myNodeIndex, peerNode);

                // Peer slice range
                NnUint peerStart, peerEnd;
                if (cpOn) {
                    blockRange(spRankPeer, &peerStart, &peerEnd);
                } else {
                    peerStart = spRankPeer * syncConfig->spKvLocalSeqLen;
                    peerEnd   = peerStart + syncConfig->spKvLocalSeqLen;
                }
                NnUint peerActive = (maxPos + 1 < peerEnd) ? (maxPos + 1) : peerEnd;
                NnSize recvBytes  = (peerActive > peerStart) ? (peerActive - peerStart) * kvDim0Bytes : 0;

                // Deadlock avoidance: lower spRank sends first
                if (mySpRank < spRankPeer) {
                    if (sendBytes > 0) {
                        network->write(peerSocket, kBuf + myStart * kvDim0Bytes, sendBytes);
                        network->write(peerSocket, vBuf + myStart * kvDim0Bytes, sendBytes);
                        network->addTaggedTraffic(true, sendBytes * 2, 0);
                    }
                    if (recvBytes > 0) {
                        network->read(peerSocket, kBuf + peerStart * kvDim0Bytes, recvBytes);
                        network->read(peerSocket, vBuf + peerStart * kvDim0Bytes, recvBytes);
                        network->addTaggedTraffic(true, 0, recvBytes * 2);
                    }
                } else {
                    if (recvBytes > 0) {
                        network->read(peerSocket, kBuf + peerStart * kvDim0Bytes, recvBytes);
                        network->read(peerSocket, vBuf + peerStart * kvDim0Bytes, recvBytes);
                        network->addTaggedTraffic(true, 0, recvBytes * 2);
                    }
                    if (sendBytes > 0) {
                        network->write(peerSocket, kBuf + myStart * kvDim0Bytes, sendBytes);
                        network->write(peerSocket, vBuf + myStart * kvDim0Bytes, sendBytes);
                        network->addTaggedTraffic(true, sendBytes * 2, 0);
                    }
                }
            }
        }
    }
}

static void writeString(NnNetwork *network, NnUint socketIndex, char *str) {
    NnUint bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnUint));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnUint socketIndex) {
    NnUint bytes;
    network->read(socketIndex, &bytes, sizeof(NnUint));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnUint socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnUint pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->write(socketIndex, &config->nPreSyncs, sizeof(config->nPreSyncs));
    for (NnUint preSyncIndex = 0; preSyncIndex < config->nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config->preSyncs[preSyncIndex];
        network->write(socketIndex, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnUint socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->ppRank, sizeof(config->ppRank));
    network->write(socketIndex, &config->tpRank, sizeof(config->tpRank));
    network->write(socketIndex, &config->tpGroupStart, sizeof(config->tpGroupStart));
    network->write(socketIndex, &config->tpGroupEnd, sizeof(config->tpGroupEnd));
    network->write(socketIndex, &config->spRank, sizeof(config->spRank));
    network->write(socketIndex, &config->spGroupStart, sizeof(config->spGroupStart));
    network->write(socketIndex, &config->spGroupEnd, sizeof(config->spGroupEnd));
    network->write(socketIndex, &config->positionPipeIndex, sizeof(config->positionPipeIndex));
    network->write(socketIndex, &config->tokenPipeIndex, sizeof(config->tokenPipeIndex));
    network->write(socketIndex, &config->xPipeIndex, sizeof(config->xPipeIndex));
    network->write(socketIndex, &config->logitsPipeIndex, sizeof(config->logitsPipeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnUint bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
            network->write(socketIndex, &syncConfig->spKvKBufferIndex, sizeof(syncConfig->spKvKBufferIndex));
            network->write(socketIndex, &syncConfig->spKvVBufferIndex, sizeof(syncConfig->spKvVBufferIndex));
            network->write(socketIndex, &syncConfig->spKvLocalSeqStart, sizeof(syncConfig->spKvLocalSeqStart));
            network->write(socketIndex, &syncConfig->spKvLocalSeqLen, sizeof(syncConfig->spKvLocalSeqLen));
            network->write(socketIndex, &syncConfig->spKvDim0, sizeof(syncConfig->spKvDim0));
            network->write(socketIndex, &syncConfig->spKvPositionPipeIdx, sizeof(syncConfig->spKvPositionPipeIdx));
        }
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            network->write(socketIndex, &opConfig->code, sizeof(opConfig->code));
            network->write(socketIndex, &opConfig->index, sizeof(opConfig->index));
            network->write(socketIndex, &opConfig->weightSize, sizeof(opConfig->weightSize));
            network->write(socketIndex, &opConfig->configSize, sizeof(opConfig->configSize));
            writeString(network, socketIndex, opConfig->name);
            network->write(socketIndex, &opConfig->input, sizeof(opConfig->input));
            network->write(socketIndex, &opConfig->output, sizeof(opConfig->output));
            if (opConfig->configSize > 0)
                network->write(socketIndex, opConfig->config, opConfig->configSize);
        }
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs) {
    for (NnUint nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnUint socketIndex = nodeIndex - 1;
        writeNet(socketIndex, netConfig);
        writeNode(socketIndex, &nodeConfigs[nodeIndex]);
    }
}

NnWorkerConfigReader::NnWorkerConfigReader(NnNetwork *network) {
    this->network = network;
}

NnNetConfig NnWorkerConfigReader::readNet() {
    network->readAck(ROOT_SOCKET_INDEX);
    NnNetConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nBatches, sizeof(config.nBatches));
    network->read(ROOT_SOCKET_INDEX, &config.nNodes, sizeof(config.nNodes));
    network->read(ROOT_SOCKET_INDEX, &config.nPipes, sizeof(config.nPipes));
    config.pipes = new NnPipeConfig[config.nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->read(ROOT_SOCKET_INDEX, &config.nPreSyncs, sizeof(config.nPreSyncs));
    config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
    for (NnUint preSyncIndex = 0; preSyncIndex < config.nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config.preSyncs[preSyncIndex];
        network->read(ROOT_SOCKET_INDEX, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnNodeConfig NnWorkerConfigReader::readNode() {
    network->readAck(ROOT_SOCKET_INDEX);

    NnNodeConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nodeIndex, sizeof(config.nodeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.ppRank, sizeof(config.ppRank));
    network->read(ROOT_SOCKET_INDEX, &config.tpRank, sizeof(config.tpRank));
    network->read(ROOT_SOCKET_INDEX, &config.tpGroupStart, sizeof(config.tpGroupStart));
    network->read(ROOT_SOCKET_INDEX, &config.tpGroupEnd, sizeof(config.tpGroupEnd));
    network->read(ROOT_SOCKET_INDEX, &config.spRank, sizeof(config.spRank));
    network->read(ROOT_SOCKET_INDEX, &config.spGroupStart, sizeof(config.spGroupStart));
    network->read(ROOT_SOCKET_INDEX, &config.spGroupEnd, sizeof(config.spGroupEnd));
    network->read(ROOT_SOCKET_INDEX, &config.positionPipeIndex, sizeof(config.positionPipeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.tokenPipeIndex, sizeof(config.tokenPipeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.xPipeIndex, sizeof(config.xPipeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.logitsPipeIndex, sizeof(config.logitsPipeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.nBuffers, sizeof(config.nBuffers));
    network->read(ROOT_SOCKET_INDEX, &config.nSegments, sizeof(config.nSegments));

    config.buffers = new NnBufferConfig[config.nBuffers];
    config.segments = new NnSegmentConfig[config.nSegments];

    for (NnUint bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvKBufferIndex, sizeof(syncConfig->spKvKBufferIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvVBufferIndex, sizeof(syncConfig->spKvVBufferIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvLocalSeqStart, sizeof(syncConfig->spKvLocalSeqStart));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvLocalSeqLen, sizeof(syncConfig->spKvLocalSeqLen));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvDim0, sizeof(syncConfig->spKvDim0));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->spKvPositionPipeIdx, sizeof(syncConfig->spKvPositionPipeIdx));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
                network->read(ROOT_SOCKET_INDEX, &opConfig->code, sizeof(opConfig->code));
                network->read(ROOT_SOCKET_INDEX, &opConfig->index, sizeof(opConfig->index));
                network->read(ROOT_SOCKET_INDEX, &opConfig->weightSize, sizeof(opConfig->weightSize));
                network->read(ROOT_SOCKET_INDEX, &opConfig->configSize, sizeof(opConfig->configSize));
                opConfig->name = readString(network, ROOT_SOCKET_INDEX);
                network->read(ROOT_SOCKET_INDEX, &opConfig->input, sizeof(opConfig->input));
                network->read(ROOT_SOCKET_INDEX, &opConfig->output, sizeof(opConfig->output));
                if (opConfig->configSize > 0) {
                    opConfig->config = new NnByte[opConfig->configSize];
                    network->read(ROOT_SOCKET_INDEX, opConfig->config, opConfig->configSize);
                }
            }
        }
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes, NnUint ppSize, NnUint nLayers) {
    this->executor = executor;
    this->network = network;
    this->nNodes = nNodes;
    this->ppSize = ppSize;
    this->nLayers = nLayers;
    this->tempSize = 0;
}

NnRootWeightLoader::~NnRootWeightLoader() {
    if (tempSize > 0)
        delete[] temp;
}

static inline bool isMissingOpByName(const std::invalid_argument &e) {
    return std::strncmp(e.what(), "Cannot locate op by name:", 24) == 0;
}

// Returns the PP stage rank that owns the given layer index.
// With pp=1 (pure TP), all layers belong to ppRank=0 (UINT_MAX = send to all).
static inline NnUint getLayerOwnerPpRank(NnUint layerIndex, NnUint nLayers, NnUint ppSize) {
    if (ppSize <= 1 || nLayers == 0) return UINT_MAX;
    NnUint layersPerStage = nLayers / ppSize;
    if (layersPerStage == 0) return UINT_MAX;
    NnUint ppRank = layerIndex / layersPerStage;
    if (ppRank >= ppSize) ppRank = ppSize - 1; // last stage gets remainder
    return ppRank;
}

// Returns true if nodeIndex belongs to the given PP stage (any TP rank within it).
static inline bool nodeInPpStage(NnUint nodeIndex, NnUint targetPpRank, NnUint tpSize) {
    return (nodeIndex / tpSize) == targetPpRank;
}

void NnRootWeightLoader::finish() {
    NnUint zeroSize = 0;
    for (NnUint socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
        network->write(socketIndex, &zeroSize, sizeof(zeroSize));
        network->readAck(socketIndex);
    }
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
    }
}

void NnRootWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnRootWeightLoader::writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    NnUint nameSize = std::strlen(opName) + 1;
    NnUint socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &offset, sizeof(offset));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    try {
        executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    } catch (const std::invalid_argument &e) {
        if (!isMissingOpByName(e))
            throw;
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    try {
        executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    } catch (const std::invalid_argument &e) {
        if (!isMissingOpByName(e))
            throw;
    }

    if (nNodes > 1u) {
        for (NnUint nodeIndex = 1u; nodeIndex < nNodes; nodeIndex++)
            writeWeight(nodeIndex, opName, opIndex, 0u, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    const NnUint tpSize = slice->nNodes;
    const NnUint ownerPpRank = getLayerOwnerPpRank(opIndex, nLayers, ppSize);
    if (nNodes == 1u) {
        try {
            executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
        } catch (const std::invalid_argument &e) {
            if (!isMissingOpByName(e))
                throw;
        }
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            if (ownerPpRank != UINT_MAX && !nodeInPpStage(nodeIndex, ownerPpRank, tpSize))
                continue;
            NnUint tpRank = nodeIndex % tpSize;
            splitRowMatmulWeight(slice, tpRank, weight, temp);
            if (nodeIndex == 0u) {
                try {
                    executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
                } catch (const std::invalid_argument &e) {
                    if (!isMissingOpByName(e))
                        throw;
                }
            } else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight, NnUint overridePpRank) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    const NnUint tpSize = slice->nNodes;
    const NnUint ownerPpRank = (overridePpRank != UINT_MAX) ? overridePpRank : getLayerOwnerPpRank(opIndex, nLayers, ppSize);
    if (nNodes == 1) {
        try {
            executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
        } catch (const std::invalid_argument &e) {
            if (!isMissingOpByName(e))
                throw;
        }
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            if (ownerPpRank != UINT_MAX && !nodeInPpStage(nodeIndex, ownerPpRank, tpSize))
                continue;
            NnUint tpRank = nodeIndex % tpSize;
            splitColMatmulWeight(slice, tpRank, weight, temp);
            if (nodeIndex == 0) {
                try {
                    executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
                } catch (const std::invalid_argument &e) {
                    if (!isMissingOpByName(e))
                        throw;
                }
            } else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnWorkerWeightReader::NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network) {
    this->executor = executor;
    this->network = network;
    this->tempSize = 0;
}

NnWorkerWeightReader::~NnWorkerWeightReader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnWorkerWeightReader::allocate(NnUint size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnUint nameSize;
    NnUint opIndex;
    NnSize offset;
    NnSize nBytes;
    while (true) {
        network->read(0, &nameSize, sizeof(nameSize));
        if (nameSize == 0) {
            network->writeAck(ROOT_SOCKET_INDEX);
            if (tempSize > 0) {
                delete[] temp;
                tempSize = 0;
            }
            break;
        }
        std::unique_ptr<char[]> opNamePtr(new char[nameSize]);
        char *opName = opNamePtr.get();
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &offset, sizeof(offset));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        try {
            executor->loadWeight(opName, opIndex, offset, nBytes, temp);
            printf("💿 Loaded %22s %3d, %12zu kB\n", opName, opIndex, nBytes / 1024);
        } catch (const std::invalid_argument &e) {
            if (std::strncmp(e.what(), "Cannot locate op by name:", 24) != 0)
                throw;
        }
    }
    printf("💿 Weights loaded\n");
}
