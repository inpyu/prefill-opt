// DerivePP 링크 캘리브레이션 (research/16-derivepp.md §2.4)
//
//   D_k(x) = alpha_k + x / beta_k
//
// alpha_k = 메시지당 고정 latency, beta_k = 유효 bandwidth.
//
// PP 경계 전송은 b_j x d activation 이므로 실제 크기 범위에서 재야 한다.
// iperf 같은 대용량 스트리밍 수치는 alpha 를 못 잡고, ssh 왕복은 프로토콜
// 처리 시간이 섞여 무효다. 그래서 전용 echo 서버/클라이언트로 잰다.
//
//   서버:      ./bench_link server <port>
//   클라이언트: ./bench_link client <host> <port> [reps]
//
// 출력(클라이언트): TSV  bytes  rtt_ms_median  oneway_ms  MBps
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

static double nowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool writeAll(int fd, const char *p, size_t n) {
    while (n) { ssize_t w = ::write(fd, p, n); if (w <= 0) return false; p += w; n -= (size_t)w; }
    return true;
}
static bool readAll(int fd, char *p, size_t n) {
    while (n) { ssize_t r = ::read(fd, p, n); if (r <= 0) return false; p += r; n -= (size_t)r; }
    return true;
}

static int runServer(int port) {
    int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((uint16_t)port);
    if (::bind(ls, (sockaddr *)&a, sizeof(a)) < 0) { perror("bind"); return 1; }
    ::listen(ls, 4);
    for (;;) {
        int cs = ::accept(ls, nullptr, nullptr);
        if (cs < 0) continue;
        ::setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::vector<char> buf(1 << 22);
        for (;;) {
            uint32_t n;
            if (!readAll(cs, (char *)&n, 4)) break;
            if (n == 0 || n > buf.size()) break;
            if (!readAll(cs, buf.data(), n)) break;
            if (!writeAll(cs, buf.data(), n)) break;   // 같은 크기를 되돌려준다
        }
        ::close(cs);
    }
}

static int runClient(const char *host, int port, int reps) {
    // PP 경계에서 실제로 오가는 크기(B x d x 원소바이트)를 포함하고,
    // alpha 를 잡기 위해 훨씬 작은 것도 넣는다.
    static const size_t SIZES[] = {
        64, 256, 1024, 4096, 16384, 65536,
        262144, 524288, 1048576, 2097152, 4194304
    };
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    char portStr[16]; std::snprintf(portStr, sizeof(portStr), "%d", port);
    if (::getaddrinfo(host, portStr, &hints, &res) != 0) { perror("getaddrinfo"); return 1; }
    int fd = ::socket(res->ai_family, res->ai_socktype, 0);
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) { perror("connect"); return 1; }
    int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ::freeaddrinfo(res);

    std::vector<char> buf(4u << 20, 'x');
    printf("# host=%s\n# bytes\trtt_ms\toneway_ms\tMBps\n", host);
    for (size_t sz : SIZES) {
        std::vector<double> ts;
        for (int r = 0; r < reps + 2; r++) {          // 앞 2회는 워밍업
            const uint32_t n = (uint32_t)sz;
            const double t0 = nowMs();
            if (!writeAll(fd, (const char *)&n, 4)) return 1;
            if (!writeAll(fd, buf.data(), sz)) return 1;
            if (!readAll(fd, buf.data(), sz)) return 1;
            const double dt = nowMs() - t0;
            if (r >= 2) ts.push_back(dt);
        }
        std::sort(ts.begin(), ts.end());
        const double rtt = ts[ts.size() / 2];
        printf("%zu\t%.4f\t%.4f\t%.1f\n", sz, rtt, rtt / 2.0,
               (double)sz / (rtt / 2.0 / 1000.0) / 1e6);
        fflush(stdout);
    }
    const uint32_t zero = 0; writeAll(fd, (const char *)&zero, 4);
    ::close(fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && std::strcmp(argv[1], "server") == 0)
        return runServer(atoi(argv[2]));
    if (argc >= 4 && std::strcmp(argv[1], "client") == 0)
        return runClient(argv[2], atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 15);
    std::fprintf(stderr, "usage: bench_link server <port> | client <host> <port> [reps]\n");
    return 2;
}
