// 노드 간 메모리 경로 특성 측정 (research/16 §7.x 순위 역전 진단)
//
// 왜 필요한가:
//   governor 실험(§7.x)에서 DVFS 가 기각됐다 — ondemand 에서도 실효 주파수가
//   2.4 GHz 였고, performance 로 고정해도 GEMM/attention 순위 역전이 유지됐다.
//   남은 유력 후보는 메모리/캐시 서브시스템이다. Q4xQ8 GEMM 은 양자화 가중치를
//   스트리밍하므로 **대역폭**에, F32 attention 은 KV cache 를 훑으므로
//   **지연**에 민감할 것으로 예상된다. 그 둘을 분리해 잰다.
//
//   두 지표가 노드별 GEMM/attention 상대속도와 상관되면 메모리 원인이 강화되고,
//   상관이 없으면 남은 후보는 background/IRQ 다.
//
// build: gcc -O2 -mcpu=native membench.c -o membench -lpthread
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

volatile size_t g_sink;   // 최적화 제거 방지

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ── STREAM triad: a[i] = b[i] + s*c[i] ─────────────────────────────────
// 배열이 L3(2 MB)를 크게 넘으므로 DRAM 대역폭을 잰다.
typedef struct { double *a, *b, *c; size_t n; int reps; } TArg;
static void *triad(void *p) {
    TArg *t = (TArg *)p;
    for (int r = 0; r < t->reps; r++)
        for (size_t i = 0; i < t->n; i++) t->a[i] = t->b[i] + 3.0 * t->c[i];
    return NULL;
}

static double stream_gbs(size_t nElem, int nThreads, int reps) {
    const size_t tot = nElem * nThreads;
    double *a = aligned_alloc(64, tot * sizeof(double));
    double *b = aligned_alloc(64, tot * sizeof(double));
    double *c = aligned_alloc(64, tot * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc 실패\n"); exit(1); }
    for (size_t i = 0; i < tot; i++) { a[i] = 0; b[i] = 1.0; c[i] = 2.0; }

    pthread_t th[8]; TArg ar[8];
    const size_t per = nElem;   // 스레드당 고정 — 총량이 스레드 수에 비례한다
    // 워밍 1회
    for (int w = 0; w < 2; w++) {
        double t0 = now();
        for (int i = 0; i < nThreads; i++) {
            ar[i] = (TArg){ a + i*per, b + i*per, c + i*per, per, reps };
            pthread_create(&th[i], NULL, triad, &ar[i]);
        }
        for (int i = 0; i < nThreads; i++) pthread_join(th[i], NULL);
        double dt = now() - t0;
        if (w == 1) {
            free(a); free(b); free(c);
            // triad 는 읽기 2 + 쓰기 1 = 24 B/elem
            return (double)per * nThreads * reps * 24.0 / dt / 1e9;
        }
    }
    return 0;
}

// ── pointer chase: 무작위 순열을 따라가는 의존 체인 ────────────────────
// 각 접근이 이전 결과에 의존하므로 프리페치가 듣지 않는다 → 순수 지연.
static double chase_ns(size_t bytes, int reps) {
    const size_t n = bytes / sizeof(size_t);
    size_t *buf = aligned_alloc(64, n * sizeof(size_t));
    if (!buf) { fprintf(stderr, "alloc 실패\n"); exit(1); }
    // 캐시라인 단위(8 elem)로 건너뛰는 순열을 만든다
    const size_t stride = 64 / sizeof(size_t);
    const size_t m = n / stride;
    size_t *idx = malloc(m * sizeof(size_t));
    for (size_t i = 0; i < m; i++) idx[i] = i;
    for (size_t i = m - 1; i > 0; i--) {          // Fisher-Yates
        size_t j = (size_t)((double)rand() / RAND_MAX * i);
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    for (size_t i = 0; i < m; i++) buf[idx[i]*stride] = idx[(i+1)%m]*stride;
    free(idx);

    size_t p = 0;
    for (size_t i = 0; i < m; i++) p = buf[p];    // 워밍
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < m; i++) p = buf[p];
    double dt = now() - t0;
    g_sink = p;                                   // 최적화 방지
    free(buf);
    return dt / ((double)m * reps) * 1e9;
}

int main(void) {
    char host[128] = {0}; gethostname(host, sizeof(host)-1);
    srand(42);                                     // 노드 간 동일 순열
    printf("# host=%s\n", host);
    printf("STREAM\t1\t%.2f\n", stream_gbs(4u<<20, 1, 3));   // 스레드당 32 MB/배열
    printf("STREAM\t4\t%.2f\n", stream_gbs(4u<<20, 4, 3));
    static const size_t SZ[] = { 32u<<10, 256u<<10, 2u<<20, 16u<<20, 128u<<20 };
    static const int RP[]    = { 200, 50, 10, 3, 1 };
    for (int i = 0; i < 5; i++)
        printf("CHASE\t%zu\t%.2f\n", SZ[i]>>10, chase_ns(SZ[i], RP[i]));
    return 0;
}
