// STREAM 스레드 확장 측정의 **타당성 검증** 전용.
//
// 왜 필요한가:
//   앞선 membench 에서 4스레드 대역폭이 1스레드보다 40% 낮게 나왔다(9.2 -> 5.5 GB/s).
//   메모리 컨트롤러 포화라면 같거나 높아야 하므로 측정 결함을 먼저 배제해야 한다.
//   아래 결함들이 모두 같은 모양을 만든다.
//     - 스레드 구간이 겹치거나 false sharing
//     - 총 처리 바이트를 스레드 수에 맞게 집계하지 않음
//     - affinity 가 깨져 일부 코어에 몰림
//     - 배열 초기화(first-touch) 효과
//     - 병렬 루프 오버헤드를 대역폭에 포함
//
//   그래서 여기서는:
//     * 스레드마다 **완전히 분리된 배열**을 malloc 한다 (false sharing 원천 차단)
//     * `pthread_setaffinity_np` 로 코어를 고정한다
//     * 스레드별 시작/종료 시각과 개별 대역폭을 모두 출력한다
//     * 각 스레드가 자기 배열을 **먼저 touch** 한 뒤 측정한다
//     * 총 바이트를 명시적으로 출력해 집계 오류를 눈으로 확인한다
//
// build: gcc -O2 -mcpu=native streamcheck.c -o streamcheck -lpthread
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static pthread_barrier_t g_bar;

typedef struct {
    int id, core; size_t n; int reps;
    double *a, *b, *c;
    double t0, t1; int coreRan;
} Arg;

static void *worker(void *p) {
    Arg *w = (Arg *)p;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(w->core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    // first touch: 자기 코어에서 자기 배열을 만진 뒤 측정한다
    for (size_t i = 0; i < w->n; i++) { w->a[i] = 0.0; w->b[i] = 1.0; w->c[i] = 2.0; }
    for (size_t i = 0; i < w->n; i++) w->a[i] = w->b[i] + 3.0 * w->c[i];   // 워밍
    pthread_barrier_wait(&g_bar);      // 모든 스레드가 같은 창에서 측정
    w->t0 = now();
    for (int r = 0; r < w->reps; r++)
        for (size_t i = 0; i < w->n; i++) w->a[i] = w->b[i] + 3.0 * w->c[i];
    w->t1 = now();
    w->coreRan = sched_getcpu();
    return NULL;
}

int main(int argc, char **argv) {
    const size_t n = (argc > 1 ? (size_t)atoi(argv[1]) : 4) << 20;   // 스레드당 elem
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    char host[128] = {0}; gethostname(host, sizeof(host)-1);
    printf("# host=%s  스레드당 %zu MB/배열 x3, reps=%d\n",
           host, n * sizeof(double) >> 20, reps);
    printf("# thr\ttotGB\twall_s\t집계GB/s\t스레드별GB/s\t실행코어\n");

    for (int T = 1; T <= 4; T++) {
        pthread_t th[4]; Arg w[4];
        for (int i = 0; i < T; i++) {
            w[i] = (Arg){ i, i, n, reps,
                          aligned_alloc(64, n*sizeof(double)),
                          aligned_alloc(64, n*sizeof(double)),
                          aligned_alloc(64, n*sizeof(double)), 0, 0, -1 };
            if (!w[i].a || !w[i].b || !w[i].c) { fprintf(stderr,"alloc 실패\n"); return 1; }
        }
        pthread_barrier_init(&g_bar, NULL, T);
        double s = now();
        for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, worker, &w[i]);
        for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
        pthread_barrier_destroy(&g_bar);
        // 공통 측정 창: 가장 늦은 시작 ~ 가장 이른 종료가 아니라,
        // 첫 시작 ~ 마지막 종료로 잡아 보수적으로(낮게) 집계한다.
        double lo = w[0].t0, hi = w[0].t1;
        for (int i = 1; i < T; i++) { if (w[i].t0 < lo) lo = w[i].t0; if (w[i].t1 > hi) hi = w[i].t1; }
        double wall = hi - lo;
        (void)s;
        // triad = 읽기 2 + 쓰기 1 = 24 B/elem
        double totB = (double)n * 24.0 * reps * T;
        printf("%4d\t%.2f\t%.3f\t%.2f\t", T, totB/1e9, wall, totB/wall/1e9);
        for (int i = 0; i < T; i++)
            printf("%.2f%s", (double)n*24.0*reps/(w[i].t1-w[i].t0)/1e9, i+1<T?",":"");
        printf("\t");
        for (int i = 0; i < T; i++) printf("%d%s", w[i].coreRan, i+1<T?",":"");
        printf("\n"); fflush(stdout);
        for (int i = 0; i < T; i++) { free(w[i].a); free(w[i].b); free(w[i].c); }
    }
    return 0;
}
