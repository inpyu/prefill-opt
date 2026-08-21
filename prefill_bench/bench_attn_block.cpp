// attention 내부 루프의 레지스터 블로킹 이득을 격리 측정한다.
//
// 현재 커널(nn-cpu-ops.cpp multiheadAttFused_F32)은 두 핵심 루프를 각각
//   (1) QK^T : (t,b) 조합마다 길이 headDim 내적 1회   -> 0.5 MAC/로드
//   (3) AV   : (t,b) 조합마다 길이 headDim axpy 1회   -> 0.5 MAC/로드
// 로 쓴다. 4x4 레지스터 타일로 바꾸면 8 로드당 16 MAC = 2.0 MAC/로드가 된다.
//
// 이 벤치는 그 구조 변경만 격리해서 재고, 두 결과가 수치적으로 같은지 확인한다.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static const int D  = 128;   // headDim
static const int BR = 32;    // 쿼리 타일
static const int TL = 128;   // KV 타일

// ---------- (1) QK^T ----------
static void qk_baseline(const float *Q, const float *K, float *S) {
    for (int t = 0; t < TL; t++) {
        const float *k = &K[(size_t)t * D];
        for (int b = 0; b < BR; b++) {
            const float *q = &Q[(size_t)b * D];
#if defined(__ARM_NEON)
            float32x4_t a = vdupq_n_f32(0.0f);
            for (int i = 0; i < D; i += 4)
                a = vmlaq_f32(a, vld1q_f32(&q[i]), vld1q_f32(&k[i]));
            float32x2_t lo = vadd_f32(vget_low_f32(a), vget_high_f32(a));
            S[(size_t)b * TL + t] = vget_lane_f32(lo, 0) + vget_lane_f32(lo, 1);
#else
            float a = 0.0f;
            for (int i = 0; i < D; i++) a += q[i] * k[i];
            S[(size_t)b * TL + t] = a;
#endif
        }
    }
}

static void qk_blocked(const float *Q, const float *K, float *S) {
#if defined(__ARM_NEON)
    for (int b0 = 0; b0 < BR; b0 += 4) {
        for (int t0 = 0; t0 < TL; t0 += 4) {
            float32x4_t a[4][4];
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) a[x][y] = vdupq_n_f32(0.0f);
            for (int i = 0; i < D; i += 4) {
                const float32x4_t q0 = vld1q_f32(&Q[(size_t)(b0+0)*D + i]);
                const float32x4_t q1 = vld1q_f32(&Q[(size_t)(b0+1)*D + i]);
                const float32x4_t q2 = vld1q_f32(&Q[(size_t)(b0+2)*D + i]);
                const float32x4_t q3 = vld1q_f32(&Q[(size_t)(b0+3)*D + i]);
                const float32x4_t k0 = vld1q_f32(&K[(size_t)(t0+0)*D + i]);
                const float32x4_t k1 = vld1q_f32(&K[(size_t)(t0+1)*D + i]);
                const float32x4_t k2 = vld1q_f32(&K[(size_t)(t0+2)*D + i]);
                const float32x4_t k3 = vld1q_f32(&K[(size_t)(t0+3)*D + i]);
                a[0][0]=vmlaq_f32(a[0][0],q0,k0); a[0][1]=vmlaq_f32(a[0][1],q0,k1);
                a[0][2]=vmlaq_f32(a[0][2],q0,k2); a[0][3]=vmlaq_f32(a[0][3],q0,k3);
                a[1][0]=vmlaq_f32(a[1][0],q1,k0); a[1][1]=vmlaq_f32(a[1][1],q1,k1);
                a[1][2]=vmlaq_f32(a[1][2],q1,k2); a[1][3]=vmlaq_f32(a[1][3],q1,k3);
                a[2][0]=vmlaq_f32(a[2][0],q2,k0); a[2][1]=vmlaq_f32(a[2][1],q2,k1);
                a[2][2]=vmlaq_f32(a[2][2],q2,k2); a[2][3]=vmlaq_f32(a[2][3],q2,k3);
                a[3][0]=vmlaq_f32(a[3][0],q3,k0); a[3][1]=vmlaq_f32(a[3][1],q3,k1);
                a[3][2]=vmlaq_f32(a[3][2],q3,k2); a[3][3]=vmlaq_f32(a[3][3],q3,k3);
            }
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) {
                float32x2_t lo = vadd_f32(vget_low_f32(a[x][y]), vget_high_f32(a[x][y]));
                S[(size_t)(b0+x)*TL + (t0+y)] = vget_lane_f32(lo,0) + vget_lane_f32(lo,1);
            }
        }
    }
#else
    qk_baseline(Q, K, S);
#endif
}

// ---------- (3) AV ----------
static void av_baseline(const float *P, const float *V, float *O) {
    for (int t = 0; t < TL; t++) {
        const float *v = &V[(size_t)t * D];
        for (int b = 0; b < BR; b++) {
            const float p = P[(size_t)b * TL + t];
            float *o = &O[(size_t)b * D];
#if defined(__ARM_NEON)
            const float32x4_t vp = vdupq_n_f32(p);
            for (int i = 0; i < D; i += 4)
                vst1q_f32(&o[i], vmlaq_f32(vld1q_f32(&o[i]), vp, vld1q_f32(&v[i])));
#else
            for (int i = 0; i < D; i++) o[i] += p * v[i];
#endif
        }
    }
}

static void av_blocked(const float *P, const float *V, float *O) {
#if defined(__ARM_NEON)
    // t 를 4개씩 묶어 O 를 레지스터에 붙잡아 둔다: 4 벡터 로드당 16 FMA.
    for (int b0 = 0; b0 < BR; b0 += 4) {
        for (int i = 0; i < D; i += 4) {
            float32x4_t o0 = vld1q_f32(&O[(size_t)(b0+0)*D + i]);
            float32x4_t o1 = vld1q_f32(&O[(size_t)(b0+1)*D + i]);
            float32x4_t o2 = vld1q_f32(&O[(size_t)(b0+2)*D + i]);
            float32x4_t o3 = vld1q_f32(&O[(size_t)(b0+3)*D + i]);
            for (int t = 0; t < TL; t += 4) {
                const float32x4_t v0 = vld1q_f32(&V[(size_t)(t+0)*D + i]);
                const float32x4_t v1 = vld1q_f32(&V[(size_t)(t+1)*D + i]);
                const float32x4_t v2 = vld1q_f32(&V[(size_t)(t+2)*D + i]);
                const float32x4_t v3 = vld1q_f32(&V[(size_t)(t+3)*D + i]);
                const float *p0 = &P[(size_t)(b0+0)*TL + t];
                const float *p1 = &P[(size_t)(b0+1)*TL + t];
                const float *p2 = &P[(size_t)(b0+2)*TL + t];
                const float *p3 = &P[(size_t)(b0+3)*TL + t];
                o0=vmlaq_n_f32(o0,v0,p0[0]); o0=vmlaq_n_f32(o0,v1,p0[1]);
                o0=vmlaq_n_f32(o0,v2,p0[2]); o0=vmlaq_n_f32(o0,v3,p0[3]);
                o1=vmlaq_n_f32(o1,v0,p1[0]); o1=vmlaq_n_f32(o1,v1,p1[1]);
                o1=vmlaq_n_f32(o1,v2,p1[2]); o1=vmlaq_n_f32(o1,v3,p1[3]);
                o2=vmlaq_n_f32(o2,v0,p2[0]); o2=vmlaq_n_f32(o2,v1,p2[1]);
                o2=vmlaq_n_f32(o2,v2,p2[2]); o2=vmlaq_n_f32(o2,v3,p2[3]);
                o3=vmlaq_n_f32(o3,v0,p3[0]); o3=vmlaq_n_f32(o3,v1,p3[1]);
                o3=vmlaq_n_f32(o3,v2,p3[2]); o3=vmlaq_n_f32(o3,v3,p3[3]);
            }
            vst1q_f32(&O[(size_t)(b0+0)*D + i], o0);
            vst1q_f32(&O[(size_t)(b0+1)*D + i], o1);
            vst1q_f32(&O[(size_t)(b0+2)*D + i], o2);
            vst1q_f32(&O[(size_t)(b0+3)*D + i], o3);
        }
    }
#else
    av_baseline(P, V, O);
#endif
}

// 실제 커널의 KV 접근은 연속이 아니다.
//   posK = &keyCache[headIndex*headDim + t*kvDim0],  kvDim0 = 1024 float = 4 kB
// 즉 t 가 1 늘 때마다 4 kB 를 건너뛰고 512 B 만 읽는다. 페이지당 1/8 만 쓰고
// 하드웨어 프리페처가 스트림으로 인식하기 어렵다.
static const int KVSTRIDE = 1024;

static void qk_blocked_strided(const float *Q, const float *Kst, float *S) {
#if defined(__ARM_NEON)
    for (int b0 = 0; b0 < BR; b0 += 4) {
        for (int t0 = 0; t0 < TL; t0 += 4) {
            float32x4_t a[4][4];
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) a[x][y] = vdupq_n_f32(0.0f);
            for (int i = 0; i < D; i += 4) {
                const float32x4_t q0 = vld1q_f32(&Q[(size_t)(b0+0)*D + i]);
                const float32x4_t q1 = vld1q_f32(&Q[(size_t)(b0+1)*D + i]);
                const float32x4_t q2 = vld1q_f32(&Q[(size_t)(b0+2)*D + i]);
                const float32x4_t q3 = vld1q_f32(&Q[(size_t)(b0+3)*D + i]);
                const float32x4_t k0 = vld1q_f32(&Kst[(size_t)(t0+0)*KVSTRIDE + i]);
                const float32x4_t k1 = vld1q_f32(&Kst[(size_t)(t0+1)*KVSTRIDE + i]);
                const float32x4_t k2 = vld1q_f32(&Kst[(size_t)(t0+2)*KVSTRIDE + i]);
                const float32x4_t k3 = vld1q_f32(&Kst[(size_t)(t0+3)*KVSTRIDE + i]);
                a[0][0]=vmlaq_f32(a[0][0],q0,k0); a[0][1]=vmlaq_f32(a[0][1],q0,k1);
                a[0][2]=vmlaq_f32(a[0][2],q0,k2); a[0][3]=vmlaq_f32(a[0][3],q0,k3);
                a[1][0]=vmlaq_f32(a[1][0],q1,k0); a[1][1]=vmlaq_f32(a[1][1],q1,k1);
                a[1][2]=vmlaq_f32(a[1][2],q1,k2); a[1][3]=vmlaq_f32(a[1][3],q1,k3);
                a[2][0]=vmlaq_f32(a[2][0],q2,k0); a[2][1]=vmlaq_f32(a[2][1],q2,k1);
                a[2][2]=vmlaq_f32(a[2][2],q2,k2); a[2][3]=vmlaq_f32(a[2][3],q2,k3);
                a[3][0]=vmlaq_f32(a[3][0],q3,k0); a[3][1]=vmlaq_f32(a[3][1],q3,k1);
                a[3][2]=vmlaq_f32(a[3][2],q3,k2); a[3][3]=vmlaq_f32(a[3][3],q3,k3);
            }
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) {
                float32x2_t lo = vadd_f32(vget_low_f32(a[x][y]), vget_high_f32(a[x][y]));
                S[(size_t)(b0+x)*TL + (t0+y)] = vget_lane_f32(lo,0) + vget_lane_f32(lo,1);
            }
        }
    }
#endif
}

// 쿼리 융합: 파이프라인 마이크로배치 G 개를 attention 커널에서만 합친다.
// 로드된 K 타일 하나를 G*BR 개 쿼리가 재사용한다.
static void qk_blocked_fused(const float *Q, const float *K, float *S, int G) {
#if defined(__ARM_NEON)
    const int BRG = BR * G;
    for (int t0 = 0; t0 < TL; t0 += 4) {
        for (int b0 = 0; b0 < BRG; b0 += 4) {
            float32x4_t a[4][4];
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) a[x][y] = vdupq_n_f32(0.0f);
            for (int i = 0; i < D; i += 4) {
                const float32x4_t q0 = vld1q_f32(&Q[(size_t)(b0+0)*D + i]);
                const float32x4_t q1 = vld1q_f32(&Q[(size_t)(b0+1)*D + i]);
                const float32x4_t q2 = vld1q_f32(&Q[(size_t)(b0+2)*D + i]);
                const float32x4_t q3 = vld1q_f32(&Q[(size_t)(b0+3)*D + i]);
                const float32x4_t k0 = vld1q_f32(&K[(size_t)(t0+0)*D + i]);
                const float32x4_t k1 = vld1q_f32(&K[(size_t)(t0+1)*D + i]);
                const float32x4_t k2 = vld1q_f32(&K[(size_t)(t0+2)*D + i]);
                const float32x4_t k3 = vld1q_f32(&K[(size_t)(t0+3)*D + i]);
                a[0][0]=vmlaq_f32(a[0][0],q0,k0); a[0][1]=vmlaq_f32(a[0][1],q0,k1);
                a[0][2]=vmlaq_f32(a[0][2],q0,k2); a[0][3]=vmlaq_f32(a[0][3],q0,k3);
                a[1][0]=vmlaq_f32(a[1][0],q1,k0); a[1][1]=vmlaq_f32(a[1][1],q1,k1);
                a[1][2]=vmlaq_f32(a[1][2],q1,k2); a[1][3]=vmlaq_f32(a[1][3],q1,k3);
                a[2][0]=vmlaq_f32(a[2][0],q2,k0); a[2][1]=vmlaq_f32(a[2][1],q2,k1);
                a[2][2]=vmlaq_f32(a[2][2],q2,k2); a[2][3]=vmlaq_f32(a[2][3],q2,k3);
                a[3][0]=vmlaq_f32(a[3][0],q3,k0); a[3][1]=vmlaq_f32(a[3][1],q3,k1);
                a[3][2]=vmlaq_f32(a[3][2],q3,k2); a[3][3]=vmlaq_f32(a[3][3],q3,k3);
            }
            for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) {
                float32x2_t lo = vadd_f32(vget_low_f32(a[x][y]), vget_high_f32(a[x][y]));
                S[(size_t)(b0+x)*TL + (t0+y)] = vget_lane_f32(lo,0) + vget_lane_f32(lo,1);
            }
        }
    }
#endif
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

static float maxRelDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const float d = std::fabs(a[i] - b[i]);
        const float s = std::fabs(a[i]) + std::fabs(b[i]) + 1e-30f;
        if (d / s > m) m = d / s;
    }
    return m;
}

int main() {
    std::vector<float> Q((size_t)BR*D), K((size_t)TL*D), V((size_t)TL*D), P((size_t)BR*TL);
    for (auto &x : Q) x = (float)drand48() - 0.5f;
    for (auto &x : K) x = (float)drand48() - 0.5f;
    for (auto &x : V) x = (float)drand48() - 0.5f;
    for (auto &x : P) x = (float)drand48();

    std::vector<float> S1((size_t)BR*TL), S2((size_t)BR*TL);
    std::vector<float> O1((size_t)BR*D, 0.0f), O2((size_t)BR*D, 0.0f);

    const int REP = 4000;
    // 정확성
    qk_baseline(Q.data(), K.data(), S1.data());
    qk_blocked(Q.data(), K.data(), S2.data());
    av_baseline(P.data(), V.data(), O1.data());
    av_blocked(P.data(), V.data(), O2.data());
    printf("정확성  QK maxRelDiff=%.3e   AV maxRelDiff=%.3e\n",
        maxRelDiff(S1,S2), maxRelDiff(O1,O2));

    // FLOPs: QK = BR*TL*D*2,  AV = BR*TL*D*2
    const double flopQK = (double)BR*TL*D*2.0*REP;
    const double flopAV = flopQK;

    double t0, t1;
    for (int w = 0; w < 200; w++) qk_baseline(Q.data(), K.data(), S1.data());
    t0 = now_ms(); for (int r = 0; r < REP; r++) qk_baseline(Q.data(), K.data(), S1.data()); t1 = now_ms();
    const double gQKb = flopQK / ((t1-t0)/1000.0) / 1e9;

    for (int w = 0; w < 200; w++) qk_blocked(Q.data(), K.data(), S2.data());
    t0 = now_ms(); for (int r = 0; r < REP; r++) qk_blocked(Q.data(), K.data(), S2.data()); t1 = now_ms();
    const double gQKt = flopQK / ((t1-t0)/1000.0) / 1e9;

    for (int w = 0; w < 200; w++) av_baseline(P.data(), V.data(), O1.data());
    t0 = now_ms(); for (int r = 0; r < REP; r++) av_baseline(P.data(), V.data(), O1.data()); t1 = now_ms();
    const double gAVb = flopAV / ((t1-t0)/1000.0) / 1e9;

    for (int w = 0; w < 200; w++) av_blocked(P.data(), V.data(), O2.data());
    t0 = now_ms(); for (int r = 0; r < REP; r++) av_blocked(P.data(), V.data(), O2.data()); t1 = now_ms();
    const double gAVt = flopAV / ((t1-t0)/1000.0) / 1e9;

    printf("QK^T   기존 %6.1f GFLOPS   블록 %6.1f GFLOPS   %.2fx\n", gQKb, gQKt, gQKt/gQKb);
    printf("AV     기존 %6.1f GFLOPS   블록 %6.1f GFLOPS   %.2fx\n", gAVb, gAVt, gAVt/gAVb);
    // 스트라이드 K (실제 커널의 접근 패턴)
    std::vector<float> Kst((size_t)TL*KVSTRIDE);
    for (int t = 0; t < TL; t++)
        for (int i = 0; i < D; i++) Kst[(size_t)t*KVSTRIDE + i] = K[(size_t)t*D + i];
    for (int w = 0; w < 200; w++) qk_blocked_strided(Q.data(), Kst.data(), S2.data());
    t0 = now_ms(); for (int r = 0; r < REP; r++) qk_blocked_strided(Q.data(), Kst.data(), S2.data()); t1 = now_ms();
    const double gQKs = flopQK / ((t1-t0)/1000.0) / 1e9;
    printf("QK^T   블록+스트라이드 %6.1f GFLOPS   (연속 대비 %.2fx)\n", gQKs, gQKs/gQKt);

    // 쿼리 융합 G=2,4
    for (int G = 2; G <= 4; G *= 2) {
        std::vector<float> Qg((size_t)BR*G*D), Sg((size_t)BR*G*TL);
        for (size_t i = 0; i < Qg.size(); i++) Qg[i] = (float)drand48() - 0.5f;
        const double flopG = (double)BR*G*TL*D*2.0*REP;
        for (int w = 0; w < 100; w++) qk_blocked_fused(Qg.data(), K.data(), Sg.data(), G);
        t0 = now_ms(); for (int r = 0; r < REP; r++) qk_blocked_fused(Qg.data(), K.data(), Sg.data(), G); t1 = now_ms();
        const double g = flopG / ((t1-t0)/1000.0) / 1e9;
        printf("QK^T   블록+융합 G=%d  %6.1f GFLOPS   (G=1 대비 %.2fx)\n", G, g, g/gQKt);
    }

    printf("합산   기존 %6.1f GFLOPS   블록 %6.1f GFLOPS   %.2fx\n",
        2.0/(1.0/gQKb + 1.0/gAVb), 2.0/(1.0/gQKt + 1.0/gAVt),
        (2.0/(1.0/gQKt + 1.0/gAVt)) / (2.0/(1.0/gQKb + 1.0/gAVb)));
    return 0;
}
