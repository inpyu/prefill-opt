#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <unordered_map>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
    #include <immintrin.h>
#endif
#include "nn-cpu-ops.hpp"
#include "nn-repack.hpp"
#include <atomic>
#include <thread>
#include "nn-quants.hpp"
#include "llamafile/sgemm.hpp"

#define DEBUG_OP_INPUT_OUTPUT false

static float dot_Q80_Q40_F32_row(const NnBlockQ80 *x, const NnBlockQ40 *wRow, const NnUint n) {
    assert(n % Q40_BLOCK_SIZE == 0);
    const NnUint nBlocks = n / Q40_BLOCK_SIZE;
    float sum = 0.0f;
    for (NnUint j = 0; j < nBlocks; j++) {
        const float s = CONVERT_F16_TO_F32(wRow[j].d) * CONVERT_F16_TO_F32(x[j].d);
        for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
            const int w0 = (wRow[j].qs[k] & 0x0F) - 8;
            const int w1 = (wRow[j].qs[k] >> 4) - 8;
            const int x0 = x[j].qs[k];
            const int x1 = x[j].qs[k + Q40_BLOCK_SIZE / 2];
            sum += (float)(w0 * x0 + w1 * x1) * s;
        }
    }
    return sum;
}

#if DEBUG_OP_INPUT_OUTPUT
    #define DEBUG_VECTOR(context, suffix, v) \
        if (threadIndex == 0) { \
            printf("%20s.%6s: ", context->name, suffix); \
            for (int k = 0; k < 12; k++) printf("%f ", v[k]); \
            printf("\n"); \
        }

    #define DEBUG_SCALAR(context, suffix, scalar) \
        if (threadIndex == 0) \
            printf("%20s.%6s: %f\n", context->name, suffix, scalar);
#else
    #define DEBUG_VECTOR(context, suffix, vec)
    #define DEBUG_SCALAR(context, suffix, scalar)
#endif

#if defined(__ARM_NEON)
static inline float32x4_t expf_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);

    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);

    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);
    float32x4_t p = c1;
    p = vaddq_f32(p, f);
    p = vaddq_f32(p, vmulq_f32(c2, f2));
    p = vaddq_f32(p, vmulq_f32(c3, f3));
    p = vaddq_f32(p, vmulq_f32(c4, f4));
    p = vaddq_f32(p, vmulq_f32(c5, f5));

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    float32x4_t two_k = vreinterpretq_f32_s32(pow2k);
    return vmulq_f32(p, two_k);
}
#endif

#if defined(__AVX2__)
static inline float horizontalSum_avx2(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float horizontalMax_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(v_low, v_high);
    __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    __m128 max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(max32);
}

static inline __m256 expf_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 n_float = _mm256_cvtepi32_ps(n);
    __m256 f = _mm256_sub_ps(y, n_float);
    __m256 p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 two_n = _mm256_castsi256_ps(exponent);
    return _mm256_mul_ps(p, two_n);
}
#endif

static float invRms_F32(const float *x, const unsigned int size, const float epsilon) {
    float sum;
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    sum = vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int j = 0; j < size; j += 8) {
        a = _mm256_loadu_ps(&x[j]);
        u = _mm256_fmadd_ps(a, a, u);
    }
    sum = horizontalSum_avx2(u);
#else
    sum = 0;
    for (unsigned int j = 0; j < size; j++) {
        sum += x[j] * x[j];
    }
#endif
    sum /= size;
    sum += epsilon;
    return 1.0f / sqrtf(sum);
}

static void rmsNorm_F32(float *output, const float *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(invRms);
    for (; i < neonEnd; i += 4) {
        fw = vld1q_f32(&w[i]);
        fx = vld1q_f32(&x[i]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&output[i], fx);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    const __m256 invRmsVec = _mm256_set1_ps(invRms);
    for (; i < avxEnd; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 wVec = _mm256_loadu_ps(&w[i]);
        __m256 scaledX = _mm256_mul_ps(xVec, invRmsVec);
        __m256 result = _mm256_mul_ps(scaledX, wVec);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++)
        output[i] = w[i] * (invRms * x[i]);
}

static void rmsNorm_Q80_F32_F32(float *output, const NnBlockQ80 *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    assert(size % Q80_BLOCK_SIZE == 0);
    const NnUint nBlocks = size / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnUint i = start; i < end; i++) {
        float d = CONVERT_F16_TO_F32(x[i].d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            output[k] = w[k] * (invRms * d * x[i].qs[j]);
        }
    }
}

static void matmul_F32_F32_F32(float *output, const float *x, const float *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    unsigned int i, j;
#if defined(__ARM_NEON)
    assert(n % 4 == 0);
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (i = start; i < end; i++) {
        z = vmovq_n_f32(0);
        for (j = 0; j < n; j += 4) {
            q = vld1q_f32(&x[j]);
            p = vld1q_f32(&w[i * n + j]);
            z = vfmaq_f32(z, q, p);
        }
        output[i] = vaddvq_f32(z);
    }
#elif defined(__AVX2__)
    assert(n % 8 == 0);
    __m256 a0, b0, u;
    for (i = start; i < end; i++) {
        u = _mm256_set1_ps(0.0f);
        for (j = 0; j < n; j += 8) {
            a0 = _mm256_loadu_ps(&x[j]);
            b0 = _mm256_loadu_ps(&w[i * n + j]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        output[i] = horizontalSum_avx2(u);
    }
#else
    for (i = start; i < end; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        output[i] = val;
    }
#endif
}

static void matmul_Q80_Q40_F32(float *output, const NnBlockQ80 *x, const NnBlockQ40 *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    assert(n % Q40_BLOCK_SIZE == 0);
    const unsigned int nBlocks = n / Q40_BLOCK_SIZE;

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    for (unsigned int di = start; di < end; di++) {
        float32x4_t sumv0 = vmovq_n_f32(0.0f);
        float32x4_t sumv1 = vmovq_n_f32(0.0f);
        float32x4_t sumv2 = vmovq_n_f32(0.0f);
        float32x4_t sumv3 = vmovq_n_f32(0.0f);

        unsigned int j = 0;
        
#if defined(__ARM_FEATURE_DOTPROD)
        for (; j + 3 < nBlocks; j += 4) {
            __builtin_prefetch(&w[di * nBlocks + j + 4]);
            __builtin_prefetch(&x[j + 4]);

            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ40 *w2 = &w[di * nBlocks + j + 2];
            const NnBlockQ40 *w3 = &w[di * nBlocks + j + 3];

            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];
            const NnBlockQ80 *x2 = &x[j + 2];
            const NnBlockQ80 *x3 = &x[j + 3];

            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w0->qs), m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w0->qs), 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w1->qs), m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w1->qs), 4)), s8b);
            int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w2->qs), m4b)), s8b);
            int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w2->qs), 4)), s8b);
            int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w3->qs), m4b)), s8b);
            int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w3->qs), 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);
            const int8x16_t x2l = vld1q_s8(x2->qs);
            const int8x16_t x2h = vld1q_s8(x2->qs + 16);
            const int8x16_t x3l = vld1q_s8(x3->qs);
            const int8x16_t x3h = vld1q_s8(x3->qs + 16);

            const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0l, x0l), w0h, x0h);
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1l, x1l), w1h, x1h);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2l, x2l), w2h, x2h);
            const int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3l, x3l), w3h, x3h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), CONVERT_F16_TO_F32(w2->d) * CONVERT_F16_TO_F32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), CONVERT_F16_TO_F32(w3->d) * CONVERT_F16_TO_F32(x3->d));
        }
#else
        for (; j + 1 < nBlocks; j += 2) {
            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];

            const uint8x16_t w0qs = vld1q_u8(w0->qs);
            const uint8x16_t w1qs = vld1q_u8(w1->qs);
            
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0qs, m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0qs, 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1qs, m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1qs, 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);

            const int16x8_t pl0l = vmull_s8(vget_low_s8(w0l), vget_low_s8(x0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(w0l), vget_high_s8(x0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8(w0h), vget_low_s8(x0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(w0h), vget_high_s8(x0h));
            
            const int16x8_t pl1l = vmull_s8(vget_low_s8(w1l), vget_low_s8(x1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(w1l), vget_high_s8(x1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8(w1h), vget_low_s8(x1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(w1h), vget_high_s8(x1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
        }
#endif

        for (; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[di * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];

            const uint8x16_t wqs = vld1q_u8(wb->qs);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wqs, m4b)), s8b);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(wqs, 4)), s8b);

            const int8x16_t xl = vld1q_s8(xb->qs);
            const int8x16_t xh = vld1q_s8(xb->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, xl), wh, xh);
#else
            const int16x8_t pll = vmull_s8(vget_low_s8(wl), vget_low_s8(xl));
            const int16x8_t plh = vmull_s8(vget_high_s8(wl), vget_high_s8(xl));
            const int16x8_t phl = vmull_s8(vget_low_s8(wh), vget_low_s8(xh));
            const int16x8_t phh = vmull_s8(vget_high_s8(wh), vget_high_s8(xh));
            
            const int32x4_t pl = vaddq_s32(vpaddlq_s16(pll), vpaddlq_s16(plh));
            const int32x4_t ph = vaddq_s32(vpaddlq_s16(phl), vpaddlq_s16(phh));
            const int32x4_t p = vaddq_s32(pl, ph);
#endif
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }

        output[di] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#elif defined(__AVX512F__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w8 = _mm_loadu_si128((const __m128i*)wb->qs);
            __m128i v_w0 = _mm_and_si128(w8, _mm_set1_epi8(0x0F));
            __m128i v_w1 = _mm_srli_epi16(w8, 4);
            v_w1 = _mm_and_si128(v_w1, _mm_set1_epi8(0x0F));
            
            v_w0 = _mm_sub_epi8(v_w0, _mm_set1_epi8(8));
            v_w1 = _mm_sub_epi8(v_w1, _mm_set1_epi8(8));

            __m256i w8_combined = _mm256_set_m128i(v_w1, v_w0);
            __m512i w16 = _mm512_cvtepi8_epi16(w8_combined);

            __m256i x8 = _mm256_loadu_si256((const __m256i*)xb->qs);
            __m512i x16 = _mm512_cvtepi8_epi16(x8);

            __m512i products = _mm512_madd_epi16(w16, x16);
            sum += _mm512_reduce_add_epi32(products) * s;
        }
        output[i] = sum;
    }
#elif defined(__AVX2__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w_packed = _mm_loadu_si128((const __m128i*)wb->qs);

            __m128i w0_low = _mm_and_si128(w_packed, _mm_set1_epi8(0x0F));
            __m128i w0 = _mm_sub_epi8(w0_low, _mm_set1_epi8(8));

            __m128i w1_high = _mm_srli_epi16(w_packed, 4);
            w1_high = _mm_and_si128(w1_high, _mm_set1_epi8(0x0F));
            __m128i w1 = _mm_sub_epi8(w1_high, _mm_set1_epi8(8));

            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

            __m128i i1_8 = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i i2_8 = _mm_loadu_si128((const __m128i*)(xb->qs + Q40_BLOCK_SIZE / 2));

            __m256i i1_16 = _mm256_cvtepi8_epi16(i1_8);
            __m256i i2_16 = _mm256_cvtepi8_epi16(i2_8);

            __m256i prod0 = _mm256_mullo_epi16(w0_16, i1_16);
            __m256i prod1 = _mm256_mullo_epi16(w1_16, i2_16);
            __m256i sum_prod = _mm256_add_epi16(prod0, prod1);

            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(sum_prod, ones);

            __m128i sum_low = _mm256_castsi256_si128(sum32);
            __m128i sum_high = _mm256_extracti128_si256(sum32, 1);
            sum_low = _mm_add_epi32(sum_low, sum_high);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            int32_t block_sum = _mm_extract_epi32(sum_low, 0);

            sum += block_sum * s;
        }
        output[i] = sum;
    }
#else
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (wb->qs[k] & 0x0F) - 8;
                const int w1 = (wb->qs[k] >> 4) - 8;
                const int i1 = xb->qs[k];
                const int i2 = xb->qs[k + Q80_BLOCK_SIZE / 2];
                sum += (w0 * i1 + w1 * i2) * s;
            }
        }
        output[i] = sum;
    }
#endif
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void silu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);

    for (; i < neonEnd; i += 4) {
        float32x4_t x = vld1q_f32(&output[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = expf_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);

    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i < avxEnd; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(output + i);
        __m256 neg_x = _mm256_sub_ps(zero, x_vec);
        __m256 exp_negx = expf_avx2(neg_x);
        __m256 denominator = _mm256_add_ps(ones, exp_negx);
        __m256 result = _mm256_div_ps(x_vec, denominator);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

static void add_F32(float *output, const float *x, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        output[i] += x[i];
    }
}

static void add_Q80_F32(float *y, const NnBlockQ80 *x, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        float *y_base = y + i * Q80_BLOCK_SIZE;
        const int8_t *qs = xi->qs;

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 16) {
            // Load 16x 8-bit quantized values
            const int8x16_t q8 = vld1q_s8(qs + j);
            
            // Split into 8-bit high/low components
            const int8x8_t q8_low = vget_low_s8(q8);
            const int8x8_t q8_high = vget_high_s8(q8);
            
            // Sign extend to 16-bit
            const int16x8_t q16_low = vmovl_s8(q8_low);
            const int16x8_t q16_high = vmovl_s8(q8_high);
            
            // Sign extend to 32-bit and convert to float
            const float32x4_t qf_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_low)));
            const float32x4_t qf_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_low)));
            const float32x4_t qf_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_high)));
            const float32x4_t qf_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_high)));
            
            // Multiply by scale factor
            const float32x4_t sf_ll = vmulq_n_f32(qf_ll, xid);
            const float32x4_t sf_lh = vmulq_n_f32(qf_lh, xid);
            const float32x4_t sf_hl = vmulq_n_f32(qf_hl, xid);
            const float32x4_t sf_hh = vmulq_n_f32(qf_hh, xid);
            
            // Load existing y values
            float32x4_t y_ll = vld1q_f32(y_base + j);
            float32x4_t y_lh = vld1q_f32(y_base + j + 4);
            float32x4_t y_hl = vld1q_f32(y_base + j + 8);
            float32x4_t y_hh = vld1q_f32(y_base + j + 12);
            
            // Accumulate results
            y_ll = vaddq_f32(y_ll, sf_ll);
            y_lh = vaddq_f32(y_lh, sf_lh);
            y_hl = vaddq_f32(y_hl, sf_hl);
            y_hh = vaddq_f32(y_hh, sf_hh);
            
            // Store results back
            vst1q_f32(y_base + j, y_ll);
            vst1q_f32(y_base + j + 4, y_lh);
            vst1q_f32(y_base + j + 8, y_hl);
            vst1q_f32(y_base + j + 12, y_hh);
        }
    }
#elif defined(__AVX2__)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 8) {
            __m128i i8_vec = _mm_loadl_epi64((const __m128i*)(xi->qs + j));

            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            __m256 f_vec = _mm256_cvtepi32_ps(i32_vec);

            __m256 scale = _mm256_set1_ps(xid);
            __m256 scaled = _mm256_mul_ps(f_vec, scale);

            float* y_ptr = y + i * Q80_BLOCK_SIZE + j;
            __m256 y_vec = _mm256_loadu_ps(y_ptr);
            y_vec = _mm256_add_ps(y_vec, scaled);
            _mm256_storeu_ps(y_ptr, y_vec);
        }
    }
#else
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] += xid * xi->qs[j];
        }
    }
#endif
}

void softmax_F32(float *x, const NnUint size) {
    if (size == 0)
        return;

#if defined(__ARM_NEON)
    NnUint j;
    float maxVal;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (NnUint i = 4; i < j; i += 4) {
            fs = vld1q_f32(&x[i]);
            fmaxv = vmaxq_f32(fmaxv, fs);
        }
        maxVal = vmaxvq_f32(fmaxv);
    } else {
        maxVal = x[0];
        j = 1;
    }
    for (; j < size; j++)
        maxVal = fmaxf(maxVal, x[j]);

    const float32x4_t maxVal_vec = vdupq_n_f32(maxVal);
    float32x4_t sumv = vdupq_n_f32(0.0f);
    NnUint i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vsubq_f32(val, maxVal_vec);
        val = expf_neon(val);
        vst1q_f32(x + i, val);
        sumv = vaddq_f32(sumv, val);
    }

    float32x2_t sum_lo = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
    float sum = vget_lane_f32(sum_lo, 0) + vget_lane_f32(sum_lo, 1);

    for (; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vmulq_f32(val, inv_sum_vec);
        vst1q_f32(x + i, val);
    }
    for (; i < size; ++i)
        x[i] /= sum;
#elif defined(__AVX2__)
    float maxVal;
    const unsigned avxEnd = size - (size % 8);
    NnUint i = 0;

    if (avxEnd >= 8) {
        __m256 max_vec = _mm256_loadu_ps(x);
        i = 8;
        for (; i < avxEnd; i += 8) {
            __m256 vec = _mm256_loadu_ps(&x[i]);
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        maxVal = horizontalMax_avx2(max_vec);
    } else {
        maxVal = x[0];
        i = 1;
    }
    for (; i < size; ++i) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }

    __m256 max_val_vec = _mm256_set1_ps(maxVal);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(&x[i]);
        vec = _mm256_sub_ps(vec, max_val_vec);
        vec = expf_avx2(vec);
        _mm256_storeu_ps(&x[i], vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    sum = horizontalSum_avx2(sum_vec);
    for (; i < size; ++i) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(x + i);
        vec = _mm256_mul_ps(vec, inv_sum_vec);
        _mm256_storeu_ps(x + i, vec);
    }
    for (; i < size; i++)
        x[i] *= inv_sum;
#else
    float maxVal = x[0];
    for (NnUint i = 1; i < size; i++) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }
    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0)
        sum = 0.000001;
    for (NnUint i = 0; i < size; i++)
        x[i] /= sum;
#endif
}

static float dotProduct_F32(const float *a, const float *b, const unsigned int size) {
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fa;
    float32x4_t fb;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int i = 0; i < size; i += 4) {
        fa = vld1q_f32(&a[i]);
        fb = vld1q_f32(&b[i]);
        fs = vmlaq_f32(fs, fa, fb);
    }
    return vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a0, b0;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int i = 0; i < size; i += 8) {
        a0 = _mm256_loadu_ps(&a[i]);
        b0 = _mm256_loadu_ps(&b[i]);
        u = _mm256_fmadd_ps(a0, b0, u);
    }
    return horizontalSum_avx2(u);
#else
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

static void multiheadAtt_F32(
    float *y, const float *q, float *att, float *keyCache, float *valueCache,
    const NnUint pos, const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads, const NnUint kvDim0, const NnUint headDim, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);

    for (NnUint h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headDim];
        const NnUint headIndex = h0 / kvMul;
        const float *hKc = &keyCache[headIndex * headDim];
        const float *hVc = &valueCache[headIndex * headDim];
        float *hAtt = &att[h0 * seqLen];

        for (NnUint t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvDim0];
            const float score = dotProduct_F32(hQ, posK, headDim) / headDimRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hY = &y[h0 * headDim];
        std::memset(hY, 0, headDim * sizeof(float));

        for (NnUint t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvDim0];
            const float posA = hAtt[t];
            for (int i = 0; i < headDim; i++) {
                hY[i] += posA * posV[i];
            }
        }
    }
}

// 블록 병렬 시뮬레이션 설정 (research/07 Phase A1)
static std::atomic<NnUint> gBlockSize{0};
static std::atomic<NnUint> gAnchorLen{0};

void nnCpuOpsSetBlockMask(NnUint blockSize, NnUint anchorLen) {
    gBlockSize.store(blockSize, std::memory_order_relaxed);
    gAnchorLen.store(anchorLen, std::memory_order_relaxed);
}

// 쿼리 위치 q 가 키 위치 t 를 볼 수 있는가.
//
// 블록 병렬에서 각 노드는 자기 블록만 갖고 있으므로, 쿼리는
//   [0, anchorLen)  ∪  [자기 블록 시작, q]
// 만 참조할 수 있다. 그 밖은 다른 노드에 있어 보이지 않는다.
static inline bool blockMaskAllows(NnUint q, NnUint t) {
    const NnUint bs = gBlockSize.load(std::memory_order_relaxed);
    if (bs == 0u)
        return true;                       // 비활성: 평소대로 causal
    if (t < gAnchorLen.load(std::memory_order_relaxed))
        return true;                       // anchor 는 모두가 본다
    return (t / bs) == (q / bs);           // 같은 블록 안
}

// Prefill 배치 어텐션.
//
// 기존 multiheadAtt_F32 는 쿼리 위치 1개를 전제로 만들어진 decode용 커널이고,
// prefill 은 이를 배치 크기만큼 반복 호출한다. 그 결과 KV 캐시를
//   (쿼리 토큰 수) x (kvMul) 번
// 반복해서 스트리밍하게 되어, 긴 프롬프트에서 attention 이 memory-bandwidth bound 가 된다.
// (S=1789 실측에서 attention 이 prefill 시간의 57% 를 차지)
//
// 여기서는 두 가지를 바꾼다. 연산 순서는 그대로라 수치 결과는 동일하다.
//   (A) GQA 그룹화: 같은 KV 헤드를 공유하는 kvMul 개 쿼리 헤드를 함께 처리 → KV 읽기 1/kvMul
//   (B) 쿼리 타일링: t 루프를 바깥으로 빼 배치 전체가 KV 한 번 읽기를 공유    → KV 읽기 1/batchSize
//
// 스레드 분할은 KV 그룹 수가 스레드 수 이상일 때만 그룹 단위로 하고(=A 적용),
// 그렇지 않으면 기존처럼 헤드 단위로 나눈다(=B 만 적용). TP 로 헤드가 쪼개진
// 구성에서 스레드가 놀지 않도록 하기 위함이다.
static void multiheadAttBatch_F32(
    NnByte **outputs, const float *query, const NnUint qSliceD0,
    float *att, const float *keyCache, const float *valueCache,
    const float *positions, const NnUint batchSize,
    const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads,
    const NnUint kvDim0, const NnUint headDim, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex)
{
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);
    const NnUint nGroups = (kvMul > 0u && nHeads0 % kvMul == 0u) ? (nHeads0 / kvMul) : 0u;
    const bool groupSplit = nGroups >= nThreads && nGroups > 0u;

    NnUint h0Begin, h0Limit, headStride;
    if (groupSplit) {
        SPLIT_THREADS(gStart, gEnd, nGroups, nThreads, threadIndex);
        h0Begin = gStart * kvMul;
        h0Limit = gEnd * kvMul;
        headStride = kvMul;              // kvMul 개씩 묶어서 순회
    } else {
        SPLIT_THREADS(hStart, hEnd, nHeads0, nThreads, threadIndex);
        h0Begin = hStart;
        h0Limit = hEnd;
        headStride = 1u;
    }

    // 이 배치에서 참조해야 하는 가장 먼 위치
    NnUint maxPos = 0u;
    for (NnUint b = 0; b < batchSize; b++) {
        const NnUint p = (NnUint)positions[b];
        if (p > maxPos)
            maxPos = p;
    }

    for (NnUint h0Base = h0Begin; h0Base < h0Limit; h0Base += headStride) {
        const NnUint nHeadsInGroup = (h0Base + headStride <= h0Limit)
            ? headStride
            : (h0Limit - h0Base);
        const NnUint headIndex = h0Base / kvMul;
        const float *hKc = &keyCache[headIndex * headDim];
        const float *hVc = &valueCache[headIndex * headDim];

        // (1) score: t 를 바깥 루프로 두어 posK 를 한 번만 읽는다.
        for (NnUint t = 0; t <= maxPos; t++) {
            const float *posK = &hKc[t * kvDim0];
            for (NnUint j = 0; j < nHeadsInGroup; j++) {
                const NnUint h0 = h0Base + j;
                for (NnUint b = 0; b < batchSize; b++) {
                    const NnUint qPos = (NnUint)positions[b];
                    if (t > qPos)
                        continue;             // causal mask
                    if (!blockMaskAllows(qPos, t))
                        continue;             // 블록 병렬 마스크 (Phase A1)
                    const float *hQ = &query[b * qSliceD0 + h0 * headDim];
                    att[(b * nHeads0 + h0) * seqLen + t] =
                        dotProduct_F32(hQ, posK, headDim) / headDimRoot;
                }
            }
        }

        // (2) softmax: (batch, head) 별로 자기 위치까지
        for (NnUint j = 0; j < nHeadsInGroup; j++) {
            const NnUint h0 = h0Base + j;
            for (NnUint b = 0; b < batchSize; b++)
                softmax_F32(&att[(b * nHeads0 + h0) * seqLen], (NnUint)positions[b] + 1u);
        }

        // (3) 출력 누적 전 0으로 초기화
        for (NnUint j = 0; j < nHeadsInGroup; j++) {
            const NnUint h0 = h0Base + j;
            for (NnUint b = 0; b < batchSize; b++)
                std::memset(&((float *)outputs[b])[h0 * headDim], 0, headDim * sizeof(float));
        }

        // (4) attention x V: 여기서도 t 를 바깥으로 두어 posV 를 한 번만 읽는다.
        for (NnUint t = 0; t <= maxPos; t++) {
            const float *posV = &hVc[t * kvDim0];
            for (NnUint j = 0; j < nHeadsInGroup; j++) {
                const NnUint h0 = h0Base + j;
                for (NnUint b = 0; b < batchSize; b++) {
                    const NnUint qPos = (NnUint)positions[b];
                    if (t > qPos)
                        continue;
                    if (!blockMaskAllows(qPos, t))
                        continue;
                    const float posA = att[(b * nHeads0 + h0) * seqLen + t];
                    float *hY = &((float *)outputs[b])[h0 * headDim];
#if defined(__ARM_NEON)
                    const float32x4_t va = vdupq_n_f32(posA);
                    NnUint i = 0;
                    for (; i + 4 <= headDim; i += 4)
                        vst1q_f32(&hY[i], vmlaq_f32(vld1q_f32(&hY[i]), va, vld1q_f32(&posV[i])));
                    for (; i < headDim; i++)
                        hY[i] += posA * posV[i];
#else
                    for (NnUint i = 0; i < headDim; i++)
                        hY[i] += posA * posV[i];
#endif
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 온라인 소프트맥스 융합 attention (FlashAttention 계열)
//
// multiheadAttBatch_F32 는 점수 행렬 att 를 (batchSize x nHeads0 x seqLen) 로
// 통째로 실체화한다. 이 버퍼가 Pi5 의 L3 2 MB 를 넘어가는 순간 매 청크가 DRAM
// 왕복이 되고, 그것이 prefill 청크 폭 B 의 천장이었다 (research/06 §4.7):
//
//     B      attn(ms)      att 버퍼
//     32       2,548        2.8 MB
//     112      9,340        9.6 MB
//     448     43,838       38.5 MB
//
// 여기서는 키를 TILE 개씩 끊어 처리하며 (running max m, running sum l) 을
// 유지한다. 점수 스크래치가 seqLen 이 아니라 TILE 에 비례하므로 위 천장이 사라진다.
//
//     m_new = max(m_old, max_t s_t)
//     alpha = exp(m_old - m_new)          // 이전 누적분 재스케일 계수
//     l     = alpha*l + sum_t exp(s_t - m_new)
//     o     = alpha*o + sum_t exp(s_t - m_new) * V_t
//     최종   o /= l
//
// 수학적으로는 기존 커널과 같은 값이지만 부동소수점 연산 순서가 달라지므로
// bit-exact 하지는 않다. 검증은 greedy(temperature 0) 출력 토큰 일치로 한다.
//
// 타일 안에서 t 를 바깥 루프로 두는 구조는 기존 커널에서 그대로 가져왔다.
// K/V 의 한 위치를 읽어 배치 전체가 재사용하는 것이 이 커널의 핵심 국소성이다.
static void multiheadAttFused_F32(
    NnByte **outputs,
    const float *query, const NnUint qSliceD0,
    const float *keyCache, const float *valueCache,
    const float *positions, const NnUint batchSize,
    const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads,
    const NnUint kvDim0, const NnUint headDim, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex)
{
    (void)seqLen;
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);
    const NnUint nGroups = (kvMul > 0u && nHeads0 % kvMul == 0u) ? (nHeads0 / kvMul) : 0u;
    const bool groupSplit = nGroups >= nThreads && nGroups > 0u;

    NnUint h0Begin, h0Limit, headStride;
    if (groupSplit) {
        SPLIT_THREADS(gStart, gEnd, nGroups, nThreads, threadIndex);
        h0Begin = gStart * kvMul;
        h0Limit = gEnd * kvMul;
        headStride = kvMul;
    } else {
        SPLIT_THREADS(hStart, hEnd, nHeads0, nThreads, threadIndex);
        h0Begin = hStart;
        h0Limit = hEnd;
        headStride = 1u;
    }

    // 타일 폭. B=32 / kvMul=4 기준 스크래치는 32*4*128*4B = 64 kB 로 L2(코어당
    // 512 kB) 안에 들어온다. seqLen 과 무관하다는 점이 이 커널의 요지다.
    const NnUint TILE = 128u;


    // 스레드마다 자기 헤드 묶음만 다루므로 thread_local 로 충분하다.
    thread_local std::vector<float> scratch;   // 점수 -> 확률 타일
    thread_local std::vector<float> stateM;    // running max
    thread_local std::vector<float> stateL;    // running sum

    for (NnUint h0Base = h0Begin; h0Base < h0Limit; h0Base += headStride) {
        const NnUint nHeadsInGroup = (h0Base + headStride <= h0Limit)
            ? headStride
            : (h0Limit - h0Base);
        const NnUint headIndex = h0Base / kvMul;
        const float *hKc = &keyCache[headIndex * headDim];
        const float *hVc = &valueCache[headIndex * headDim];

        // 쿼리도 타일링한다 (FlashAttention 의 바깥 루프).
        //
        // t 가 바깥 루프라 매 t 마다 배치 전체의 쿼리/누적기/스크래치를 다시 훑는다.
        // 그 워킹셋은 B 에 비례하므로(헤드 그룹당 약 B x 6 kB) B 가 커지면 코어당
        // L2 512 kB 를 넘어 매 t 가 DRAM 왕복이 된다. KV 만 타일링해서는 사라지지
        // 않는다 — B=112 에서 att 실체화를 없앴는데도 9,340 -> 9,338 ms 로 무변화였다.
        // BR 개씩 끊으면 워킹셋이 B 와 무관해진다.
        const NnUint BR = 32u;
        for (NnUint bBase = 0; bBase < batchSize; bBase += BR) {
        const NnUint bCount = std::min(BR, batchSize - bBase);

        // 이 타일에서 참조해야 하는 가장 먼 위치. 타일별로 잡으면 causal 상한이
        // 좁아져 뒤쪽 타일이 앞쪽 타일의 위치까지 훑는 낭비가 없다.
        NnUint maxPos = 0u;
        for (NnUint b = 0; b < bCount; b++) {
            const NnUint p = (NnUint)positions[bBase + b];
            if (p > maxPos)
                maxPos = p;
        }

        const NnUint nSlots = bCount * nHeadsInGroup;
        if (scratch.size() < (std::size_t)nSlots * TILE)
            scratch.resize((std::size_t)nSlots * TILE);
        stateM.assign(nSlots, -std::numeric_limits<float>::infinity());
        stateL.assign(nSlots, 0.0f);

        for (NnUint j = 0; j < nHeadsInGroup; j++) {
            for (NnUint b = 0; b < bCount; b++)
                std::memset(&((float *)outputs[bBase + b])[(h0Base + j) * headDim], 0, headDim * sizeof(float));
        }

        for (NnUint t0 = 0; t0 <= maxPos; t0 += TILE) {
            const NnUint t1 = std::min(t0 + TILE, maxPos + 1u);

            // (1) 점수. t 를 바깥에 두어 K 의 한 위치를 배치 전체가 재사용한다.
            for (NnUint t = t0; t < t1; t++) {
                const float *posK = &hKc[t * kvDim0];
                for (NnUint j = 0; j < nHeadsInGroup; j++) {
                    const NnUint h0 = h0Base + j;
                    for (NnUint b = 0; b < bCount; b++) {
                        const NnUint qPos = (NnUint)positions[bBase + b];
                        if (t > qPos)
                            continue;                       // causal mask
                        if (!blockMaskAllows(qPos, t))
                            continue;                       // 블록 병렬 마스크
                        const float *hQ = &query[(bBase + b) * qSliceD0 + h0 * headDim];
                        scratch[(std::size_t)(j * bCount + b) * TILE + (t - t0)] =
                            dotProduct_F32(hQ, posK, headDim) / headDimRoot;
                    }
                }
            }

            // (2) 온라인 소프트맥스 갱신. 이 타일에서 (b,h) 별 max/sum 을 합치고
            //     이전 누적분 o 를 alpha 로 재스케일한다.
            for (NnUint j = 0; j < nHeadsInGroup; j++) {
                const NnUint h0 = h0Base + j;
                for (NnUint b = 0; b < bCount; b++) {
                    const NnUint slot = j * bCount + b;
                    float *tile = &scratch[(std::size_t)slot * TILE];
                    const NnUint qPos = (NnUint)positions[bBase + b];

                    if (t0 > qPos) {
                        // 이 타일 전체가 미래다. (3)이 스크래치를 읽으므로 0 으로 지운다.
                        std::memset(tile, 0, (t1 - t0) * sizeof(float));
                        continue;
                    }

                    float tileMax = -std::numeric_limits<float>::infinity();
                    for (NnUint t = t0; t < t1; t++) {
                        if (t > qPos || !blockMaskAllows(qPos, t))
                            continue;
                        const float v = tile[t - t0];
                        if (v > tileMax)
                            tileMax = v;
                    }
                    if (tileMax == -std::numeric_limits<float>::infinity()) {
                        std::memset(tile, 0, (t1 - t0) * sizeof(float));
                        continue;                            // 전부 마스크된 타일
                    }

                    const float mOld = stateM[slot];
                    const float mNew = (mOld > tileMax) ? mOld : tileMax;
                    const float alpha = (mOld == -std::numeric_limits<float>::infinity())
                        ? 0.0f
                        : expf(mOld - mNew);

                    if (alpha != 1.0f) {
                        stateL[slot] *= alpha;
                        float *hY = &((float *)outputs[bBase + b])[h0 * headDim];
#if defined(__ARM_NEON)
                        const float32x4_t va = vdupq_n_f32(alpha);
                        NnUint i = 0;
                        for (; i + 4 <= headDim; i += 4)
                            vst1q_f32(&hY[i], vmulq_f32(vld1q_f32(&hY[i]), va));
                        for (; i < headDim; i++)
                            hY[i] *= alpha;
#else
                        for (NnUint i = 0; i < headDim; i++)
                            hY[i] *= alpha;
#endif
                    }

                    float sum = 0.0f;
                    for (NnUint t = t0; t < t1; t++) {
                        if (t > qPos || !blockMaskAllows(qPos, t)) {
                            tile[t - t0] = 0.0f;
                            continue;
                        }
                        const float p = expf(tile[t - t0] - mNew);
                        tile[t - t0] = p;
                        sum += p;
                    }
                    stateM[slot] = mNew;
                    stateL[slot] += sum;
                }
            }

            // (3) o += p * V. 여기서도 t 를 바깥에 두어 V 의 한 위치를 재사용한다.
            for (NnUint t = t0; t < t1; t++) {
                const float *posV = &hVc[t * kvDim0];
                for (NnUint j = 0; j < nHeadsInGroup; j++) {
                    const NnUint h0 = h0Base + j;
                    for (NnUint b = 0; b < bCount; b++) {
                        const float p = scratch[(std::size_t)(j * bCount + b) * TILE + (t - t0)];
                        if (p == 0.0f)
                            continue;
                        float *hY = &((float *)outputs[bBase + b])[h0 * headDim];
#if defined(__ARM_NEON)
                        const float32x4_t va = vdupq_n_f32(p);
                        NnUint i = 0;
                        for (; i + 4 <= headDim; i += 4)
                            vst1q_f32(&hY[i], vmlaq_f32(vld1q_f32(&hY[i]), va, vld1q_f32(&posV[i])));
                        for (; i < headDim; i++)
                            hY[i] += p * posV[i];
#else
                        for (NnUint i = 0; i < headDim; i++)
                            hY[i] += p * posV[i];
#endif
                    }
                }
            }
        }

        // 정규화: 지금까지 o 는 분자만 누적돼 있다.
        for (NnUint j = 0; j < nHeadsInGroup; j++) {
            const NnUint h0 = h0Base + j;
            for (NnUint b = 0; b < bCount; b++) {
                const float l = stateL[j * bCount + b];
                if (l <= 0.0f)
                    continue;
                const float inv = 1.0f / l;
                float *hY = &((float *)outputs[bBase + b])[h0 * headDim];
#if defined(__ARM_NEON)
                const float32x4_t vi = vdupq_n_f32(inv);
                NnUint i = 0;
                for (; i + 4 <= headDim; i += 4)
                    vst1q_f32(&hY[i], vmulq_f32(vld1q_f32(&hY[i]), vi));
                for (; i < headDim; i++)
                    hY[i] *= inv;
#else
                for (NnUint i = 0; i < headDim; i++)
                    hY[i] *= inv;
#endif
            }
        }
        } // b-타일
    }
}

static void mul_F32(float *y, const float *x, const float *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;

#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 8);
    for (; i < neonEnd; i += 4) {
        float32x4_t out_vec = vld1q_f32(&x[i]);
        float32x4_t x_vec = vld1q_f32(&m[i]);
        float32x4_t res_vec = vmulq_f32(out_vec, x_vec);
        vst1q_f32(&y[i], res_vec);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    for (; i < avxEnd; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&x[i]);
        __m256 x_vec = _mm256_loadu_ps(&m[i]);
        __m256 res_vec = _mm256_mul_ps(out_vec, x_vec);
        _mm256_storeu_ps(&y[i], res_vec);
    }
#endif
    for (; i < end; i++)
        y[i] = x[i] * m[i];
}

static void scale_F32(const float *i, float *o, const float s, NnSize size, NnUint nThreads, NnUint threadIndex) {
    for (NnUint x = threadIndex; x < size; x += nThreads)
        o[x] = i[x] * s;
}

static void mul_Q80_F32(float *y, const float *x, const NnBlockQ80 *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        const NnBlockQ80 *b = &m[i];
        float d = CONVERT_F16_TO_F32(b->d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            y[k] = x[k] * d * b->qs[j];
        }
    }
}

static void copy_UNK(NnByte *output, const NnByte *x, NnSize size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    NnUint s = end - start;
    if (s != 0)
        std::memcpy(&output[start], &x[start], s);
}


static void ropeLlama_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint dim0Half = (isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const NnUint shift = isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const NnUint iStart = s * 2;
    const NnUint iEnd = e * 2;

    const float *posCache = &cache[pos * slice->sliceDim + shift];

    for (NnUint i = iStart; i < iEnd; i += 2) {
        const float fcr = posCache[i];
        const float fci = posCache[i + 1];
        const float v0 = x[i];
        const float v1 = x[i + 1];

        float x0 = v0 * fcr - v1 * fci;
        float x1 = v0 * fci + v1 * fcr;
        x[i] = x0;
        x[i + 1] = x1;
    }
}

static void ropeFalcon_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    unsigned int dim0 =  isQ ? slice->qDim0 : slice->kvDim0;
    assert(dim0 % slice->headDim == 0);
    unsigned int nHeads0 = dim0 / slice->headDim;
    SPLIT_THREADS(h0s, h0e, nHeads0, nThreads, threadIndex);

    const float *posCache = &cache[pos * slice->headDim];

    for (unsigned int h = h0s; h < h0e; h++) {
        const unsigned int o = h * slice->headDim;
        for (unsigned int j = 0; j < slice->headDim / 2; j++) {
            const float fcr0 = posCache[j];
            const float fci0 = posCache[j + slice->headDim / 2];

            float q0 = x[o + j];
            float q1 = x[o + j + slice->headDim / 2];
            x[o + j] = q0 * fcr0 - q1 * fci0;
            x[o + j + slice->headDim / 2] = q0 * fci0 + q1 * fcr0;
        }
    }
}

static void mergeSum_F32(float **output, float **input, const NnSize size, const NnSize batchSize, const NnSize nBatches, const NnSize nZ, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);

    for (NnUint y = 0u; y < batchSize; y++) {
        for (NnUint x = start; x < end; x++) {
            float s = 0.0f;
            for (NnUint z = 0u; z < nZ; z++)
                s += input[y + z * nBatches][x];
            output[y][x] = s;
        }
    }
}

static void topk_F32(const float *x, NnUint *y, NnSize size, NnUint k) {
    assert(k <= size);
    assert(k > 0u);

    std::vector<NnUint> items(size);
    for (NnSize i = 0u; i < size; i++)
        items[i] = i;

    std::sort(items.begin(), items.end(),
        [&x](int a, int b) {
            return x[a] > x[b];
        }
    );

    for (NnUint i = 0u; i < k; i++)
        y[i] = items[i];
}

//

static void mergeAddForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnUint nSlices = context->inputSize.x / context->outputSize.x;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *input = (float *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            float *i = &input[sliceIndex * context->outputSize.x];
            DEBUG_VECTOR(context, "input", i);
            add_F32(
                output,
                i,
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeAddForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->inputSize.floatType == F_Q80);
    assert(context->outputSize.floatType == F_32);

    NnUint nSlices = context->inputSize.x / context->outputSize.x;
    NnUint xSize = context->outputSize.x / Q80_BLOCK_SIZE;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            add_Q80_F32(
                output,
                &input[sliceIndex * xSize],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeSumForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.z, 1u);
    assert(context->inputSize.z >= 1u);

    mergeSum_F32(
        (float **)context->output,
        (float **)context->input,
        context->outputSize.x,
        batchSize,
        context->nBatches,
        context->inputSize.z,
        nThreads,
        threadIndex);
}

static void initEmbeddingForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, 1);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.x, context->outputSize.x);
}

static void embeddingForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        copy_UNK(
            context->output[batchIndex],
            &context->weight[token * dimSize],
            dimSize,
            nThreads,
            threadIndex);
    }
}

static void embeddingForward_F32_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        quantizeF32toQ80(
            (float *)&context->weight[token * dimSize],
            (NnBlockQ80 *)context->output[batchIndex],
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void initInvRmsForward(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    assert(context->outputSize.x >= config->nColumns);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
}

static void invRmsForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnInvRmsOpConfig *config = (NnInvRmsOpConfig *)context->opConfig;
    const NnUint colSize = context->inputSize.x / config->nColumns;

    for (NnUint batchIndex = threadIndex; batchIndex < batchSize; batchIndex += nThreads) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            float rms = invRms_F32(
                &input[colIndex * colSize],
                colSize,
                config->epsilon);
            output[colIndex] = rms;
            DEBUG_SCALAR(context, "output", rms);
        }
    }
}

static void initRmsNormForward_ANY_F32_F32(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    NnBufferConfig *rmsBufferConfig = &context->bufferConfigs[config->invRmsBufferIndex];
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.x % config->nColumns, 0);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.floatType, F_32);
    ASSERT_EQ(context->weightSize.y, 1);
    ASSERT_EQ(context->weightSize.x, context->inputSize.x / config->nColumns);
    ASSERT_EQ(rmsBufferConfig->size.floatType, F_32);
    assert(rmsBufferConfig->size.x >= config->nColumns);
    ASSERT_EQ(rmsBufferConfig->size.y, context->nBatches);
}

static void rmsNormForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const NnUint invRmsBatchSize = context->bufferConfigs[config->invRmsBufferIndex].size.x;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    const NnUint colSize = context->weightSize.x;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            rmsNorm_F32(
                &output[colIndex * colSize],
                &input[colIndex * colSize],
                invRms[batchIndex * invRmsBatchSize + colIndex],
                weight,
                colSize,
                nThreads,
                threadIndex);
        }
        DEBUG_VECTOR(context, "output", output);
    }
}

static void rmsNormForward_Q80_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    ASSERT_EQ(config->nColumns, 1); // TODO: add support multiple columns

    const float *weight = (float *)context->weight;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            invRms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void initMatmulForward(NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->weightSize.y);
    ASSERT_EQ(context->inputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->outputSize.x, context->weightSize.x);
    ASSERT_EQ(context->outputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->weightSize.z, std::max(config->nExperts, 1u));

    if (!context->hasInputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for input\n", context->name);
    if (!context->hasOutputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for output\n", context->name);

}

static void initMatmulArgmaxForward(NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->weightSize.y);
    ASSERT_EQ(context->inputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->outputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->weightSize.z, std::max(config->nExperts, 1u));

    // Argmax-fused matmul writes per-thread partials (idx,val) instead of full logits.
    // So outputSize.x is expected to be a small scratch width (>= 2 * nThreads at runtime),
    // not necessarily equal to weightSize.x (vocab size).
    ASSERT_EQ(context->outputSize.floatType, F_32);

    if (!context->hasInputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for input\n", context->name);
    if (!context->hasOutputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for output\n", context->name);
}

// prefill/decode 단계. NnExecutor::setDecodePhase 에서 갱신된다.
// 단일 추론 프로세스 기준의 전역 상태이며, 실행기 스레드들은 op 경계에서만
// 동기화되므로 forward 중에는 값이 바뀌지 않는다.
static std::atomic<bool> gDecodePhase{true};

// prefill attention 을 온라인 소프트맥스 융합 커널로 처리할지.
// 기존 커널과 A/B 비교가 가능해야 하므로 스위치로 둔다(--attn-fused 0/1).
static std::atomic<bool> gAttnFused{true};

void nnCpuOpsSetAttnFused(bool enabled) {
    gAttnFused.store(enabled, std::memory_order_relaxed);
}

void nnCpuOpsSetDecodePhase(bool isDecodePhase) {
    gDecodePhase.store(isDecodePhase, std::memory_order_relaxed);
}

bool nnCpuOpsIsDecodePhase() {
    return gDecodePhase.load(std::memory_order_relaxed);
}

// prefill 에서 lm_head 가 실제로 계산해야 하는 행 범위.
// prefill 은 위치 0..n-2 만 처리하고 로짓을 전혀 읽지 않는다(decode 첫 스텝이
// 마지막 입력 토큰을 처리해 첫 출력 로짓을 만든다). 따라서 나머지 행의 계산은
// 순수한 낭비다. 그래프 형태와 sync 는 건드리지 않아 다중 노드 정합성을 유지하고,
// 연산만 줄인다.
static inline void lmHeadRowRange(const NnCpuOpContext *context, NnUint batchSize,
                                  NnUint *rowBegin, NnUint *rowCount) {
    if (context->isLmHead && batchSize > 1u && !nnCpuOpsIsDecodePhase()) {
        *rowBegin = batchSize - 1u;
        *rowCount = 1u;
    } else {
        *rowBegin = 0u;
        *rowCount = batchSize;
    }
}

static bool matmulForward_llamafile(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (!context->hasInputContinuousMemory || !context->hasOutputContinuousMemory || context->inputSize.z != 1u)
        return false;

    const NnUint n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnUint d = context->weightSize.x;
    NnUint rowBegin, rowCount;
    lmHeadRowRange(context, batchSize, &rowBegin, &rowCount);
    const NnSize inRowBytes = getBytes(context->inputSize.floatType, context->weightSize.y);
    return llamafile_sgemm(
        d, rowCount, n,
        context->weight, n,
        context->input[0] + (std::size_t)rowBegin * inRowBytes, n,
        context->output[0] + (std::size_t)rowBegin * d * sizeof(float), d,
        threadIndex, nThreads, 0,
        context->weightSize.floatType,
        context->inputSize.floatType,
        F_32
    );
}

static void matmulForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            float *output = (float *)context->output[e * context->outputSize.y + y];
            matmul_F32_F32_F32(
                output,
                (float *)context->input[e * context->inputSize.y + y],
                (float *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY],
                context->weightSize.y,
                context->weightSize.x,
                nThreads,
                threadIndex);
            DEBUG_VECTOR(context, "output", output);
        }
    }
}

// Q4_0 repack 경로 (research/03 §8).
// 가중치는 로드 시점에 block_q4_0x4 로 재배치돼 있다.
//   - 배치 4의 배수 부분: 활성화를 block_q8_0x4 로 옮겨 gemm
//   - 나머지 행(및 decode batch=1): 평범한 Q80 을 그대로 gemv
// 실측 3.0~3.2x (Cortex-A76 4스레드, batch 32).
static bool matmulForward_repack(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
#if NN_REPACK_AVAILABLE
    if (!context->isRepacked ||
        !context->hasInputContinuousMemory ||
        !context->hasOutputContinuousMemory ||
        context->inputSize.z != 1u)
        return false;

    const NnUint d = context->weightSize.x;
    const NnUint kElems = context->weightSize.y;
    const NnUint kBlocks = kElems / Q40_BLOCK_SIZE;
    const NnByte *w = context->weight;

    // 출력 열을 스레드로 나눈다. 경계는 4의 배수로 맞춘다(커널 제약).
    const NnUint nCols4 = d / 4u;
    const NnUint per4 = (nCols4 + nThreads - 1u) / nThreads;
    const NnUint c0 = threadIndex * per4 * 4u;
    if (c0 >= d)
        return true;
    NnUint cN = per4 * 4u;
    if (c0 + cN > d)
        cN = d - c0;

    NnUint rowBegin, rowCount;
    lmHeadRowRange(context, batchSize, &rowBegin, &rowCount);
    const NnBlockQ80 *x80 = (const NnBlockQ80 *)context->input[0] + (std::size_t)rowBegin * kBlocks;
    float *out = (float *)context->output[0] + (std::size_t)rowBegin * d;
    const block_q4_0x4 *wCol = (const block_q4_0x4 *)&w[(std::size_t)(c0 / 4u) * kBlocks * sizeof(block_q4_0x4)];

    const NnUint nGemm = rowCount & ~3u;   // 4의 배수 부분

    if (nGemm > 0u) {
        // 활성화를 block_q8_0x4 로 옮긴다.
        //
        // 스레드는 출력 "열"로 나뉘므로 모든 스레드가 배치 전체의 활성화를 필요로 한다.
        // 실행기는 op 경계에서만 동기화하고 op 내부 배리어가 없으므로,
        // 각 스레드가 자기 스크래치에 전체를 중복 변환한다.
        //
        // 중복 비용은 무시할 수준이다. batch 32 / k 4096 기준 스레드당 ~139 kB 셔플인데,
        // 같은 op 의 가중치 읽기는 수십 MB 다. 반대로 "행"으로 나누면 배리어는 없어지지만
        // 모든 스레드가 가중치 전체를 읽어 가중치 트래픽이 nThreads 배가 된다.
        thread_local std::vector<NnByte> scratch;
        const std::size_t need = (std::size_t)(nGemm / 4u) * kBlocks * sizeof(block_q8_0x4);
        if (scratch.size() < need)
            scratch.resize(need);
        block_q8_0x4 *xr = (block_q8_0x4 *)scratch.data();

        for (NnUint g = 0; g < nGemm / 4u; g++)
            nnPackQ80To4x4(&x80[(std::size_t)g * 4u * kBlocks], &xr[(std::size_t)g * kBlocks], kBlocks);

        ggml_gemm_q4_0_4x4_q8_0((int)kElems, &out[c0], d, wCol, xr, (int)nGemm, (int)cN);
    }

    // 나머지 행: gemv (활성화는 Q80 그대로)
    for (NnUint b = nGemm; b < rowCount; b++) {
        ggml_gemv_q4_0_4x4_q8_0((int)kElems, &out[(std::size_t)b * d + c0], d,
            wCol, &x80[(std::size_t)b * kBlocks], 1, (int)cN);
    }
    return true;
#else
    (void)nThreads; (void)threadIndex; (void)batchSize; (void)context;
    return false;
#endif
}

// ---- 캘리브레이션 덤프 (research/07 깊이 분해 검증) ----
//
// DLLAMA_CALIB_DIR 이 설정되면 block_matmul_k / block_matmul_v 실행 직후
// 입력 phi(x_l) 과 출력 K_l, V_l (RoPE 이전) 을 파일로 append 한다.
//
// RoPE 이전 값을 쓰는 이유: RoPE 는 위치 의존이므로, 위치와 무관한 선형 사상
//   K_l = A_l phi(x_k)
// 을 적합하려면 사영 직후 값이어야 한다. RoPE 는 그 뒤에 평소대로 적용된다.
//
// 형식: float32 raw. 각 파일은 [토큰 x 차원] 이 순서대로 이어붙는다.
static const char *calibDir() {
    static const char *d = std::getenv("DLLAMA_CALIB_DIR");
    return d;
}

static void calibDump(const char *kind, NnUint layer, const float *data, NnUint nRows, NnUint nCols) {
    const char *dir = calibDir();
    if (dir == nullptr)
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_l%02u.f32", dir, kind, layer);
    FILE *f = fopen(path, "ab");
    if (f == nullptr)
        return;
    fwrite(data, sizeof(float), (std::size_t)nRows * nCols, f);
    fclose(f);
}

// 이 op 가 캘리브레이션 대상인가 (block_matmul_k / block_matmul_v)
static inline const char *calibKindOf(const NnCpuOpContext *context) {
    if (calibDir() == nullptr || context->name == nullptr)
        return nullptr;
    if (std::strcmp(context->name, "block_matmul_k") == 0) return "k";
    if (std::strcmp(context->name, "block_matmul_v") == 0) return "v";
    return nullptr;
}

// 사영 직후 출력을 덤프한다.
//
// 주의: 출력 버퍼는 스레드들이 각자 다른 열 구간에 쓴다. 실행기는 op "경계"에서만
// 동기화하므로, op 안에서 스레드 0 이 바로 읽으면 다른 스레드가 아직 쓰는 중인
// 부분을 읽게 된다(실제로 이 버그 때문에 레이어 16 의 재구성 오차가 0 이 아니라
// 0.37 로 나왔다 — K_16 = W_k^16 phi(x_16) 이므로 정의상 0 이어야 한다).
// 캘리브레이션은 진단 전용이므로 여기서만 스핀 배리어를 쓴다.
static void calibDumpAfterMatmul(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const char *kind = calibKindOf(context);
    if (kind == nullptr)
        return;

    // op 별 도착 카운터. 모든 스레드가 도달해야 버퍼가 완성된다.
    static std::atomic<unsigned> arrived{0};
    static std::atomic<unsigned> epoch{0};
    const unsigned myEpoch = epoch.load(std::memory_order_acquire);
    const unsigned n = arrived.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (n < nThreads) {
        while (epoch.load(std::memory_order_acquire) == myEpoch)
            std::this_thread::yield();
        return;
    }
    // 마지막 도착 스레드가 덤프하고 다음 라운드를 연다.
    arrived.store(0, std::memory_order_relaxed);
    const NnUint dOut = context->weightSize.x;
    calibDump(kind, context->layerIndex, (const float *)context->output[0], batchSize, dOut);
    // K 쪽에서만 입력을 함께 남긴다(K 와 V 의 입력은 같은 yq 버퍼다).
    // 입력은 Q80 이므로 역양자화해서 F32 로 기록한다.
    if (std::strcmp(kind, "k") == 0) {
        const NnUint kElems = context->weightSize.y;
        const NnUint nBlocks = kElems / Q80_BLOCK_SIZE;
        std::vector<float> tmp((std::size_t)batchSize * kElems);
        const NnBlockQ80 *x = (const NnBlockQ80 *)context->input[0];
        for (NnUint b = 0; b < batchSize; b++) {
            for (NnUint bi = 0; bi < nBlocks; bi++) {
                const NnBlockQ80 *blk = &x[(std::size_t)b * nBlocks + bi];
                const float d = CONVERT_F16_TO_F32(blk->d);
                for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++)
                    tmp[(std::size_t)b * kElems + bi * Q80_BLOCK_SIZE + j] = d * blk->qs[j];
            }
        }
        calibDump("x", context->layerIndex, tmp.data(), batchSize, kElems);
    }
    epoch.fetch_add(1, std::memory_order_release);
}

static void matmulForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_repack(nThreads, threadIndex, batchSize, context)) {
        calibDumpAfterMatmul(nThreads, threadIndex, batchSize, context);
        return;
    }
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context)) {
        calibDumpAfterMatmul(nThreads, threadIndex, batchSize, context);
        return;
    }

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            float *output = (float *)context->output[e * context->outputSize.y + y];
            matmul_Q80_Q40_F32(
                output,
                (NnBlockQ80 *)context->input[e * context->inputSize.y + y],
                (NnBlockQ40 *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY],
                context->weightSize.y,
                context->weightSize.x,
                nThreads,
                threadIndex);
            DEBUG_VECTOR(context, "output", output);
        }
    }
}

static void matmulArgmaxForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (nThreads == 0)
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];
    const NnUint prevKeySize = batchSize * nActiveExpertsOr1;
    thread_local std::unordered_map<const void *, std::vector<NnUint>> prevBestPerOp;
    std::vector<NnUint> &prevBest = prevBestPerOp[(const void *)context];
    if (prevBest.size() != prevKeySize)
        prevBest.assign(prevKeySize, 0u);
    const NnUint outStride = context->outputSize.x; // expected >= 2 * nThreads
    if (outStride < nThreads * 2u) {
        if (threadIndex == 0)
            throw std::runtime_error("argmax output buffer too small");
        return;
    }

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            SPLIT_THREADS(start, end, context->weightSize.x, nThreads, threadIndex);
            const NnUint shardRows = end - start;
            float localMax = -std::numeric_limits<float>::infinity();
            NnUint localIdx = start;

            const NnBlockQ80 *x = (NnBlockQ80 *)context->input[e * context->inputSize.y + y];
            const NnBlockQ40 *wBase =
                (NnBlockQ40 *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY];
            const NnUint prevKey = y * nActiveExpertsOr1 + e;

            if (shardRows > 0) {
                // Seed local max with previous step winner in this shard (exact, no approximation).
                {
                    const NnUint nBlocks = context->weightSize.y / Q40_BLOCK_SIZE;
                    const NnUint seedIdx = prevBest[prevKey];
                    if (seedIdx >= start && seedIdx < end && seedIdx < context->weightSize.x) {
                        const float seedVal = dot_Q80_Q40_F32_row(x, &wBase[seedIdx * nBlocks], context->weightSize.y);
                        localMax = seedVal;
                        localIdx = seedIdx;
                    }
                }

                const NnUint nBlocks = context->weightSize.y / Q40_BLOCK_SIZE;

                // Keep per-thread scratch to avoid heap churn on decode hot path.
                // Important: only compute this thread's shard rows (not full vocab).
                thread_local std::vector<float> logitsShardScratch;
                logitsShardScratch.resize(shardRows);

                const NnBlockQ40 *wShard = wBase + start * nBlocks;
                matmul_Q80_Q40_F32(
                    logitsShardScratch.data(),
                    x,
                    wShard,
                    context->weightSize.y,
                    shardRows,
                    1u,
                    0u);

                for (NnUint i = 0; i < shardRows; i++) {
                    const float v = logitsShardScratch[i];
                    if (v > localMax) {
                        localMax = v;
                        localIdx = start + i;
                    }
                }
            }

            float *partial = (float *)context->output[e * context->outputSize.y + y];
            partial[threadIndex * 2u] = (float)localIdx;
            partial[threadIndex * 2u + 1u] = localMax;
            prevBest[prevKey] = localIdx;
        }
    }
}

static void argmaxReduceForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (threadIndex != 0)
        return;

    const NnArgmaxReduceOpCodeConfig *config = (NnArgmaxReduceOpCodeConfig *)context->opConfig;
    const NnUint maxThreads = std::max(1u, config->maxThreads);
    const NnUint usedThreads = std::min(nThreads, maxThreads);

    for (NnUint y = 0; y < batchSize; y++) {
        const float *partial = (const float *)context->input[y];
        float bestVal = -std::numeric_limits<float>::infinity();
        NnUint bestIdx = 0u;
        for (NnUint t = 0; t < usedThreads; t++) {
            const float idxF = partial[t * 2u];
            const float val = partial[t * 2u + 1u];
            if (val > bestVal) {
                bestVal = val;
                bestIdx = (NnUint)idxF;
            }
        }
        float *out = (float *)context->output[y];
        out[0] = (float)bestIdx;
    }
}

static void siluForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            float *output = (float *)context->output[z * context->outputSize.y + y];
            silu_F32(output, context->outputSize.x, nThreads, threadIndex);
        }
    }
}

static void geluForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        gelu_F32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void initRopeForward_F32(NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    if (context->bufferFlags[config->ropeCacheBufferIndex] == 1)
        return;
    context->bufferFlags[config->ropeCacheBufferIndex] = 1;

    float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    fullfillRopeCache(config, cache);
}

static void ropeForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];
    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    const bool isQ = config->isQ == 1;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        const NnUint pos = (NnUint)positions[batchIndex];
        if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1)
            ropeLlama_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
        else if (config->type == ROPE_FALCON)
            ropeFalcon_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
        else
            throw std::runtime_error("Unsupported rope type");
    }
}

static void initMultiHeadAttForward(NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->outputSize.x, config->qSliceD0);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    NnSize3D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    ASSERT_EQ(querySize->x, config->qSliceD0);
    NnSize3D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);
}

static void multiHeadAttForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    float *query = (float *)context->buffers[config->queryBufferIndex];
    // SP>1: KV buffer holds full seqLen after SYNC_SP_KV allgather.
    // SP=1: KV buffer holds full seqLen normally.
    // Either way, we use the full seqLen KV buffer.
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    // prefill(batchSize > 1)은 KV 캐시를 배치 전체가 공유하는 배치 커널로 처리한다.
    // decode(batchSize == 1)는 재사용할 대상이 없으므로 기존 경로가 그대로 최적이다.
    if (batchSize > 1u) {
#ifndef NDEBUG
        for (NnUint b = 0; b < batchSize; b++)
            assert((NnUint)positions[b] < config->seqLen);
#endif
        if (gAttnFused.load(std::memory_order_relaxed)) {
            // att 버퍼를 쓰지 않는다. 점수 스크래치가 seqLen 이 아니라 TILE 에 비례한다.
            multiheadAttFused_F32(
                context->output, query, config->qSliceD0,
                keyCache, valueCache,
                positions, batchSize,
                config->nHeads, config->nHeads0,
                config->nKvHeads, config->kvDim0, config->headDim,
                config->seqLen,
                nThreads, threadIndex);
            return;
        }
        multiheadAttBatch_F32(
            context->output, query, config->qSliceD0,
            att, keyCache, valueCache,
            positions, batchSize,
            config->nHeads, config->nHeads0,
            config->nKvHeads, config->kvDim0, config->headDim,
            config->seqLen,
            nThreads, threadIndex);
        return;
    }

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *y = (float *)context->output[batchIndex];
        float *q = &query[batchIndex * config->qSliceD0];
        NnUint pos = (NnUint)positions[batchIndex];
        assert(pos < config->seqLen);

        DEBUG_VECTOR(context, "input", y);
        DEBUG_VECTOR(context, "q", q);

        multiheadAtt_F32(y, q,
            &att[batchIndex * config->nHeads0 * config->seqLen],
            keyCache, valueCache,
            pos,
            config->nHeads, config->nHeads0,
            config->nKvHeads, config->kvDim0, config->headDim,
            config->seqLen,
            nThreads, threadIndex);

        DEBUG_VECTOR(context, "output", y);
    }
}

static void initMulForward(NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
}

static void mulForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const float *multiplier = (float *)context->buffers[config->multiplierBufferIndex];

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            mul_F32(
                (float *)context->output[zOffset + y],
                (float *)context->input[zOffset + y],
                &multiplier[context->outputSize.x * (zOffset + y)],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void scaleForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)context->opConfig;
    const float *scale = (float *)context->buffers[config->scaleBufferIndex];

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            const NnUint index = z * context->inputSize.y + y;
            const float s = scale[index];
            const float *i = (float *)context->input[index];
            float *o = (float *)context->output[index];
            scale_F32(i, o, s, context->inputSize.x, nThreads, threadIndex);
        }
    }
}

static void initCastForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
}

static void castForward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnUint rowBytes = context->outputSize.nBytes / context->outputSize.y;

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            copy_UNK(
                context->output[zOffset + y],
                context->input[zOffset + y],
                rowBytes,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            quantizeF32toQ80(
                (float *)context->input[zOffset + y],
                (NnBlockQ80 *)context->output[zOffset + y],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            dequantizeQ80toF32(
                (NnBlockQ80 *)context->input[zOffset + y],
                (float *)context->output[zOffset + y],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void initRepeatZForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, 1u);
    assert(context->inputSize.z <= context->outputSize.z);
    assert(context->inputSize.z > 0u);
}

static void repeatZForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);
    const NnSize dimSize = getBytes(F_Q80, context->outputSize.x);

    for (NnUint z = 0u; z < context->outputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            NnByte *output = context->output[z * context->outputSize.y + y];
            if (z == 0u) {
                quantizeF32toQ80(
                    (float *)context->input[y],
                    (NnBlockQ80 *)output,
                    context->outputSize.x,
                    nThreads,
                    threadIndex);
            } else {
                copy_UNK(
                    output,
                    context->output[y],
                    dimSize,
                    nThreads,
                    threadIndex);
            }
        }
    }
}

static void shiftForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->hasInputContinuousMemory, true);
    ASSERT_EQ(context->hasOutputContinuousMemory, true);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, 1);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const float *indexes = (float *)context->pipes[config->indexPipeIndex];
    const NnSize dimBytes = getBytes(F_32, context->inputSize.x);
    NnByte *output = context->output[0];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnSize index = (NnSize)indexes[batchIndex];
        // SP write guard: only write if position is in this rank's local sequence range
        if (config->localSeqLen > 0) {
            if (index < config->localSeqStart || index >= config->localSeqStart + config->localSeqLen)
                continue;
        }
        assert((index + 1) * context->inputSize.x <= context->outputSize.x);
        copy_UNK(
            &output[index * dimBytes],
            context->input[batchIndex],
            dimBytes,
            nThreads,
            threadIndex);
    }
}

static void softmaxForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(*context->input == *context->output);

    for (NnUint y = threadIndex; y < batchSize; y += nThreads)
        softmax_F32(
            (float *)context->output[y],
            context->outputSize.x);
}

static void initMoeGateForward(NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    assert(context->inputSize.x >= config->k);
    ASSERT_EQ(context->outputSize.z, config->k);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.x, 1u);
}

static void moeGateForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    float *indexes = (float *)context->buffers[config->indexesBufferIndex];

    std::vector<NnUint> pos(config->k);
    for (NnUint y = threadIndex; y < batchSize; y += nThreads) {
        float *input = (float *)context->input[y];

        topk_F32(input, pos.data(), context->inputSize.x, config->k);

        float sum;
        if (config->normTopk == 1u) {
            sum = 0.0f;
            for (NnUint i = 0u; i < config->k; i++)
                sum += input[pos[i]];
        } else {
            sum = 1.0f;
        }

        for (NnUint k = 0u; k < config->k; k++) {
            const NnUint p = pos[k];
            indexes[y * config->k + k] = (float)p;

            // (nActiveExperts, nBatches, 1)
            float *output = (float *)context->output[k * context->outputSize.y + y];
            *output = input[p] / sum;
        }

        DEBUG_VECTOR(context, "indexes", (&indexes[y * config->k]));
    }
}

// device

void printCpuInstructionSet() {
    printf("🧠 CPU:");
#if defined(__ARM_NEON)
    printf(" neon");
#if defined(__ARM_FEATURE_DOTPROD)
    printf(" dotprod");
#endif
#if defined(__ARM_FP16_FORMAT_IEEE)
    printf(" fp16");
#endif
#endif
#if defined(__AVX2__)
    printf(" avx2");
#endif
#if defined(__AVX512F__)
    printf(" avx512f");
#endif
    printf("\n");
}

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_EMBEDDING)
        return initEmbeddingForward;
    if (code == OP_INV_RMS)
        return initInvRmsForward;
    if (code == OP_RMS_NORM)
        return initRmsNormForward_ANY_F32_F32;
    if (code == OP_ROPE)
        return initRopeForward_F32;
    if (code == OP_MULTIHEAD_ATT)
        return initMultiHeadAttForward;
    if (code == OP_MATMUL)
        return initMatmulForward;
    if (code == OP_MATMUL_ARGMAX)
        return initMatmulArgmaxForward;
    if (code == OP_ARGMAX_REDUCE)
        return nullptr;
    if (code == OP_MUL)
        return initMulForward;
    if (code == OP_CAST)
        return initCastForward;
    if (code == OP_REPEAT_Z)
        return initRepeatZForward;
    if (code == OP_MOE_GATE)
        return initMoeGateForward;
    return nullptr;
}

NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return mergeAddForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mergeAddForward_Q80_F32;
    }
    if (code == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return mergeSumForward_F32_F32;
    }
    if (code == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return embeddingForward_F32_F32_F32;
        if (quantType == F32_F32_Q80) return embeddingForward_F32_F32_Q80;
    }
    if (code == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return invRmsForward_F32_F32;
    }
    if (code == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return rmsNormForward_F32_F32_F32;
        if (quantType == Q80_F32_F32) return rmsNormForward_Q80_F32_F32;
    }
    if (code == OP_MATMUL) {
        if (quantType == F32_F32_F32) return matmulForward_F32_F32_F32;
        if (quantType == Q80_Q40_F32) return matmulForward_Q80_Q40_F32;
    }
    if (code == OP_MATMUL_ARGMAX) {
        if (quantType == Q80_Q40_F32) return matmulArgmaxForward_Q80_Q40_F32;
    }
    if (code == OP_ROPE) {
        if (quantType == F32_F32_F32) return ropeForward_F32_F32;
    }
    if (code == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return multiHeadAttForward_F32_F32;
    }
    if (code == OP_GELU) {
        if (quantType == F32_F32_F32) return geluForward_F32_F32_F32;
    }
    if (code == OP_SILU) {
        if (quantType == F32_F32_F32) return siluForward_F32_F32;
    }
    if (code == OP_MUL) {
        if (quantType == F32_F32_F32) return mulForward_F32_F32;
    }
    if (code == OP_SCALE) {
        if (quantType == F32_F32_F32) return scaleForward_F32_F32;
    }
    if (code == OP_CAST) {
        if (quantType == F32_F32_F32) return castForward_ANY;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_Q80) return castForward_ANY;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    if (code == OP_REPEAT_Z) {
        if (quantType == F32_F32_Q80) return repeatZForward_F32_Q80;
    }
    if (code == OP_SHIFT) {
        if (quantType == F32_F32_F32) return shiftForward_F32_F32;
    }
    if (code == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return softmaxForward_F32_F32;
    }
    if (code == OP_ARGMAX_REDUCE) {
        if (quantType == F32_F32_F32) return argmaxReduceForward_F32_F32;
    }
    if (code == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) return moeGateForward_F32_F32;
    }
    return nullptr;
}
