#ifndef NN_REPACK_H
#define NN_REPACK_H

#include "nn-quants.hpp"

// Q4_0 가중치 repack 경로.
//
// 배경: research/02, research/03 참조.
//   기존 llamafile tinyBLAS_Q0_ARM 은 GEMM 마다 4비트 가중치를 언팩하느라
//   명령어의 절반 이상을 쓰고 ~110 GFLOPS 에서 포화한다.
//   가중치를 미리 SIMD 친화적 레이아웃으로 재배치하면 330 GFLOPS 가 나온다(3.1x).
//
// 커널 자체는 llama.cpp/ggml(MIT)에서 가져왔다. nn-repack.cpp 상단 주석 참조.

#define QK4_0 32
#define QK8_0 32

// ggml 의 block<K,N> 과 바이트 레이아웃이 정확히 같아야 한다(커널이 인라인 asm 이다).
//   block_q4_0x4: q4_0 블록 4개를 출력행 4개로 인터리브. 4*2 + (32*4*4)/8 = 72 B
//   block_q8_0x4: q8_0 블록 4개를 배치행 4개로 인터리브. 4*2 + (32*4*8)/8 = 136 B
typedef struct {
    std::uint16_t d[4];
    std::int8_t qs[64];
} block_q4_0x4;

typedef struct {
    std::uint16_t d[4];
    std::int8_t qs[128];
} block_q8_0x4;

// ggml 의 block_q8_0 / block_q4_0 은 dllama 의 것과 바이트 레이아웃이 동일하다.
// 커널이 이 이름들을 쓰므로 별칭만 준다.
typedef NnBlockQ80 block_q8_0;
typedef NnBlockQ40 block_q4_0;

static_assert(sizeof(block_q4_0x4) == 72, "block_q4_0x4 layout mismatch");
static_assert(sizeof(block_q8_0x4) == 136, "block_q8_0x4 layout mismatch");
// in-place repack 이 성립하는 근거
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(NnBlockQ40), "in-place repack impossible");
// gemv 가 활성화를 그대로 받을 수 있는 근거
static_assert(sizeof(NnBlockQ80) == 34, "NnBlockQ80 must match ggml block_q8_0");

// 이 빌드에서 repack 경로를 쓸 수 있는가.
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    #define NN_REPACK_AVAILABLE 1
#else
    #define NN_REPACK_AVAILABLE 0
#endif

// F32 활성화를 block_q8_0x4 로 양자화 + 인터리브 (배치 4행 단위).
void ggml_quantize_mat_q8_0_4x4(const float *x, void *vy, std::int64_t k);

// 배치 1행: vy 는 평범한 block_q8_0(= NnBlockQ80) 배열.
void ggml_gemv_q4_0_4x4_q8_0(int n, float *s, std::size_t bs, const void *vx, const void *vy, int nr, int nc);

// 배치 4행 배수: vy 는 block_q8_0x4 배열.
void ggml_gemm_q4_0_4x4_q8_0(int n, float *s, std::size_t bs, const void *vx, const void *vy, int nr, int nc);

// ---- dllama 쪽 헬퍼 ----

// 가중치 슬라이스를 in-place 로 repack 한다.
//   weight: NnBlockQ40 배열, [d행 x kBlocks열] (행 우선)
//   d 는 4의 배수여야 한다. 아니면 false 를 반환하고 아무것도 바꾸지 않는다.
bool nnRepackQ40InPlace(NnByte *weight, NnUint d, NnUint kBlocks);

// 이 형상에 repack GEMM 을 쓸 수 있는가.
bool nnRepackSupported(NnUint d, NnUint kBlocks);

// NnBlockQ80 4행(각 kBlocks개) -> block_q8_0x4 kBlocks개.
void nnPackQ80To4x4(const NnBlockQ80 *in, block_q8_0x4 *out, NnUint kBlocks);

#endif
