# SharedPack-SDOT — 공유 packed activation을 production에 넣는 절차

이 문서는 [19-sharedpack-implementation.md](../19-sharedpack-implementation.md)의 구현 보조자료다.
SharedPack이 해결하는 문제, executor에 별도 op가 필요한 이유, 정확성·성능 검증 순서와
**현재 worktree의 blocker**를 처음 보는 사람도 따라갈 수 있게 정리한다.

문서의 현재 상태 기준일은 **2026-08-24**다.

---

## 1. SharedPack이 바꾸는 것은 무엇인가

기존 `matmulForward_repack`은 네 worker thread가 각각 다음 작업을 수행한다.

```text
1. Q80 activation 전체를 자기 thread_local scratch에 Q8×4로 pack
2. 자기에게 배정된 output column의 Q4×Q8 GEMM 수행
```

SharedPack은 pack을 독립된 parallel op로 분리한다.

```text
Q80 activation
       │
       ▼
OP_PACK_Q80X4
  thread 0 ── group [g0, g1) ─┐
  thread 1 ── group [g1, g2) ─┤
  thread 2 ── group [g2, g3) ─┼─ 하나의 shared block_q8_0x4 buffer
  thread 3 ── group [g3, g4) ─┘
       │
       ▼  executor op boundary가 barrier
OP_MATMUL
  네 thread가 같은 packed buffer를 읽고 서로 다른 output column 계산
```

여기서 공유하는 것은 GEMM 결과가 아니라 **read-only activation layout**이다. Weight와
output column 소유권은 기존과 같다.

---

## 2. 왜 matmul 내부에서 thread 0만 pack하지 않는가

thread 0 하나가 pack하고 나머지가 기다리게 만들려면 matmul 내부 barrier가 필요하다.
기존 executor는 op 내부가 아니라 **op 경계**에서 worker들을 동기화한다.

따라서 별도 op가 자연스럽다.

```text
OP_PACK_Q80X4 실행 중 : 모든 thread가 pack 작업을 나눠 수행
op 종료                : executor가 모든 thread 도착을 보장
다음 OP_MATMUL          : 완성된 shared buffer를 안전하게 읽음
```

한 thread만 pack하면 나머지 세 thread가 놀기 때문에 pack 비용이 그대로 critical path에
추가된다. 실제 분할식은 다음과 같다.

```text
nGroups = floor(batchSize / 4)
g0(t) = floor(nGroups × t / nThreads)
g1(t) = floor(nGroups × (t+1) / nThreads)

thread t가 [g0(t), g1(t))를 pack
```

각 group은 token row 네 개와 모든 K block으로 구성된다.

---

## 3. Inter-thread 공유와 inter-projection 공유

두 공유는 구분해야 한다.

### 3.1 Inter-thread sharing — 현재 1차 구현

한 GEMM 호출 안에서 네 thread가 packed activation 하나를 공유한다.

```text
Down 입력 dq ── pack 1회 ── Down의 네 GEMM worker
```

Down microbenchmark의 1.68배는 주로 이 효과의 kernel-only 상한이다.

### 3.2 Inter-projection sharing — 다음 단계

입력이 같은 projection들이 pack 결과를 재사용한다.

```text
attention norm 출력 xq ── pack 1회 ── Q, K, V
FFN norm 출력 yq       ── pack 1회 ── Gate, Up
FFN hidden dq          ── pack 1회 ── Down
attention 출력         ── pack 1회 ── O
```

현재 우선순위는 Down 하나에서 production 이득과 정확성을 먼저 확인하는 것이다. 처음부터
Q/K/V와 Gate/Up까지 바꾸면 어느 family에서 오류나 회귀가 생겼는지 분리하기 어렵다.

---

## 4. Packed buffer의 모양

`block_q8_0x4` 하나는 네 Q80 block을 SDOT가 읽는 순서로 묶는다.

```text
Q80 block 1개          = 32 int8 + FP16 scale = 34 bytes
block_q8_0x4 1개       = 4 × 34 bytes          = 136 bytes
K block 수             = K / 32
batch group 수         = floor(B / 4)
```

따라서 필요한 byte 수는 다음과 같다.

```text
packedBytes = floor(B/4) × (K/32) × 136
```

Down, `B=32`, `K=14336`이면:

```text
8 groups × 448 K blocks × 136 bytes = 487,424 bytes ≈ 476 KiB
```

이것은 전체 shared buffer 크기다. GEMM microkernel이 한 번에 재사용하는 16-row 단위는
그 절반인 약 238 KiB다. 두 값을 섞으면 과거 K-sweep 해석 오류가 반복된다.

---

## 5. 기존 경로와 fallback을 함께 유지하는 이유

`NnMatmulOpConfig.prepackedBufferIndex`가 `NN_NO_PREPACK`이면 기존 경로를 사용한다.

```text
prepacked buffer 있음  → shared Q8×4로 4행 배수 GEMM
prepacked buffer 없음  → thread_local private pack 후 기존 GEMM
DLLAMA_REPACK=0        → 기존 non-repack 경로
남은 1~3 token row     → 원본 Q80 GEMV
```

이렇게 해야 다음이 가능하다.

- 새 경로를 Down 하나에만 단계적으로 적용
- 환경 변수로 기존 baseline 재현
- 마지막 microbatch tail 처리
- lm_head처럼 일부 row만 계산하는 특수 경로 제외
- 오류가 나면 기존 경로로 안전하게 복귀

---

## 6. 코드에서 어디를 보면 되는가

| 역할 | 위치 |
|---|---|
| `OP_PACK_Q80X4`와 config | [nn-core.hpp](../../src/nn/nn-core.hpp) |
| op 이름 직렬화 | [nn-core.cpp](../../src/nn/nn-core.cpp) |
| 실제 parallel pack | [nn-cpu-ops.cpp](../../src/nn/nn-cpu-ops.cpp) `packQ80x4Forward` |
| prepacked matmul 분기 | [nn-cpu-ops.cpp](../../src/nn/nn-cpu-ops.cpp) `matmulForward_repack` |
| Down 앞 pack op와 buffer | [llm.cpp](../../src/llm.cpp) `buildLlmNet` |
| dup/shared microbenchmark | [bench_shape.cpp](../../prefill_bench/bench_shape.cpp) |

op enum은 root와 worker 사이에 직렬화된다. 따라서 새 enum 값은 기존 값 사이가 아니라 끝에
추가해야 한다. 중간에 삽입하면 이후 모든 opcode 번호가 달라진다.

---

## 7. 2026-08-24 현재 구현 상태

### 완료

| 항목 | 상태 |
|---|---|
| opcode/config 추가 | 완료 |
| parallel `packQ80x4Forward` | 완료 |
| matmul의 shared/fallback 분기 | 완료 |
| 기존 13개 matmul config에 `NN_NO_PREPACK` 지정 | 완료 |
| Down용 shared buffer와 pack op net builder 배선 | worktree에 구현됨 |

### 현재 blocker

첫 N=1 실행은 다음 오류로 중단됐다.

```text
Unsupported CPU op code: PACK_Q80X4,
quant: Q80_Q80_F32,
op name: block_pack_dq
```

원인은 packed buffer의 **저장 형식과 type system의 의미가 일치하지 않기 때문**이다.

현재 builder는 필요한 byte 수를 확보하기 위해 `d_pack`을 `F_32` buffer로 선언했다.
하지만 그 안에 실제로 쓰는 것은 float가 아니라 opaque `block_q8_0x4` bytes다. 런타임은
output buffer 선언을 보고 이 op를 `Q80_Q80_F32`로 분류한다. 반면 forward table에는
`Q80_Q80_Q80` 조합만 등록돼 있어 handler를 찾지 못한다.

```text
buffer 선언          F_32       ← byte container로 임시 사용
실제 buffer 내용     block_q8_0x4
runtime quant type   Q80_Q80_F32
등록된 handler       Q80_Q80_Q80
결과                 unsupported op
```

따라서 **아직 valid logits나 production 성능 결과는 없다.** `333.8 GOPS`는 계속
microbenchmark의 pack-excluded upper bound다.

### blocker 해소의 두 층

1. 기능 검증용 최소 변경: 현재 opaque F32 container를 유지하고 `Q80_Q80_F32` 조합에도
   `packQ80x4Forward`를 연결한다. 이 경우 주석과 assertion으로 “F32 값이 아니라 byte
   storage”임을 분명히 해야 한다.
2. 최종 구현: `block_q8_0x4`를 나타내는 명시적 packed storage type 또는 raw workspace를
   도입해 allocation, pointer resolution, 직렬화가 실제 의미를 표현하게 한다.

첫 방법으로 정확성과 pack-inclusive 성능을 빠르게 판정한 뒤, 성공한 경우 두 번째 방법으로
정리하는 순서가 합리적이다. type enum도 wire protocol에 영향을 줄 수 있으므로 root/worker
동일 binary hash를 확인해야 한다.

---

## 8. 정확성 검증 절차

성능보다 아래 순서를 먼저 통과해야 한다.

### 8.1 Pack 자체

```text
기존 private nnPackQ80To4x4 결과
새 shared OP_PACK_Q80X4 결과
→ 사용한 전체 byte 범위 memcmp == 0
```

### 8.2 GEMM 출력

같은 Q80 input과 Q4 weight에 대해 기존 private-packed GEMM과 shared-packed GEMM의 F32
출력을 비교한다.

```text
memcmp(output_baseline, output_shared) == 0
```

pack은 레이아웃만 바꾸므로 산술 순서는 바뀌지 않는다.

### 8.3 Production logits

동일한 실행 구성끼리 비교한다.

| 구성 | 기존 reference hash |
|---|---|
| N=1, B=16 | `8b8178a50a97` |
| N=1, B=32 | `8b8178a50a97` |
| N=8, B=16 | `c52e50e37e65` |

`N=1`과 `N=8`은 기존에도 서로 다르므로 둘을 직접 bit gate로 비교하지 않는다.

검증 순서:

```text
N=1 B=16
→ N=1 B=32
→ 마지막 4행 미만 tail
→ N=8 B=16 wave pipeline
```

---

## 9. 성능 검증 절차

shared microbenchmark는 buffer를 timed loop 밖에서 만들었다. production 판정에는 다음을
모두 포함한다.

```text
T_new-down = T(block_pack_dq) + T(block_matmul_w2)
```

기존과 새 경로를 같은 session에서 교차 측정한다.

```text
warm-up
baseline, shared, baseline, shared, ...
```

| 판정 항목 | 기준 |
|---|---:|
| Down, pack 포함 | 1.45× 이상, 목표 1.60× |
| Gate/Up | 확장 후 비열화 금지 |
| 전체 projection 가중평균 | 1.15× 이상 |
| 단일 노드 E2E | 1.12× 1차, 1.15× 강한 성공 |
| 정확성 | packed bytes, GEMM output, 동일 구성 logits bit-identical |

`block_matmul_w2`만 빨라진 시간을 쓰면 안 된다. pack을 별도 op로 옮겼으므로 반드시 두 op의
합을 기존 `block_matmul_w2`와 비교한다.

---

## 10. 성공한 뒤 novelty를 강화하는 단계

Shared input packing은 전통적인 GEMM/BLAS 최적화 원리이므로 이것만으로 새로운 알고리즘을
주장하기 어렵다. 현재 구현의 1차 가치는 production 병목 제거다.

다음 단계는 producer가 downstream layout을 직접 만드는 것이다.

```text
현재:
F32 activation → Q80 buffer → 별도 Q80→Q8×4 pack → SDOT GEMM

목표:
F32 activation → 기존과 동일한 Q80 quantization
                 ├─ tail/fallback용 Q80
                 └─ GEMM용 shared Q8×4 layout 직접 생성
```

그 뒤 Q/K/V와 Gate/Up이 같은 packed panel을 재사용하도록 lifetime을 projection family까지
확장한다.

강한 기여 문장은 다음 결합에서 나온다.

> 여러 CPU thread와 연속 projection이 하나의 quantized activation panel을 공유하고,
> producer가 downstream SDOT layout을 정확하게 직접 생성하는 packed-activation dataflow.

이 단계에서도 Q80 bytes, Q8×4 bytes, 최종 GEMM 출력의 bit-identical gate를 유지한다.

---

## 11. 다음 작업 체크리스트

```text
[ ] PACK_Q80X4 quant-type mismatch를 기능 검증 수준에서 해소
[ ] N=1 B=16 pack bytes memcmp
[ ] N=1 B=16 Down output bit comparison
[ ] N=1 B=16/B=32 logits reference 통과
[ ] pack-inclusive Down timing 수집
[ ] Down ≥1.45×이면 N=8 wave 검증
[ ] 성공 후 packed storage type 정식화
[ ] Gate/Up family 공유
[ ] Q/K/V family 공유
[ ] F32 producer에서 shared Q8×4 직접 생성
```

