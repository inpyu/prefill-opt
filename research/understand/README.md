# understand/ — 현재 개발 방향을 읽기 위한 보조 자료

이 폴더는 `research/`에 시간순으로 남은 가설·실험·기각 기록을, **현재 무엇을 개발하고
있는가**를 기준으로 다시 설명한다. 오래된 문서에는 당시에는 타당했지만 이후 교락 실험으로
철회된 해석이 함께 남아 있으므로, 처음부터 번호순으로만 읽으면 현재 방향을 오해할 수 있다.

---

## 1. 지금 개발하려는 것

현재 최우선 목표는 scheduling knob을 더 탐색하는 것이 아니다.

> **정확도를 전혀 바꾸지 않으면서 CPU prefill의 Q4×Q8 GEMM 실행 경로를 개선하고,
> 동일 activation의 중복 packing 때문에 무너진 4-thread 성능을 회복하는 것**이 목표다.

현재 개발 이름은 **SharedPack-SDOT**이다.

```text
기존
  Q80 activation
    ├─ thread 0이 자기 scratch에 전체 pack → output column 일부 계산
    ├─ thread 1이 자기 scratch에 전체 pack → output column 일부 계산
    ├─ thread 2가 자기 scratch에 전체 pack → output column 일부 계산
    └─ thread 3이 자기 scratch에 전체 pack → output column 일부 계산

개발 중
  Q80 activation
          │
          ▼
  OP_PACK_Q80X4
    네 thread가 서로 다른 token-row group을
    하나의 shared Q8×4 buffer에 병렬로 작성
          │
          ▼  executor op boundary가 barrier
  Q4×Q8 SDOT GEMM
    네 thread가 같은 packed activation을 읽고
    서로 다른 output column 계산
```

이 변경은 weight precision, activation quantization, SDOT 누적 순서를 바꾸지 않는다.
목표 정확성은 근사 오차 허용이 아니라 **기존 실행과 bit-identical**이다.

---

## 2. 왜 이 문제를 선택했는가

`S=447`, `B=32`, 단일 노드의 production op breakdown은 다음과 같다.

| 구간 | 전체 prefill 시간 비중 |
|---|---:|
| Down Q4×Q8 GEMM | 29.44% |
| Gate + Up Q4×Q8 GEMM | 39.28% |
| Q/K/V/O 등의 projection GEMM | 약 15.4% |
| **Q4×Q8 GEMM 합계** | **약 84%** |

반면 FFN의 SiLU, multiply, F32→Q80 cast와 같은 graph-level 중간 연산은 합계 2.45%였다.
따라서 큰 F32 buffer를 제거하는 QCFuse의 이상적 상한도 1.025배뿐이었다. 전체의 20% 이상을
노리려면 GEMM 경로를 개선해야 한다.

가장 큰 단일 op인 Down은 동일 MAC 수의 Gate/Up보다 4-thread 성능이 크게 낮았다.
packing 정책만 바꾼 대조 실험은 다음 결과를 냈다.

```text
K=14336, B=32, 4 threads

스레드별 private pack     198.7 GOPS
하나의 packed buffer 공유 333.8 GOPS
kernel-only 차이             1.68×
```

`K`와 packed panel 크기를 바꿔도 shared 경로는 327~334 GOPS로 평탄했다. 따라서 과거의
“긴 K panel 자체가 cache에 들어가지 않는다”는 해석은 철회했다.

현재의 정확한 해석은 다음과 같다.

> 긴 `K`는 기존 4-thread duplicate-packing 구현에서 네 개의 독립 packed footprint를
> 크게 만드는 증폭 변수다. 긴 K 산술 자체가 느린 것이 아니며, activation 하나를 공유하면
> 성능 붕괴가 사라진다.

---

## 3. 개발 범위

### 지금 구현하는 것

1. Down 앞에 별도 `OP_PACK_Q80X4` 추가
2. 네 worker가 shared buffer의 서로 다른 batch-row group을 병렬로 pack
3. executor op boundary를 synchronization으로 사용
4. `block_matmul_w2`가 shared packed buffer를 읽도록 연결
5. 기존 private-pack 경로와 GEMV tail fallback 유지
6. packed bytes, GEMM output, production logits의 bit-identical 검증
7. pack과 executor 경계까지 포함한 production 성능 측정

### Down 성공 후 확장할 것

```text
pack(yq) 1회 → Gate와 Up이 공유
pack(xq) 1회 → Q, K, V가 공유
pack(attention output) 1회 → O
```

그다음에는 F32→Q80 producer가 downstream SDOT용 Q8×4 layout을 직접 생성하도록 확장한다.
이 단계가 단순한 전통적 BLAS shared packing을 넘어서는 알고리즘 기여 후보다.

### 현재 우선하지 않는 것

- `N`, `B`, sublayer partition을 다시 탐색하는 scheduling 연구
- KP-SDOT K-panelization
- WCEP weight-specific 산술 합성
- QCFuse FFN 중간 F32 buffer fusion
- 정확도를 바꾸는 근사 attention, sparsity 또는 추가 양자화

DerivePP의 wave pipeline과 기존 scheduling 결과는 폐기하지 않는다. 분산 실행 기반으로
유지하지만, 현재 개발 자원은 CPU 연산 경로에 우선 배정한다.

---

## 4. 현재 구현 상태 — 2026-08-24

| 항목 | 상태 |
|---|---|
| duplicate/shared 원인 대조 microbenchmark | 완료 |
| `OP_PACK_Q80X4` opcode와 config | 완료 |
| parallel `packQ80x4Forward` | 완료 |
| matmul shared/fallback 분기 | 완료 |
| Down용 buffer·pack op net builder 배선 | worktree에 구현 |
| valid N=1 logits | **미완료** |
| pack-inclusive production 성능 | **미측정** |
| Gate/Up, Q/K/V family 공유 | 미구현 |
| F32 producer→Q8×4 직접 생성 | 미구현 |

현재 첫 실행은 다음 type mismatch에서 중단된다.

```text
Unsupported CPU op code: PACK_Q80X4,
quant: Q80_Q80_F32,
op name: block_pack_dq
```

builder는 필요한 byte 공간을 얻기 위해 packed buffer를 `F_32` container로 선언했지만,
실제 내용은 opaque `block_q8_0x4` bytes다. 런타임은 이를 `Q80_Q80_F32` op로 분류하고,
forward table에는 `Q80_Q80_Q80` handler만 등록되어 있다.

따라서 다음 사실을 구분해야 한다.

```text
확정: shared kernel upper bound는 Down에서 1.68×
미확정: pack + executor boundary를 포함한 production 가속률
미확정: production bit-identical logits
```

상세 blocker와 해결 절차는 [13-sharedpack-sdot.md](13-sharedpack-sdot.md)를 따른다.

---

## 5. 성공 기준

shared microbenchmark의 333.8 GOPS는 packed buffer를 timed loop 밖에서 만든 값이다.
production에서는 반드시 다음 합을 기존 Down 시간과 비교한다.

```text
T_new-down = T(block_pack_dq) + T(block_matmul_w2)
```

| 판정 항목 | 기준 |
|---|---:|
| Down, pack 포함 | 1.45× 이상, 목표 1.60× |
| Gate/Up | 확장 후 비열화 금지 |
| 전체 projection 가중평균 | 1.15× 이상 |
| 단일 노드 E2E | 1.12× 1차, 1.15× 강한 성공 |
| 최종 연산 경로 | E2E 1.20× 이상 |
| 정확성 | packed bytes, GEMM output, 동일 구성 logits bit-identical |

단순 시간 추정으로 inter-thread와 projection-family 공유가 모두 기대대로 작동하면 E2E
약 1.16배를 예상한다. 이는 아직 실측 결과가 아니라 **개발 목표를 정하기 위한 추정**이다.
20%까지는 SharedPack 이후 전체 GEMM에서 추가 약 4%의 개선이 필요하다.

---

## 6. 현재 방향을 이해하는 가장 짧은 읽기 순서

```text
이 README
  → 12-cpu-compute-path
       왜 WCEP/QCFuse/KP-SDOT이 탈락했고
       K-sweep의 교락을 어떻게 분리했는가
  → 13-sharedpack-sdot
       SharedPack을 executor와 net builder에 어떻게 넣는가
       현재 blocker와 정확성·성능 gate는 무엇인가
  → 18-compute-path
       원자료 수치와 시간순 실험 기록
  → 19-sharedpack-implementation
       source 변경과 인수인계 상태
```

### Pipeline/scheduling 배경까지 이해하려면

```text
00-overview
  → 04-concepts
  → 06-math-pipeline
  → 07-math-measurement
  → 11-axis-cert
  → 16-derivepp
  → 12-cpu-compute-path
  → 13-sharedpack-sdot
```

---

## 7. 파일별 역할

| 파일 | 역할 |
|---|---|
| [00-overview.md](00-overview.md) | 기존 CPU cluster PP 문제와 DerivePP 시스템 흐름 |
| [01-code-map.md](01-code-map.md) | 개념과 실제 source/function의 대응 |
| [02-glossary-extra.md](02-glossary-extra.md) | 기호·이름 충돌·추가 용어 |
| [03-glossary-full.md](03-glossary-full.md) | 프로젝트에서 사용한 상세 용어와 측정 맥락 |
| [04-concepts.md](04-concepts.md) | Transformer, KV cache, PP/CP/TP 원리 |
| [05-math-attention.md](05-math-attention.md) | attention, RoPE, KV cache, 정확성 유도 |
| [06-math-pipeline.md](06-math-pipeline.md) | `M`, `N`, `B`, bubble, pipeline 시간 유도 |
| [07-math-measurement.md](07-math-measurement.md) | warm-up, paired A/B, 정확성·측정 규율 |
| [08-math-executor-calibration.md](08-math-executor-calibration.md) | kernel sum과 production executor 비용의 차이 |
| [09-math-planner.md](09-math-planner.md) | completion-vector recurrence와 dominance pruning |
| [10-calibration-history.md](10-calibration-history.md) | calibration 설계가 바뀐 역사와 제거한 가정 |
| [11-axis-cert.md](11-axis-cert.md) | scheduling 축 활성화 절차 — 현재 구현 우선순위는 아님 |
| [12-cpu-compute-path.md](12-cpu-compute-path.md) | **현재 연산 방향의 근거** — 기각 결과와 K-sweep 교락 분리 |
| [13-sharedpack-sdot.md](13-sharedpack-sdot.md) | **현재 구현 절차** — SharedPack 구조, 검증 gate, runtime blocker |

---

## 8. 문서 사용 규칙

1. 이 폴더는 이해를 돕는 설명이며 원자료를 대체하지 않는다. Scheduling 수치의 정본은
   [16-derivepp.md](../16-derivepp.md), CPU 연산 경로의 정본은
   [18-compute-path.md](../18-compute-path.md)와
   [19-sharedpack-implementation.md](../19-sharedpack-implementation.md), 그리고 run artifact다.
2. `18 §6b`의 초기 K-sweep 해석은 `18 §6c`와
   [12-cpu-compute-path.md](12-cpu-compute-path.md)의 교락 분리 결과로 대체한다.
3. 새로운 연산 최적화는 대상 op의 production 시간 비중, Amdahl 상한, 강한 baseline,
   교락 대조군과 bit-identical gate를 먼저 고정한다.
4. kernel-only upper bound, pack-inclusive op 시간, 단일 노드 E2E, 분산 E2E를 서로
   바꾸어 인용하지 않는다.
