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

구현 완료, 검증 중
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

## 3. 개발·검증 범위

### 구현을 완료한 것

1. Down 앞에 별도 `OP_PACK_Q80X4` 추가
2. 네 worker가 shared buffer의 서로 다른 batch-row group을 병렬로 pack
3. executor op boundary를 synchronization으로 사용
4. `block_matmul_w2`가 shared packed buffer를 읽도록 연결
5. 기존 private-pack 경로와 GEMV tail fallback 유지
6. Q/K/V와 Gate/Up이 projection family별 packed buffer를 공유
7. 동일 바이너리에서 `DLLAMA_SHARED_PACK=0/1`로 baseline과 SharedPack 전환
8. decode가 섞이지 않는 `[PREFILL ONLY]` op profile
9. N=1, B=16/B=32 production logits bit-identical 검증

### 지금 검증하는 것

```text
동일 바이너리 paired A/B
  → prefill-only op 시간과 prefillMs 확정
  → 열·DVFS·세션 드리프트를 anchor로 보정
  → N=8 wave pipeline 정확성과 E2E 검증
```

그다음 선택 과제는 F32→Q80 producer가 downstream SDOT용 Q8×4 layout을 직접 생성하는
것이다. 다만 현재 pack 비용이 prefill op 시간의 약 0.3%라 성능 기여는 작다. 단순히
novelty를 꾸미기 위해 넣지 않고, 데이터 흐름 단순화와 일반성의 가치가 있을 때 진행한다.

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
| Down용 buffer·pack op net builder 배선 | 완료 |
| Gate/Up, Q/K/V family 공유 | 완료 |
| valid N=1 logits | B=16/B=32, 6회 bit-identical |
| pack-inclusive production op 성능 | 사전등록 기준 통과 |
| 동일 바이너리 runtime A/B flag | 완료 |
| prefill-only op profile | 완료 |
| paired anchor A/B | **진행 중 — 최종 판정 전** |
| N=8 wave 정확성·E2E | 미완료 |
| F32 producer→Q8×4 직접 생성 | 미구현 |

초기 구현에서는 다음 type mismatch가 있었다.

```text
Unsupported CPU op code: PACK_Q80X4,
quant: Q80_Q80_F32,
op name: block_pack_dq
```

packed buffer를 `F_32` byte container로 선언했지만 실제 내용은 opaque `block_q8_0x4`라
런타임 quant-type과 handler가 맞지 않았던 문제다. 이 blocker는 해소됐고 production
그래프가 완주한다. 최종적으로 명시적 packed storage type 또는 raw workspace로 정리할
기술 부채는 남아 있다.

현재 확정 범위는 다음과 같다.

```text
확정: Down pack-inclusive op 1.64×
확정: 기존 누적 op profile 비교 34,346.5 → 28,511.0 ms = 1.205×
확정: 테스트한 N=1 B=16/B=32 logits bit-identical
진행: 동일 바이너리·prefill-only paired anchor로 1.205×의 재확정
미확정: N=8 wave pipeline 정확성과 E2E
```

기존 1.205배 표는 서로 다른 세션의 누적 op profile 비교였고 decode도 포함했다. 현재
`DLLAMA_SHARED_PACK=0/1`과 `[PREFILL ONLY]` profiler를 추가해 이 두 한계를 제거한 A/B를
수행 중이다. 측정 해석은 [14-performance-validation.md](14-performance-validation.md)를
따른다.

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

초기 누적 op profile 비교에서는 1.205배로 연산 계층 목표를 통과했다. 다만 최종 논문 수치는
진행 중인 동일 바이너리 paired A/B가 끝난 뒤 고정한다. 부분 결과에서 baseline과
SharedPack의 차이는 선명하지만, 온도 상승과 2.0~2.4 GHz DVFS가 함께 관측됐으므로 완료 전
수치와 안정성 주장을 확정하지 않는다.

---

## 6. 현재 방향을 이해하는 가장 짧은 읽기 순서

```text
이 README
  → 12-cpu-compute-path
       왜 WCEP/QCFuse/KP-SDOT이 탈락했고
       K-sweep의 교락을 어떻게 분리했는가
  → 13-sharedpack-sdot
       SharedPack을 executor와 net builder에 어떻게 넣었는가
       정확성·성능 gate는 무엇인가
  → 14-performance-validation
       paired anchor, 열/DVFS, prefill-only op와 E2E를 어떻게 구분하는가
  → 18-compute-path
       원자료 수치와 시간순 실험 기록
  → 19-sharedpack-implementation
       source 변경과 인수인계 상태
  → 20-measurement-environment
       이 클러스터에서 새 실험을 실행하는 운영 규칙
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
  → 14-performance-validation
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
| [13-sharedpack-sdot.md](13-sharedpack-sdot.md) | **현재 구현·결과** — SharedPack 구조, projection 확장, 정확성·성능 gate |
| [14-performance-validation.md](14-performance-validation.md) | **현재 측정 해설** — anchor 보정, 열/DVFS, 계측 층위와 재현 artifact |

---

## 8. 문서 사용 규칙

1. 이 폴더는 이해를 돕는 설명이며 원자료를 대체하지 않는다. Scheduling 수치의 정본은
   [16-derivepp.md](../16-derivepp.md), CPU 연산 경로의 정본은
   [18-compute-path.md](../18-compute-path.md)와
   [19-sharedpack-implementation.md](../19-sharedpack-implementation.md), 그리고 run artifact다.
   플랫폼 제약과 실행 규칙의 정본은 [20-measurement-environment.md](../20-measurement-environment.md)다.
2. `18 §6b`의 초기 K-sweep 해석은 `18 §6c`와
   [12-cpu-compute-path.md](12-cpu-compute-path.md)의 교락 분리 결과로 대체한다.
3. 새로운 연산 최적화는 대상 op의 production 시간 비중, Amdahl 상한, 강한 baseline,
   교락 대조군과 bit-identical gate를 먼저 고정한다.
4. kernel-only upper bound, pack-inclusive op 시간, 단일 노드 E2E, 분산 E2E를 서로
   바꾸어 인용하지 않는다.
- `../21-graph-hoisted-packing.md` — SharedPack novelty 평가와 CTGHP 일반화 설계
