# DerivePP — 저속 Ethernet CPU 클러스터의 Prefill 계층적 Dataflow Co-design

## 논문 메시지

> 저속 Ethernet CPU 클러스터의 prefill 은 **두 계층에서** 비효율이 발생한다.
> **노드 사이**에서는 직렬 실행과 통신 대기로 CPU 가 놀고,
> **노드 안**에서는 SDOT worker 가 동일한 activation 을 중복 packing 하여
> cache capacity 를 소모한다.
>
> **Wave Pipeline** 은 노드 간 연산·통신을 겹치고,
> **SharedPack** 은 노드 내 packed activation 을 한 번만 생성해 worker 가 공유한다.

두 기법을 관통하는 공통 원리는 하나다.

> **데이터를 소비자마다 반복 준비하지 않고, 적절한 계층에서 한 번 materialize 한 뒤
> 파이프라인으로 전달하거나 공유한다.**

```
클러스터 계층
  Prompt ─▶ microbatch wave ─▶ stage 0 ─▶ stage 1 ─▶ ... ─▶ stage 7
                                  │
                                  ▼
노드 내부 계층
  F32 activation ─▶ Q80x4 pack 1회 ─┬─▶ SDOT worker 0
                                    ├─▶ SDOT worker 1
                                    ├─▶ SDOT worker 2
                                    └─▶ SDOT worker 3
```

---

## 기여 세 가지

| # | 기여 | 계층 |
|---|---|---|
| 1 | **Low-bandwidth CPU Wave Pipeline** — tensor collective 를 피하고 prompt microbatch 를 stage 사이에 흘려 연산과 통신을 겹치는 exact prefill 실행 구조 | 노드 간 |
| 2 | **Executor-hoisted SharedPack** — 동적 activation packing 을 SDOT worker 내부에서 실행 그래프로 끌어올려, Q80×4 표현을 한 번 생성하고 여러 worker 가 공유하는 정확한 연산 경로 | 노드 내 |
| 3 | **End-to-end hierarchical co-design** — 로컬 projection 1.205× 개선이 N=8 TTFT 1.436× 로 이어지는 현상을 stage recurrence 와 ablation 으로 설명하고, 공개 CPU 분산 baseline 과 직접 비교 | 결합 |

---

## 문서 목록

| 문서 | 내용 |
|---|---|
| [01-problem.md](01-problem.md) | 두 계층의 비효율 — 왜 기존 CPU 분산 prefill 이 확장되지 않는가 |
| [02-wave-pipeline.md](02-wave-pipeline.md) | **기여 1.** 노드 간 wave 실행 |
| [02b-microbatch-size.md](02b-microbatch-size.md) | `B` — wave 의 실행 파라미터 (최적화 "축" 이 아니다) |
| [03-sharedpack.md](03-sharedpack.md) | **기여 2.** 노드 내 exact shared packing |
| [04-codesign.md](04-codesign.md) | **기여 3.** 2×2 ablation 과 recurrence 검증 |
| [05-design-space.md](05-design-space.md) | 왜 TP/CP/core split/복잡한 partition 이 아닌가 |
| [07-measurement.md](07-measurement.md) | 측정 방법론과 검증 계층 |
| [08-status.md](08-status.md) | 현재 상태, 진행 중, 리스크 |

---

## 현재 성과

| 항목 | 수치 |
|---|---|
| 공개 baseline — llama.cpp RPC (N=2) | **0.41×** (느려진다) |
| 공개 baseline — 공식 distributed-llama (N=4) | **0.44×** (느려진다) |
| Wave Pipeline — 로컬 llama.cpp 대비 (S=447 / S=1789) | **4.68× / 4.57×** |
| Wave Pipeline — 자기 단일 노드 대비 (8노드) | **6.64×**, 효율 83% |
| SharedPack — wave 시스템 위 추가 (N=8 TTFT) | **1.436×**, 로짓 비트 동일 |

> **금지.** `4.57 × 1.436 = 6.56×` 같은 곱셈을 하지 않는다. 베이스라인·세션·조건이
> 다르다. 최종 시스템 대 llama.cpp 는 동일 조건에서 다시 재야 한다.

---

## novelty 를 어떻게 주장하는가

개별 기법만 보면 강하지 않다. microbatch wave PP 는 GPipe 계열과 구조적으로
유사하고, shared packing 은 BLAS/quantized GEMM 에서 알려진 원리다.

그래서 **이렇게 주장하지 않는다.**

> ~~새로운 pipeline 알고리즘과 새로운 packing 알고리즘을 개발했다~~

**시스템 공동설계로 주장한다.**

> 기존 공개 CPU 분산 구현은 저속 Ethernet 에서 단일 노드의 0.41~0.44× 에 머물렀다.
> 본 연구는 **inter-node wave execution** 과 **intra-node exact shared packing** 을
> 공동 설계하여 실제 CPU 클러스터 prefill 을 확장 가능하게 만들었다.

시스템 논문의 novelty 는 각 구성요소가 완전히 새로운가로만 결정되지 않는다.
네 가지가 중요하다.

| # | 항목 | 현재 |
|---|---|---|
| 1 | 기존 시스템이 왜 확장되지 않는지 **구체적으로 규명** | 확보 (소스 확인) |
| 2 | 두 계층을 연결한 **일관된 설계** | 확보 |
| 3 | 공개 baseline 보다 큰 **end-to-end 개선** | 확보 |
| 4 | 여러 입력·모델·실행 조건에서 **재현** | **진행 중** |
