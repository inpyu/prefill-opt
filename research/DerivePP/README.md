# DerivePP — 저속 Ethernet CPU 클러스터의 Prefill 계층적 Dataflow Co-design

## 정확성 범위 (먼저 명확히)

```
SharedPack 은 동일한 N, B, wave 설정, activation-transfer 설정의 BASE 에 대해
bit-identical 하다.
```

**Wave Pipeline 자체를 "exact" 라고 부르지 않는다.** wave on 과 wave off 는
로짓이 다르다(`c52e50e37e65` vs `8b8178a50a97`). wave 경로가 활성값을 q80 로
전송하기 때문이며 SharedPack 이전부터 있던 성질이다.
정확성 비교는 **반드시 같은 wave 구성 안에서** 한다.

Wave Pipeline 에 대해 말할 수 있는 것은 **"토큰 생략이나 근사 attention 을
사용하지 않는 deterministic quantized prefill"** 이다.

---

## 논문 메시지

> 저속 Ethernet CPU 클러스터의 prefill 은 **세 계층에서** 비효율이 발생하며,
> 하나를 없애면 다음 계층이 드러난다.
>
> **노드 사이**에서는 직렬 실행과 통신 대기로 CPU 가 논다.
> 그것을 없애면 **노드 안**에서 SDOT worker 가 동일한 activation 을 중복
> packing 하여 cache capacity 를 소모하는 것이 드러난다.
> 그것마저 없애면 **긴 문맥에서 attention 이 지배 병목**이 된다.

세 기법을 관통하는 공통 원리는 하나다.

> **CPU prefill 에서 반복적으로 소비되는 데이터를 각 소비자가 다시 준비하지 않는다.
> 데이터를 생성하는 시점에 소비 방식에 맞는 표현으로 한 번 materialize 하고,
> 노드·스레드·query head 사이에서 공유한다.**

```
클러스터 계층   Wave Pipeline
  Prompt ─▶ microbatch wave ─▶ stage 0 ─▶ stage 1 ─▶ ... ─▶ stage 7
                                  │
                                  ▼
Projection 계층  SharedPack-SDOT
  activation ─▶ Q80x4 pack 1회 ─┬─▶ SDOT worker 0
                                ├─▶ SDOT worker 1
                                ├─▶ SDOT worker 2
                                └─▶ SDOT worker 3
                                  │
                                  ▼
Attention 계층   RoleSplit-KV        ← 개발 착수 단계
  V projection ─▶ blocked V append 1회 ─┬─▶ GQA query head 0
                                        ├─▶ GQA query head 1
                                        ├─▶ GQA query head 2
                                        └─▶ GQA query head 3
```

### 병목이 이동하는 서사

```
저속 링크에서 기존 분산 실행이 확장되지 않는다
      ↓
Wave Pipeline 으로 노드 간 idle 을 제거한다
      ↓
projection 의 thread-private pack 이 새로운 병목으로 드러난다
      ↓
SharedPack 으로 노드 내 중복 footprint 를 제거한다
      ↓
긴 문맥에서는 attention 이 새로운 지배 병목이 된다   ← 지금 여기
      ↓
RoleSplit-KV 로 KV locality 와 GQA 재사용을 회수한다
      ↓
짧은 문맥부터 긴 문맥까지 확장되는 CPU prefill 시스템
```

---

## 기여 세 가지

| # | 기여 | 계층 |
|---|---|---|
| 1 | **Low-bandwidth CPU Wave Pipeline** — tensor collective 를 피하고 prompt microbatch 를 stage 사이에 흘려 연산과 통신을 겹치는 prefill 실행 구조. 토큰 생략이나 근사 attention 을 쓰지 않는다 | 노드 간 |
| 2 | **Executor-hoisted SharedPack** — 동적 activation packing 을 SDOT worker 내부에서 실행 그래프로 끌어올려, Q80×4 표현을 한 번 생성하고 여러 worker 가 공유하는 연산 경로. 동일 구성의 BASE 에 대해 bit-identical | 노드 내 |
| 3 | **K/V Role-Split GQA Attention** *(개발 착수)* — K 는 `QK^T` 내적에 맞춰 token-major 로 두고, V 는 `AV` 누적에 맞춰 feature-block 으로 저장하며 append 시점에 직접 기록한다. `kvMul` 개 GQA query head 가 레지스터에 올린 V 하나를 공유 | attention |
| 4 | **End-to-end hierarchical co-design** *(검증 중)* — Wave Pipeline 위에서 SharedPack 이 N=8 TTFT 를 추가로 **1.436×** 단축했다. 두 계층의 interaction 과 critical-path 변화는 2×2 ablation 및 recurrence 분석으로 **검증 중이다** | 결합 |

> **기여 3 은 아직 가설이다.** 지금 확정 가능한 것은 기여 1 과 2 이고,
> "두 계층이 서로를 강화한다" 는 2×2 앵커와 recurrence 분해가 끝난 뒤에만
> 확정 기여로 올린다. → [08-status.md](10-status.md) §8.4

---

## 문서 목록

| 문서 | 내용 |
|---|---|
| [01-problem.md](01-problem.md) | 세 계층의 비효율 — 왜 기존 CPU 분산 prefill 이 확장되지 않는가 |
| [02-wave-pipeline.md](02-wave-pipeline.md) | **기여 1.** 노드 간 wave 실행 |
| [03-microbatch-size.md](03-microbatch-size.md) | `B` — wave 의 실행 파라미터 (최적화 "축" 이 아니다) |
| [04-sharedpack.md](04-sharedpack.md) | **기여 2.** projection 계층 — exact shared packing |
| [05-attention-layer.md](05-attention-layer.md) | **기여 3.** attention 계층 — 세 번째 병목과 RoleSplit-KV *(개발 중)* |
| [06-codesign.md](06-codesign.md) | **기여 4.** ablation 과 recurrence 검증 *(검증 중)* |
| [07-design-space.md](07-design-space.md) | 왜 TP/CP/core split/복잡한 partition 이 아닌가 |
| [08-evaluation.md](08-evaluation.md) | 평가 — 실험 구성, 공개 baseline 비교, ablation |
| [09-measurement.md](09-measurement.md) | 측정 방법론과 검증 계층 |
| [10-status.md](10-status.md) | 현재 상태, 진행 중, 리스크 |

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
