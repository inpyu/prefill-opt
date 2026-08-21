# 실행기 캘리브레이션의 수학 — 배리어, 스레드 불균형, 비용 모델

[16-derivepp §7.6](../16-derivepp.md)은 op 단위 비용 모델이 production 시간을
15~25% 작게 예측한 사실과 후속 측정 계획을 기록한다. 이 문서는 그 결론에 도달하는
과정을 처음부터 유도한다.

> 핵심 질문은 “빈 배리어 하나가 몇 µs인가?”가 아니다.
> **배리어가 있을 때 각 op의 가장 느린 thread를 매번 기다리는 비용이 얼마인가?**다.

이 차이를 이해하면 다음 세 가지가 분명해진다.

1. 왜 커널 시간을 더하기만 한 모델이 production보다 낙관적인가.
2. 왜 레이어 순서대로 op를 실행한 macrobench도 격차를 해결하지 못했는가.
3. 왜 no-op step 측정은 필요하지만 그것만으로 원인 규명이 끝나지 않는가.

전제 지식은 거의 없다. 파이프라인 비용 모델은
[06-math-pipeline](06-math-pipeline.md), 측정 잡음과 held-out 검증은
[07-math-measurement](07-math-measurement.md)을 필요할 때 참고하면 된다.

---

## 목차와 권장 읽기 절차

처음 읽는다면 아래 순서를 그대로 따른다.

1. [관측된 문제](#1-무엇이-문제였나)를 보고 설명해야 할 숫자를 확정한다.
2. [세 캘리브레이션 수준](#2-세-종류의-캘리브레이션)을 구분한다.
3. [실행기 구조](#3-production-executor는-어떻게-op를-실행하나)를 따라간다.
4. [`Σ max`와 `max Σ`](#4-핵심-수학--max와-max-의-차이)를 작은 예제로 계산한다.
5. [838-step 계측](#5-실제-step-수로-가설의-크기를-검산한다)을 대입한다.
6. [no-op 실험](#6-c안--no-op-step으로-무엇을-잴-수-있나)과
   [thread-skew 실험](#7-빈-step이-못-재는-것--thread-load-imbalance)을 구분한다.
7. 마지막으로 [C→A 판정 절차](#9-최종-판정-절차--c를-gate로-a를-fallback으로)를 읽는다.

급하면 **§1 → §4-2 → §5 → §9**만 읽어도 된다.

---

# 1. 무엇이 문제였나

## 1-1. 스케줄러가 원하는 두 종류의 정확성

DerivePP는 후보 계획의 시간을 예측해 가장 빠른 계획을 고른다. 이때 예측에는 두 종류의
정확성이 있다.

| 종류 | 질문 | 현재 상태 |
|---|---|---|
| 순위 정확성 | 어떤 `B`, 노드 순서, 분할이 더 빠른가? | held-out `B=16` 선택 적중 |
| 절대값 정확성 | 실제 실행이 몇 초 걸리는가? | 10~20%, 성분별 15~25% 과소예측 |

순위만 맞아도 스케줄러는 실용적으로 동작할 수 있다. 하지만 논문에서 분석 모델의 정확성과
best-of-grid attainment를 주장하려면 절대값 편향도 설명해야 한다.

## 1-2. FFN에서 관측한 격차

단일 노드, `S=448`, `B=16`, 32레이어이면 마이크로배치 수는 다음과 같다.

```text
M = S/B = 448/16 = 28 chunks
```

FFN 전체 시간은 다음처럼 나왔다.

| 경로 | 전체 시간 | production 대비 |
|---|---:|---:|
| op 단위 커널 합 | 18.0 s | −25% |
| barrier-free 레이어 macrobench | 18.7 s | −22% |
| production stage timing | 24.0 s | 기준 |

레이어 macrobench의 단위 시간은 약 20.9 ms/layer/chunk다.

```text
20.9 ms × 32 layers × 28 chunks = 18.7 s
```

production과의 차이는 다음과 같다.

```text
전체 잔차 = 24.0 − 18.7 = 5.3 s
청크당 잔차 = 5.3 s / 28 = 189 ms/chunk
레이어·청크당 잔차 = 189 ms / 32 = 5.9 ms
```

우리가 찾아야 하는 원인은 **청크마다 약 189 ms**를 만들어야 한다. 후보를 검토할 때는
항상 이 크기와 비교해야 한다.

> “6 ms가 크다”처럼 절대값만 보면 안 된다. 설명해야 할 잔차 189 ms 중 몇 %인지
> 계산해야 원인 가설의 크기가 맞는지 판단할 수 있다.

## 1-3. 18.0초와 18.7초를 과대 해석하면 안 된다

두 값의 차이는 0.7초, 약 3.7%다. 그러나 동일 조건의 paired A/B가 아니다.

| 18.0 s | 18.7 s |
|---|---|
| `calibrate.cpp` | `calibrate_layer.cpp` |
| 한 projection 형상을 반복 | 실제 레이어 op 순서를 실행 |
| 한 가중치 형상 중심 | 가중치 사본 4개를 순회 |

calibrator의 세션 분산도 약 2.8%다. 따라서 3.7% 차이를 전부 “op 교대가 만든 비용”으로
귀속하기 어렵다.

안전한 결론은 하나다.

> op 합과 barrier-free 레이어 macrobench **둘 다** production보다 22% 이상 작다.
> 레이어 단위로 올리는 것만으로는 production 비용 함수가 되지 않는다.

---

# 2. 세 종류의 캘리브레이션

세 방식은 비슷해 보이지만 서로 다른 질문에 답한다.

## 2-1. 수준 1 — op microbenchmark

```text
같은 GEMM → 같은 GEMM → 같은 GEMM → ...
```

장점:

- 특정 커널의 처리율을 깨끗하게 측정한다.
- `B`에 따른 타일 포화점과 `B=24` notch를 볼 수 있다.
- 노드별 GEMM/attention 순위 역전을 측정할 수 있다.

빠지는 것:

- 실제 op 순서
- KV append
- graph buffer copy
- executor step 경계
- op별 thread load imbalance가 매 경계에서 노출되는 현상

따라서 이 값은 **raw kernel cost**다.

## 2-2. 수준 2 — barrier-free layer macrobenchmark

```text
RMSNorm → gate/up GEMM → SwiGLU → down GEMM → residual
```

실제 순서를 사용하지만 각 thread가 자기 op 열을 끝까지 독립적으로 실행한다.

```text
thread 0: op0 → op1 → op2 → ... → 끝
thread 1: op0 → op1 → op2 → ... → 끝
thread 2: op0 → op1 → op2 → ... → 끝
thread 3: op0 → op1 → op2 → ... → 끝
```

한 thread가 op1을 끝내면 다른 thread를 기다리지 않고 op2로 넘어간다.

이 벤치는 다음을 포함한다.

- 연산자 교대
- 여러 가중치 스트림
- 낮은 차수 연산

하지만 production executor와 같은 op 경계 동기화는 없다. 코드의 죽은 `Pool` 구조체는
제거됐고, 현재 주석도 이 역할을 명시한다.

## 2-3. 수준 3 — executor-isomorphic calibration

```text
모든 thread가 op0 종료
       ↓ barrier
모든 thread가 op1 종료
       ↓ barrier
모든 thread가 op2 종료
       ↓
```

production과 같은 다음 요소를 사용한다.

- persistent thread pool
- `doneThreadCount` atomic counter
- `currentStepIndex`
- 대기 thread의 `yield` loop
- `STEP_EXECUTE_OP`/`STEP_SYNC_NODES` 구분
- production graph와 buffer/KV 경로

이것이 방안 A다. 구현 비용은 더 크지만 실제 요청의 여러 계획을 반복하는 end-to-end
sweep은 아니다.

### 세 수준의 관계

```text
op microbench
    + 실제 op 순서
    = barrier-free layer macrobench

barrier-free layer macrobench
    + executor step 전환
    + 경계마다 노출되는 thread 불균형
    + production KV/copy 경로
    = executor-isomorphic calibration
```

각 단계에서 추가되는 항을 따로 재야 중복 집계하지 않는다.

---

# 3. Production executor는 어떻게 op를 실행하나

정본은 [`src/nn/nn-executor.cpp`](../../src/nn/nn-executor.cpp)다.

## 3-1. step의 두 종류

```text
STEP_EXECUTE_OP   실제 CPU op 실행
STEP_SYNC_NODES  노드 간 동기화/전송 실행
```

단일 노드에서는 network synchronizer가 필요 없으므로 `STEP_SYNC_NODES=0`일 수 있다.
분산 graph에서는 sync step이 추가된다.

둘을 하나의 `h_step`으로 합치면 안 된다.

| step | 포함 가능한 비용 |
|---|---|
| `EXECUTE_OP` | 함수 dispatch, atomic step 경계, thread 대기 |
| `SYNC_NODES` + fake synchronizer | sync step 자체의 executor 경계 |
| `SYNC_NODES` + network synchronizer | 경계 + socket wait + 실제 전송 + protocol 처리 |

## 3-2. 한 step이 끝나는 조건

각 thread는 같은 op를 자기 담당 범위에 대해 실행한다. 끝나면 다음을 수행한다.

```text
doneThreadCount.fetch_add(1)
```

마지막 thread만 다음 step으로 진행시킨다.

```text
doneThreadCount = 0
currentStepIndex += 1
```

먼저 끝난 thread들은 다음 조건이 바뀔 때까지 `yield`한다.

```text
currentStepIndex == 현재 step
```

따라서 step 시간은 평균 thread 시간이 아니라 **가장 늦게 끝난 thread 시간**으로
결정된다.

## 3-3. 왜 이것을 일반적인 pthread barrier로 재면 안 되나

production 경계에는 다음이 함께 들어간다.

- op 함수 호출과 복귀
- thread마다 다른 작업 분할
- atomic `fetch_add`
- 마지막 thread의 step index 갱신
- 다른 thread의 spin/yield와 재스케줄링

`pthread_barrier_wait()`만 반복하면 같은 경로를 재지 않는다. C안의 no-op 실험도 반드시
실제 `NnExecutor` 안에 `STEP_EXECUTE_OP`을 구성해 측정해야 한다.

---

# 4. 핵심 수학 — `Σ max`와 `max Σ`의 차이

## 4-1. 기호

```text
o = op 인덱스, 1...K
t = thread 인덱스, 1...P
c[o,t] = op o에서 thread t가 자기 몫을 계산하는 시간
h[o]   = op o 뒤의 순수 executor step 전환 비용
```

## 4-2. barrier-free macrobench

각 thread가 전체 op 열을 독립적으로 실행한다. thread `t`의 총시간은

```text
Σ_o c[o,t]
```

전체 벤치는 가장 늦은 thread가 끝날 때 종료된다.

```text
T_macro ≈ max_t Σ_o c[o,t]
```

한 op에서 늦었던 thread가 다음 op에서는 빨라질 수 있다. op 사이에 기다리지 않으므로
불균형이 서로 상쇄된다.

## 4-3. production executor

매 op 뒤에서 가장 느린 thread를 기다린다.

```text
T_exec ≈ Σ_o max_t c[o,t] + Σ_o h[o]
```

따라서 차이는 다음 두 항이다.

```text
T_exec − T_macro
  ≈ {Σ_o max_t c[o,t] − max_t Σ_o c[o,t]}   # 동기화가 노출한 불균형
    + Σ_o h[o]                               # 순수 step 전환 비용
```

첫 번째 항은 음수가 될 수 없다.

```text
모든 t에 대해 Σ_o c[o,t] ≤ Σ_o max_t c[o,t]
따라서 max_t Σ_o c[o,t] ≤ Σ_o max_t c[o,t]
```

## 4-4. 두 thread, 두 op 예제

다음처럼 서로 다른 op에서 느린 thread가 바뀐다고 하자.

| | thread 0 | thread 1 |
|---|---:|---:|
| op 0 | 10 ms | 1 ms |
| op 1 | 1 ms | 10 ms |

barrier-free에서는:

```text
thread 0 총시간 = 10+1 = 11 ms
thread 1 총시간 = 1+10 = 11 ms
T_macro = max(11,11) = 11 ms
```

executor에서는:

```text
op 0 = max(10,1) = 10 ms
op 1 = max(1,10) = 10 ms
T_exec = 10+10 = 20 ms       # h=0이라고 해도
```

> 빈 barrier 비용이 **0이어도 11→20 ms**가 된다.
> 이것이 no-op barrier benchmark만으로 production 격차를 설명할 수 없는 이유다.

## 4-5. 항상 같은 thread가 느리면 차이가 작다

| | thread 0 | thread 1 |
|---|---:|---:|
| op 0 | 10 ms | 1 ms |
| op 1 | 10 ms | 1 ms |

```text
T_macro = max(20,2) = 20 ms
T_exec  = max(10,1)+max(10,1) = 20 ms
```

순수 step 비용을 제외하면 같다. 따라서 핵심은 단순한 “한 thread가 느리다”가 아니라
**op 종류에 따라 병목 thread가 바뀌는가**다.

---

# 5. 실제 step 수로 가설의 크기를 검산한다

## 5-1. 왜 정적 op 개수를 세면 안 되나

모델 정의에는 조건부 경로가 있다.

- MoE 여부
- pruning 여부
- TP/PP 구성
- stage가 맡은 레이어와 서브블록
- final norm/lm_head 포함 여부

따라서 소스의 op 목록을 손으로 세기보다 생성된 executor graph를 세는 것이 정확하다.
이를 위해 `DLLAMA_DUMP_STEPS=1`이 추가됐다.

```text
🧮 [STEPS] total=838 execute_op=838 sync_nodes=0 other=0
```

## 5-2. 레이어당 step 수

레이어 밖의 고정 step 6개를 제외한다.

```text
838 − 6 = 832 layer-local steps
832 / 32 layers = 26 steps/layer
```

현재 이름 기준 대략적인 분류는 다음과 같다.

```text
attention ≈ 14 steps/layer
FFN       ≈ 12 steps/layer
```

이 값은 현재 단일 노드 Llama-3 8B graph의 값이다. 다른 모델이나 plan의 상수가 아니다.

## 5-3. 10 µs barrier 가설 — 전체 step을 쓰는 상한

설명해야 할 FFN 잔차는 189 ms/chunk다. 모든 832개 layer step을 FFN 잔차에 귀속하면
배리어 가설에 가장 유리한 상한이 된다.

```text
832 × 10 µs = 8.32 ms/chunk
8.32 / 189 = 4.4%
```

잔차 전체를 순수 step 비용으로 설명하려면:

```text
189 ms / 832 = 227 µs/step
```

## 5-4. FFN step만 쓰는 직접 비교

189 ms는 FFN component 잔차이므로 직접 대응하는 step은 약 12×32개다.

```text
K_ff = 12 × 32 = 384 steps
384 × 10 µs = 3.84 ms/chunk
3.84 / 189 = 2.0%
```

잔차 전체를 설명하려면:

```text
189 ms / 384 = 492 µs/FFN-step
```

> 새 step 계측은 순수 barrier 가설을 강화하지 않는다. 오히려 10 µs 가정이 잔차보다
> 두 자릿수 작다는 사실을 더 명확하게 만든다.

## 5-5. `sync_nodes=0`의 의미

이 결과는 단일 노드다. 네트워크가 없으므로 sync step도 없다.

분산에서는 다음을 따로 출력해야 한다.

```text
stage별 execute_op 개수
stage별 sync_nodes 개수
각 sync가 activation, KV, logits 중 무엇인지
```

`STEP_SYNC_NODES` 시간에는 실제 전송과 상류/하류 대기가 들어갈 수 있다. 이를 compute
barrier 비용에 합치면 통신을 이중 집계하거나, 반대로 누락할 수 있다.

---

# 6. C안 — no-op step으로 무엇을 잴 수 있나

## 6-1. 측정 모델

실제 executor에 계산을 하지 않는 execute step `K`개를 만든다.

```text
T_null-op(K) = a_op + h_op·K + ε
```

- `a_op`: `forward()` 시작, worker 깨우기, 종료 확인 같은 고정비
- `h_op`: no-op `STEP_EXECUTE_OP` 한 개의 평균 전환비
- `ε`: 스케줄러 잡음

`K=0`이 있어야 `a_op`와 slope를 분리할 수 있다.

## 6-2. 왜 두 점 나눗셈보다 회귀가 낫나

다음 한 점만 사용하면:

```text
h = (T(838)−T(0))/838
```

838점 하나의 이상치가 slope 전체를 움직인다. 여러 `K`를 사용해 직선을 적합하면
선형성 자체도 검증할 수 있다.

```text
K ∈ {0,1,8,32,128,384,448,832,838}
```

- 384: FFN 추정 step 수
- 448: attention 추정 step 수
- 832: 반복 레이어 전체
- 838: 고정 step을 포함한 전체 graph

## 6-3. 선형성을 먼저 본다

단순히 `R²`만 보고 끝내지 않는다.

확인할 것:

1. residual이 `K`에 따라 휘지 않는가.
2. 작은 `K`와 큰 `K`의 slope가 같은가.
3. 반복마다 thread migration이나 frequency 변화가 없는가.
4. `h_op`의 confidence interval이 충분히 좁은가.

비선형이면 `n_step·h_step` 하나로 production 비용을 표현할 수 없다.

## 6-4. 측정 절차

```text
1. CPU affinity, thread 수, governor를 production과 동일하게 고정
2. 실제 NnExecutor와 동일한 step dispatch를 사용
3. warm-up forward 폐기
4. 각 K를 교차 순서로 여러 번 실행
5. 한 측정 구간이 충분히 길도록 forward를 내부 반복
6. K별 중앙값과 분산 기록
7. T(K)=a+hK 회귀, residual plot 저장
```

`DLLAMA_DUMP_STEPS`는 graph 확인 때만 켠다. timing 중에는 출력 비용이 섞이지 않도록
끈다.

## 6-5. sync step은 별도 실험이다

```text
T_null-sync(K) = a_sync + h_sync·K + ε
```

두 조건을 분리한다.

| 조건 | 재는 것 |
|---|---|
| fake/no-op synchronizer | executor의 sync-step 경계 자체 |
| network synchronizer | 경계 + wait + transfer + protocol |

실제 network 비용은 [16-derivepp §3](../16-derivepp.md)의 `E_{k,j}` 링크 recurrence와
연결해야 한다.

## 6-6. no-op 결과의 판정

측정한 `h_op`로 두 값을 계산한다.

```text
전체-step 상한 = 832·h_op
FFN 직접 항     = 384·h_op
```

판정:

| 결과 | 의미 |
|---|---|
| FFN 잔차의 50% 이상 | 순수 step 전환이 주원인 후보 |
| 50% 미만 | 주원인 가설 기각. 독립 측정된 작은 항은 보존하되 다른 원인 계속 측정 |

어떤 경우에도 held-out 조건에서 검증하기 전에는 비용 모델에 넣지 않는다.

---

# 7. 빈 step이 못 재는 것 — thread load imbalance

## 7-1. 필요한 시간값

각 production op `o`와 thread `t`에 대해 다음을 기록해야 한다.

```text
start[o,t]   thread t가 op o를 시작한 시각
finish[o,t]  thread t가 op o를 끝낸 시각
c[o,t]       finish[o,t] − start[o,t]
```

op별로 다음을 계산한다.

```text
compute_max[o] = max_t c[o,t]
compute_min[o] = min_t c[o,t]
skew[o]        = compute_max[o] − compute_min[o]
```

## 7-2. 직접 보고 싶은 양

```text
T_sync-compute = Σ_o max_t c[o,t]
T_macro-like   = max_t Σ_o c[o,t]
Δ_imbalance    = T_sync-compute − T_macro-like
```

`Δ_imbalance`가 크면 빈 barrier 비용과 무관하게 executor가 느려진다.

## 7-3. 계측 자체가 비용을 만들지 않게 한다

모든 step에서 printf나 lock을 사용하면 재려는 현상을 바꾼다.

권장 방식:

- thread마다 미리 할당한 배열의 자기 slot에만 timestamp 기록
- timed region 안에서 문자열 formatting 금지
- 측정 뒤 마지막 thread 또는 main thread가 집계
- 전체 실행이 아니라 일부 대표 청크에서만 상세 trace
- profiling on/off의 wall-time 차이를 함께 보고

계측 오버헤드가 1%를 넘으면 보정하기보다 샘플링 빈도를 낮추는 것이 안전하다.

## 7-4. 어떤 op에서 skew가 커질 수 있나

| 원인 | 예시 |
|---|---|
| 나머지 작업 불균등 | output column 수가 thread 수로 나누어떨어지지 않음 |
| 데이터 의존 경로 | pruning, MoE, active-row 수 차이 |
| 커널별 분할 축 차이 | GEMM은 column, attention은 KV head, norm은 row 분할 |
| OS 스케줄링 | 특정 thread만 preemption |
| memory locality | thread별 NUMA는 없더라도 cache/TLB 상태 차이 |

특히 GEMM과 attention은 작업 분할 축이 다르므로, op 종류가 바뀔 때 병목 thread가 바뀔
가능성이 있다. 이것이 §4-4의 예제와 같은 구조다.

---

# 8. 남은 두 구조적 후보

executor 비용을 측정해도 잔차가 남을 수 있다.

## 8-1. KV cache write

현재 layer attention macrobench는 K/V cache를 미리 채우고 주로 읽는다. production은
매 청크에서 새 K/V를 계산한 뒤 cache에 쓴다.

```text
production: Q/K/V projection → RoPE → KV append → attention read
macrobench: prepared K/V ----------------------→ attention read
```

검증은 같은 executor graph에서 KV append만 on/off한 paired A/B로 한다.

주의:

- KV write는 attention 성분에만 해당한다.
- FFN 189 ms 잔차를 직접 설명하지 못한다.
- 전체 TTFT 잔차에는 기여할 수 있다.

## 8-2. CAST/MERGE_ADD와 pipe buffer copy

production graph에는 macrobench에 없는 이동이 존재할 수 있다.

```text
CAST
MERGE_ADD
activation pipe copy
stage bridge
```

각 op에 대해 다음을 기록한다.

```text
호출 횟수 × bytes/call × measured time/call
```

메모리 대역폭으로만 추정해 residual에 맞추지 않는다. production op를 직접 계측한다.

## 8-3. 중복 집계를 피하는 법

executor-backed A가 모든 실제 op를 포함하면 KV/copy 비용은 이미 그 안에 들어 있다.
이때 별도 측정값을 다시 더하면 이중 집계다.

```text
분해 모델을 쓸 때:
T = raw kernels + step tax + imbalance tax + KV + copy

executor macrotable을 쓸 때:
T = measured T_att/T_ff                 # 위 항들이 이미 포함됨
```

두 표현 중 하나를 선택해야 한다.

---

# 9. 최종 판정 절차 — C를 gate로, A를 fallback으로

## 9-1. 전체 흐름

```text
[완료] 실제 step 수 덤프
  838 execute, 26/layer, sync=0(single node)
        ↓
[다음] no-op STEP_EXECUTE_OP 회귀
  T(K)=a+hK, K={0,...,838}
        ↓
  h가 FFN 잔차를 얼마나 설명하는가?
        ↓
op별 thread completion 계측
  Δ_imbalance = Σ max − max Σ
        ↓
KV append / graph copy A/B
        ↓
held-out component error ≤ 15% ?
   ├─ 예: 검증된 항만 X_k에 추가
   └─ 아니오: executor-isomorphic 한 레이어 캘리브레이션(A)
```

## 9-2. C가 성공했다고 말할 조건

다음을 모두 만족해야 한다.

1. `T_null-op(K)`가 측정 범위에서 거의 선형이다.
2. `h_op`는 end-to-end 잔차가 아니라 no-op executor에서 독립 측정됐다.
3. 한 조건에서 측정한 항이 다른 `B`, prefix 또는 노드의 방향과 크기를 예측한다.
4. executor 항을 넣은 att/ff component 오차가 held-out에서 대부분 15% 이내다.

순위가 맞았다는 사실만으로 절대시간 모델이 통과한 것은 아니다.

## 9-3. C가 실패하면 A로 가는 이유

다음 중 하나면 A로 간다.

- no-op step은 작지만 production 격차는 크다.
- `T(K)`가 비선형이다.
- thread skew가 op/B에 따라 크게 변해 단일 상수로 표현되지 않는다.
- KV/copy까지 더해도 held-out 오차가 15%를 넘는다.

A는 production graph builder, `NnExecutor`, CPU op, buffer/KV 경로로 synthetic weight의
att/ff 한 서브블록을 짧게 실행한다.

## 9-4. A는 왜 zero-tuning을 깨지 않나

zero-tuning의 정의는 다음이다.

```text
하지 않음: 실제 요청을 (N,B,π,p) 후보마다 end-to-end replay
허용됨:    플랫폼 비용 함수를 만들기 위한 짧은 자동 캘리브레이션
```

A는 특정 plan의 성능을 보고 그 plan을 선택하는 sweep이 아니다. production 실행 경로를
사용해 `T_att,k(B,prefix)`와 `T_ff,k(B)`라는 플랫폼 비용 함수를 만드는 과정이다.

모델 파일도 원리적으로 필수는 아니다. 모델 config에서 형상을 읽고 synthetic weight를
생성할 수 있다.

## 9-5. held-out을 어떻게 나누나

예:

```text
calibration:
  node root, B={8,16,32}, prefix={512,2048}

held-out:
  다른 node 1개
  B={24,48}
  prefix={128,4096}
```

결과를 보고 같은 조건의 식을 다시 고치면 held-out이 아니다. 모델을 수정했다면 새 조건을
다시 미측정 상태로 남겨 검증해야 한다.

---

# 10. 자주 생기는 오해

## 오해 1. “op가 838개니까 배리어가 원인이다”

횟수만 많다고 원인이 아니다. `횟수 × 단위비용`을 잔차와 비교해야 한다.

```text
838 × 아주 작은 값 = 여전히 작은 값
```

## 오해 2. “빈 barrier가 작으면 executor 영향은 없다”

틀리다. `Σ max−max Σ` 불균형 항은 빈 barrier에 나타나지 않는다.

## 오해 3. “레이어 벤치니까 production과 같다”

연산 순서가 같다는 것과 실행기 경계가 같다는 것은 별개다.
현재 `calibrate_layer.cpp`는 의도적으로 barrier-free 대조군이다.

## 오해 4. “18.0→18.7초이므로 교대 비용은 정확히 0.7초다”

측정 경로와 가중치 조건이 다르고 차이가 세션 분산과 비슷하다. 같은 세션 paired A/B가
아니므로 정확한 인과 효과로 말할 수 없다.

## 오해 5. “분산의 sync step도 execute step과 같은 상수다”

`STEP_SYNC_NODES`는 network wait와 transfer를 포함할 수 있다. fake synchronizer와
실제 synchronizer를 분리해야 한다.

## 오해 6. “A는 실제 실행기를 쓰므로 end-to-end tuning이다”

실행기를 사용하는 것과 후보 plan을 실측 탐색하는 것은 다르다. A는 비용 함수의 자동
캘리브레이션이고, 실제 `(N,B,π,p)` 조합을 반복 실행하지 않는다.

---

# 11. 코드에서 확인할 위치

| 개념 | 위치 |
|---|---|
| step graph 생성과 `DLLAMA_DUMP_STEPS` | [`nn-executor.cpp`](../../src/nn/nn-executor.cpp) |
| `STEP_EXECUTE_OP`, `STEP_SYNC_NODES` 정의 | [`nn-executor.hpp`](../../src/nn/nn-executor.hpp) |
| atomic/yield barrier | [`nn-executor.cpp`](../../src/nn/nn-executor.cpp) worker loop |
| barrier-free layer 대조군 | [`calibrate_layer.cpp`](../../prefill_bench/calibrate_layer.cpp) |
| op 단위 calibrator | [`calibrate.cpp`](../../prefill_bench/calibrate.cpp) |
| 최종 비용식 `X_k` | [16-derivepp §2.3](../16-derivepp.md) |
| 실측과 C→A 판정 | [16-derivepp §7.6](../16-derivepp.md) |
| 개발 순서와 kill criteria | [16-derivepp §10~11](../16-derivepp.md) |

---

# 요약

| 질문 | 답 |
|---|---|
| 왜 op 합이 낙관적인가 | production executor와 KV/copy 경로가 빠져 있다 |
| 왜 레이어 macrobench도 부족한가 | op 순서는 같지만 thread가 op 경계에서 만나지 않는다 |
| barrier-free 시간 | `max_t Σ_o c[o,t]` |
| executor 시간 | `Σ_o max_t c[o,t] + Σ_o h[o]` |
| 빈 barrier가 못 재는 것 | `Σ max−max Σ`, 즉 경계가 노출하는 thread 불균형 |
| 실제 step 수 | 단일 노드 838 execute, 레이어당 26개, att≈14/ff≈12 |
| 10 µs의 설명력 | 전체-step 상한 4.4%, FFN 직접 비교 2.0% |
| C의 역할 | 실제 executor no-op step의 순수 전환비를 독립 측정하는 진단 gate |
| A의 역할 | C로 남은 오차를 설명하지 못할 때 production과 동형인 비용 함수를 측정 |
| zero-tuning 유지 조건 | 실제 후보 plan의 end-to-end sweep을 하지 않는다 |

> 최종 교훈: **커널 비용을 정확히 재는 것과 프로그램 비용을 정확히 재는 것은 다르다.**
> 멀티스레드 실행에서는 연산량뿐 아니라 “thread들이 어디서 다시 만나는가”가 실행시간의
> 일부다.

← 이전: [07-math-measurement](07-math-measurement.md)
