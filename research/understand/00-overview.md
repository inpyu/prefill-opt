# DerivePP — CPU Cluster Prefill의 축-판별 스케줄러

이 문서는 현재 연구의 **정본 진입점**이다. 시간 순으로 쌓인 실험 기록이 아니라,
논문이 답하려는 질문 → 현재까지의 증거 → 개발 계획 순서로 읽힌다.

세부 수식과 원자료는 [16-derivepp.md](../16-derivepp.md)에, 용어와 보조 유도는
이 폴더의 문서에 둔다.

---

## 1. 한 문장

> **DerivePP는 저속 링크의 CPU cluster에서 단일 요청 LLM prefill에 실제로 중요한
> 스케줄링 축만 판별해 활성화하고, end-to-end 후보 sweep 없이 실행계획을 결정한다.**

여기서 실행계획은 적어도 다음을 포함한다.

| 기호 | 결정 | 뜻 |
|---|---|---|
| `N` | 활성 PP stage 수 | 이번 요청에 사용할 node/stage 구성 |
| `B` | microbatch 폭 | 한 stage가 한 번에 처리할 토큰 수 |
| `p` | 연속 sublayer 경계 | att/ff 블록을 stage에 나누는 방법 |
| `π` | node 순서 | 선택한 node와 pipeline 연결 순서 |

과거의 DerivePP는 이 모든 축을 항상 공동 최적화하는 DP를 목표로 했다. 현재 방향은
다르다. **모든 축을 최적화할 이유가 있는지부터 판정한다.**

---

## 2. 왜 CPU cluster prefill인가

LLM inference에는 두 구간이 있다.

```text
prefill : S개 프롬프트 토큰을 모든 레이어에 통과시켜 KV cache를 생성
decode  : 생성 토큰 하나를 추가하고 KV cache를 읽음
```

사용자가 보는 첫 응답 시간(TTFT)은 prefill이 지배한다. 긴 prompt에서는 causal
attention이 이전 KV 전체를 읽으므로, 뒤쪽 토큰일수록 비용이 커진다.

현재 대상은 다음 조건을 동시에 가진 commodity CPU cluster다.

- 모델은 한 node에 모두 올리기 어렵거나 prefill이 너무 느리다.
- node 사이 Ethernet은 tensor parallelism처럼 매 레이어 큰 tensor를 교환하기에는 느리다.
- 따라서 레이어 구간을 node에 나누고, token chunk를 시간축으로 흘리는 **pipeline
  parallelism(PP)** 이 기본 축이다.
- CPU의 quantized GEMM은 batch 폭에 따라 비매끄러운 처리율 knee를 가진다. 현재
  플랫폼에서는 `B=16`이 첫 효율 구간이다.

토큰을 `B`개씩 나눈다고 context parallelism(CP)이 되는 것은 아니다. 모든 chunk가
`stage0 → stage1 → …`의 서로 다른 레이어 구간을 통과하므로 현재 기본 구조는 PP다.

기초 개념은 [04-concepts.md](04-concepts.md), PP의 시간 모델은
[06-math-pipeline.md](06-math-pipeline.md)를 먼저 읽으면 된다.

---

## 3. 문제 — 자유변수는 많지만, 실제 레버는 적을 수 있다

겉으로는 `(N, B, p, π)`의 조합이 매우 크다. 예를 들어 64개 att/ff subblock을 8개
stage로 연속 분할하는 경우만 해도 `C(63,7)`개다. 후보마다 실제 prompt를 실행해
고르는 방식은 느릴 뿐 아니라, thermal/session drift 때문에 비교 자체가 흔들린다.

그러나 이 변수들은 TTFT의 **서로 다른 항**을 바꾼다. 이것을 구분하지 않으면 `joint
optimization`이라는 말만 남고, 왜 어떤 기능은 구현하지 않고 어떤 기능은 계속 개발하는지
설명할 수 없다.

```text
TTFT(prefill) ≈ 첫 microbatch의 fill 시간
             + 뒤따르는 microbatch의 병목 stage 대기 시간
             + stage 경계 activation 전송 및 마지막 응답 반환 시간
```

| 축 | 무엇을 바꾸는가 | TTFT를 바꿀 수 있는 경로 | 현재 DerivePP의 행동 |
|---|---|---|---|
| `B` | microbatch의 토큰 폭, 즉 `M=ceil(S/B)` | 작은 `B`에서는 GEMM 타일·호출 고정비가 손해이고, 큰 `B`에서는 `M`이 줄어 PP fill/drain 버블이 커진다 | **평가한다.** calibration table의 `B` 후보만 합성 비용으로 비교한다 |
| `p` | 각 stage가 맡는 연속 att/ff subblock 경계 | stage별 attention/FFN 비율과 병목 stage가 바뀌어, fill과 steady-state 병목이 이동할 수 있다 | 현재 조건에서는 **균등 분할로 고정**한다 |
| `N` | 실제 활성 stage/node 수 | stage가 늘면 stage당 계산은 줄지만 fill/drain과 activation 경계는 늘며, 모델이 들어가는 worker set도 달라질 수 있다 | **미판정.** memory-feasible worker set을 통제한 검증 뒤에만 선택한다 |
| `π` | 같은 node들의 선택·pipeline 순서 | node별 GEMM/attention 속도와 인접 link 대역폭이 달라 계산 병목과 통신 대기가 함께 바뀐다 | **미판정.** placement A/B와 link trace가 확인되기 전에는 고정 순서를 쓴다 |
| `r` | 한 stage 안에서 attention·FFN을 나눠 돌리는 core split 비율 | 이론적으로는 stage 내부 overlap을 만들 수 있지만, 실제로는 같은 CPU core·DRAM·cache를 경쟁한다 | **사용하지 않는다.** CoRePP 측정에서 이득이 없었다 |
| `g` | 두 node를 CP/TP island 하나로 묶는 group 크기 | service time이 실제로 줄어야 PP stage 수 감소의 bubble 이득을 얻을 수 있다. 동시에 Q/O/KV 통신이 추가된다 | **prototype하지 않는다.** 현재 조건의 낙관적 상한도 구현 gate를 넘지 못했다 |

`N, B, p, π`는 현재 PP 실행계획의 직접 결정값이다. `r`과 `g`는 그 실행 구조 자체를
바꾸는 **대안 축**이다. 따라서 `r`과 `g`를 끈다고 해서 planner가 정보를 잃는 것이 아니라,
현재 플랫폼에서 이득보다 간섭·통신 비용이 큰 구조 변경을 배제하는 것이다.

하지만 더 중요한 사실은 반대쪽에 있다.

> **많은 자유변수 중 일부는 실제 TTFT를 거의 바꾸지 않는다.**

현재까지의 증거는 다음과 같다.

| 축 | 현재 판정 | 이 판정이 뜻하는 것 |
|---|---|---|
| `B` | **활성** | `S=1789, N=8`의 paired J2에서 calibration table의 `B=16`이 기존 기본값 `B=32`보다 **1.136×** 빨랐다. 따라서 이 축은 고정하면 안 되고 후보 table을 평가해야 한다 |
| `p` | 이 조건에서 **비활성에 가까움** | `S=1789, N=8, B=32`의 anchor A/B에서 당시 DP plan은 균등 분할의 `0.999×`였다. 새 calibration의 다른 후보도 약 2% 수준이며 재현성이 약하다. 이 조건에서는 균등 분할을 기본값으로 둔다 |
| `r` (CoRePP core split) | **비활성** | stage 내부 att/ff lane 분할의 측정 이득이 `G=0.97`이었다. 병렬화가 아니라 core/DRAM 간섭만 추가했다 |
| `g=2` (CP2 island) | **현재 조건에서 prototype 기각** | 균형 PP라는 조건의 낙관적 bubble 상한과 Ethernet Q/O/KV 통신 회계를 합쳐도 사전 구현 gate를 넘지 못했다 |
| `N` | **미확정** | `N`을 바꾸면 worker 수·메모리 가능성·node 종류가 함께 바뀔 수 있다. 이 교락을 분리한 scaling audit 전에는 자동 선택을 주장하지 않는다 |
| `π` | **미확정** | link topology와 연산자별 node 속도는 이종이지만, node 순서를 바꿔 얻는 TTFT 이득은 아직 end-to-end로 검증하지 않았다 |

따라서 논문의 질문은 더 이상 “가장 복잡한 joint DP를 만들 수 있는가”가 아니다.

> **어떤 축이 이 요청·모델·플랫폼에서 충분히 큰 레버인지, 그리고 언제 그 축의 탐색을
> 생략해도 되는지를 calibration으로 판별할 수 있는가?**

---

## 4. 해결 방법 — DerivePP와 AxisCert

**DerivePP**는 시스템과 논문 이름이다. **AxisCert**는 DerivePP 안에서 축을 켜고 끄는
결정 절차다. AxisCert가 답하는 질문은 “어느 후보가 가장 빠른가?”보다 먼저,
**“이 축을 바꾸는 것이 TTFT를 실용 임계값 이상 바꿀 가능성이 있는가?”**다.

현재 `B`와 `p`에 대한 residual 계산은 분석 스크립트로 검증했으며, 이를 planner가
자동으로 호출해 축을 고정/열거하도록 통합하는 구현은 남은 개발 항목이다. 따라서 아래
그림은 현재 시스템의 동작과 최종 DerivePP의 목표 동작을 함께 나타낸다.

```text
모델 구조 + production-path calibration + link calibration
                         │
                         ▼
                 AxisCert: 축별 leverage 판정
                         │
     ┌───────────────────┼───────────────────────┬───────────────────────┐
     ▼                   ▼                       ▼                       ▼
  active `B`       inactive `p`, `r`       rejected `g`         uncertified `N`, `π`
     │                   │                       │                       │
     ▼                   ▼                       ▼                       ▼
  B table만         `p`는 균등, `r`는 off     CP/TP island를       자동 선택하지 않음;
  합성 비용 평가                              만들지 않음            별도 audit/gate 필요
     │                   │                       │                       │
     └───────────────────┴───────────────────────┴───────────────────────┘
                                         │
                                         ▼
현재 안전한 plan: `B`만 table에서 선택하고, `p=p0`, `r=off`, `g=off`,
                  `N=N0`, `π=π0`는 검증된 고정 구성 사용
                                         │
                                         ▼
                           plan manifest와 Schedule Atlas
```

### 4.1 calibration은 하지만 end-to-end 후보 sweep은 하지 않는다

`sweep-free`는 측정이 없다는 뜻이 아니다.

- 한다: node별 production executor 비용, kernel `B` table, link payload/time, 실행 환경을
  짧게 측정한다.
- 하지 않는다: 실제 요청을 `(N,B,p,π)` 후보마다 끝까지 실행하고 가장 빠른 값을 고른다.

캘리브레이션은 planner의 입력이고, end-to-end 실행은 마지막 검증용 oracle이다.

### 4.2 절대 예측 오차가 아니라 차분 결정 오차를 쓴다

현재 executor 비용 모델에는 plan 공통의 절대 편향이 남을 수 있다. 이때 절대 TTFT
오차만으로 축을 판단하면 모든 후보의 구간이 겹쳐 아무 축도 끌 수 없다.

AxisCert는 기준 계획 `q0`와 후보 `q`의 **상대 오차**를 사용한다.

```text
e_x(q; q0)
  = log[T_meas(q) / T_meas(q0)]
  - log[T_pred(q) / T_pred(q0)]
```

`x`는 바꾼 축이다. `p`를 바꿨다면 `N,B,π`는 고정한 paired experiment에서
`e_p`를 얻는다. `B`를 바꿨다면 J2처럼 동일 세션의 anchor block에서 `e_B`를 얻는다.

이 축별 residual bound를 `δ_x`라 할 때:

```text
predicted log speedup = log[T_pred(q0) / T_pred(q)]

상한이 실용 임계값 이하  → 축 비활성
하한이 실용 임계값 초과  → 축 활성
그 사이                  → indeterminate; targeted calibration 또는 보수적 기본값
```

따라서 DerivePP는 “절대시간을 완벽히 맞혀야만 하는 planner”가 아니라, **계획 간 순위를
맞힐 만큼 정확한 planner**를 목표로 한다. 자세한 유도는
[11-axis-cert.md](11-axis-cert.md)에 있다.

### 4.3 DP의 위치가 바뀐다

completion-vector DP는 여전히 필요하다. causal attention 때문에 뒤쪽 chunk의 비용이
커지고, 그때 단일 scalar stage time만으로는 pipeline 완료시각을 계산할 수 없다.

그러나 DP는 논문의 목적이 아니라 **활성화된 partition/placement 축을 평가하는 정확한
계산 수단**이다. `p` gate가 닫히면 DP는 실행하지 않고 균등 분할을 쓴다.

DP와 dominance pruning의 수학은 [09-math-planner.md](09-math-planner.md)를 본다.

---

## 5. 왜 몇몇 구조 변경을 미리 끌 수 있는가

모든 새 병렬화가 새 연산 용량을 만드는 것은 아니다. 예를 들어 fixed worker budget에서
CP2 island는 두 worker를 한 logical stage에 묶는다. 기존 PP가 이미 균형이고, 묶은
group의 service time도 줄지 않는다는 조건에서는 `N→N−1`로 줄어드는 bubble만 이득이다.

```text
T_N / T_(N−1) ≤ (M + N − 1) / (M + N − 2)
순수 bubble 이득 = 1 / (M + N − 2)
```

긴 prompt에서는 `M`이 크므로 이 상한은 0으로 수렴한다. CP에는 query/partial-output/KV
append 통신이 추가되므로 실제 이득은 이보다 작다. 이 논거는 **이미 균형이고 service
time이 줄지 않는 경우**에만 적용한다. 따라서 AxisCert는 구조만으로 무조건 기각하지
않고, 그 적용 조건과 protocol payload를 먼저 확인한다.

이 보조정리와 CP 통신 회계는 [11-axis-cert.md](11-axis-cert.md) 및
[16-derivepp.md §12.4](../16-derivepp.md#124-cp-island--capacity-contraction-상한-구현-없이-판정)에 있다.

---

## 6. 최종 개발 계획

### Phase A — 증거와 artifact를 고정한다

1. J2를 끝내 `B=16` 대 기본 `B=32`의 paired effect와 `δ_B`를 확정한다.
2. calibration TSV, J2 raw log, planner input, model/build hash, host/governor 정보를
   run-id artifact로 보존한다.
3. 사라진 `pl_*.tsv`를 재수집하고, `δ_p` 계산에 필요한 raw profile을 복구한다.

산출물은 재현 가능한 calibration artifact다. 논문 수치가 임시 세션 파일에만 남아 있으면
다음 단계의 모든 gate가 무효다.

### Phase B — AxisCert의 차분 신뢰도를 만든다

1. J1b의 prediction/measurement 쌍으로 `δ_p`를 계산한다.
2. J2 anchor blocks로 `δ_B`를 계산한다.
3. `N`은 memory-feasible worker set을 사전 고정한 뒤 별도의 `δ_N` 실험으로 다룬다.
4. `S=7212`에서는 먼저 P0 반복으로 해당 조건에서 비교 가능한 변동성인지 확인한다.

절대 MAPE와 axis-specific paired residual을 섞지 않는다.

### Phase C — 활성 축만으로 plan을 선택한다

1. 기본 distributed-llama 설정(`nBatches=32`)을 primary baseline으로 고정한다.
2. `B` table selector와 `N` selector를 구현한다.
3. `p`, CoRePP, CP의 gate 결과를 plan manifest에 기록한다.
4. planner 선택과 measured best-of-grid의 regret를 held-out 길이에서 비교한다.

### Phase D — 일반성과 논문 주장을 검증한다

1. 3B/8B/13B, 짧음/중간/장문 길이에 대해 calibration 후 계획을 사전 등록한다.
2. core/frequency cap 및 link 조건으로 axis activation 전환을 검증한다.
3. 자연 조건에서는 꺼진 `p`가 통제된 이종성에서 켜지는지 확인한다. 켜지지 않으면
   partition은 이 논문의 핵심 기여에서 완전히 제외한다.
4. CP는 corrected capacity/protocol gate가 통과하지 않는 한 구현하지 않는다.

### Phase E — Atlas와 논문 artifact를 만든다

길이 bucket별 plan과 gate 결과를 Atlas로 저장한다. 공개 artifact에는 raw log, calibration
input, plan manifest, preregistration, oracle evaluation을 포함한다.

---

## 7. 논문에서 주장할 것과 주장하지 않을 것

### 주장할 수 있는 것

- CPU cluster prefill의 `B` 선택은 kernel knee와 pipeline geometry를 함께 봐야 하며,
  짧은 production calibration으로 이를 결정할 수 있다.
- absolute latency bias가 있어도 paired plan residual이 작다면, 축 활성화와 plan 선택은
  신뢰성 있게 할 수 있다.
- 복잡한 partition, core split, CP가 항상 이득이라는 가정을 버리고, 이득 상한 또는
  차분 신뢰도로 탐색 축을 제거할 수 있다.
- planner의 평가는 MAPE만이 아니라 regret, attainment, inactive-axis false-positive로
  해야 한다.

### 주장하지 않을 것

- 현재 natural cluster에서 partition DP가 speedup을 만들었다.
- CoRePP 또는 CP2 island를 구현해 성능을 높였다.
- `B*=B_min`이 모든 CPU·모델에 성립한다.
- 절대 TTFT를 모든 길이에서 일정 오차 이내로 맞힌다.

---

## 8. 읽는 순서

```text
00-overview (이 문서)
  → 04-concepts             # Transformer, KV cache, PP/CP/TP 용어
  → 06-math-pipeline         # M, N, B, bubble
  → 07-math-measurement      # paired experiment와 measurement discipline
  → 11-axis-cert             # 현재 논문의 결정 알고리즘
  → 16-derivepp              # 정식 비용 모델, 실측, 개발 현황
```

필요할 때만 다음 문서로 들어간다.

| 궁금한 것 | 문서 |
|---|---|
| source와 실험 개념의 대응 | [01-code-map.md](01-code-map.md) |
| 기호와 프로젝트 용어 | [02-glossary-extra.md](02-glossary-extra.md), [03-glossary-full.md](03-glossary-full.md) |
| attention/KV의 수학과 정확성 | [05-math-attention.md](05-math-attention.md) |
| executor calibration이 필요한 이유 | [08-math-executor-calibration.md](08-math-executor-calibration.md) |
| completion-vector DP와 dominance proof | [09-math-planner.md](09-math-planner.md) |
| 과거 calibration 가설과 교체 이유 | [10-calibration-history.md](10-calibration-history.md) |

시간 순 연구 기록은 참고용이다. 과거 문서의 가설이나 수치를 현재 논문 주장으로 인용하기
전에 반드시 [16-derivepp.md](../16-derivepp.md)의 최신 판정과 대조한다.
