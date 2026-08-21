# AxisCert — 어느 스케줄링 축을 실제로 탐색할 것인가

AxisCert는 DerivePP 안에서 쓰는 **축 활성화 절차**다. 목표는 모든 가능한 scheduling
knob을 최적화하는 것이 아니라, 현재 모델·요청·플랫폼에서 TTFT를 바꿀 만큼 중요한
knob만 남기는 것이다.

---

## 1. 왜 축을 먼저 판정하는가

단일 요청 CPU prefill에는 다음 선택지가 있어 보인다.

```text
N : pipeline stage 수
B : microbatch 폭
p : att/ff sublayer 경계
π : node subset과 순서
r : stage 내부 core split
g : CP/TP group 크기
```

앞 네 축은 현재 PP 실행계획의 직접 결정값이고, 뒤 두 축은 실행 구조를 바꾸는 대안이다.

| 기호 | 한 줄 정의 | TTFT에 영향을 줄 수 있는 곳 |
|---|---|---|
| `N` | 활성 pipeline stage/node 수 | stage당 계산량 대 fill/drain·경계 수의 교환 |
| `B` | microbatch 토큰 폭 | CPU kernel 효율 대 microbatch 수·PP bubble의 교환 |
| `p` | 연속 att/ff subblock 경계 | stage별 계산량과 병목 stage의 이동 |
| `π` | node subset과 pipeline 순서 | node별 연산자 비용 및 인접 link 비용 |
| `r` | 한 stage 내부의 core split | core/DRAM 경쟁을 감수한 operator overlap |
| `g` | CP/TP island의 group 크기 | service-time 감소 가능성 대 새 통신 비용 |

따라서 `p`를 inactive로 판정한다는 것은 attention/FFN 비용이 없다는 뜻이 아니다. 현재
조건에서 **경계를 옮겨도 병목 완료시각이 실용 임계값만큼 변하지 않는다**는 뜻이다.

하지만 현재 natural cluster에서는 서로 다른 결과가 나왔다.

| 축 | 관측 | 현재 해석 |
|---|---|---|
| `B` | `B=16`과 기본 `B=32`의 paired 차이가 큼 | 활성 후보 |
| `p` | anchor A/B에서 DP plan이 균등 plan과 사실상 동률 | 현재 조건에서는 비활성 |
| `r` | CoRePP의 최적 동시 실행 비 `G=0.97` | 비활성 |
| `g=2` CP island | capacity/communication gate 미달 | 구현 전 기각 후보 |

따라서 큰 search space 자체가 논문의 대상이 아니다. **그 search space의 유효 차원을
calibration으로 줄이는 것**이 대상이다.

---

## 2. 절대 오차와 결정 오차는 다르다

예측 시간을 `T_pred(q)`, 실측 시간을 `T_meas(q)`라 하자. 여기서 `q`는 하나의 plan이다.

절대 오차는 다음과 같다.

```text
MAPE(q) = |T_pred(q) - T_meas(q)| / T_meas(q)
```

이는 latency model의 품질을 나타내지만, plan 선택에 필요한 전부는 아니다. 예를 들어
executor의 공통 step tax가 빠지면 모든 후보를 비슷한 비율로 과소예측할 수 있다. MAPE는
커도 후보 순위는 맞을 수 있다.

AxisCert가 쓰는 것은 기준 plan `q0`에 대한 paired log residual이다.

```text
e_x(q; q0)
 = log[T_meas(q) / T_meas(q0)]
 - log[T_pred(q) / T_pred(q0)]
```

`x`는 바꾼 축이다. 예를 들어 `p`의 residual을 구할 때에는 `N`, `B`, `π`를 고정한다.
이렇게 하면 plan 공통의 multiplicative bias가 상쇄된다.

> 같은 calibration residual을 모든 축에 재사용하면 안 된다. `B`는 chunk 수와 executor
> step 수를 바꾸고, `N`은 worker set을 바꾸며, CP는 wire protocol 자체를 바꾼다.

---

## 3. 활성·비활성·미판정의 정의

후보 `q`의 예측 log speedup을 다음처럼 둔다.

```text
ŝ(q; q0) = log[T_pred(q0) / T_pred(q)]
```

축 `x`에 대해 held-out paired experiment에서 얻은 residual bound를 `δ_x`라 하자.
실제 log speedup은 보수적으로 다음 구간에 있다.

```text
ŝ(q; q0) - δ_x  ≤  s(q; q0)  ≤  ŝ(q; q0) + δ_x
```

사전 정의한 실용 임계값을 `ε`라 하면 AxisCert는 다음을 적용한다.

| 판정 | 조건 | 동작 |
|---|---|---|
| inactive | 모든 후보에 대해 `ŝ + δ_x ≤ log(1+ε)` | 축을 기본값으로 고정 |
| active | 어떤 후보에서 `ŝ - δ_x > log(1+ε)` | 해당 축의 후보를 planner가 평가 |
| indeterminate | 둘 다 아님 | targeted calibration, 또는 보수적 기본값 |

이것은 절대 latency confidence interval로 모든 후보를 감싸는 방식보다 강하다. 공통
absolute bias가 커도 plan 간 차분 residual이 작다면 축을 끌 수 있다.

---

## 4. 구조적 gate — 측정 전에 끌 수 있는 축

어떤 축은 paired residual을 모으기 전에 물리적 상한으로도 걸러진다.

### Balanced-capacity contraction

`N`개의 동일 worker가 이미 균형 잡힌 PP를 실행하고, 모든 stage의 service time이 `c`라고
하자. 두 worker를 하나의 local CP/TP group으로 묶었지만 group service time이 `c`보다
작아지지 않는다면, logical stage 수만 `N`에서 `N−1`로 줄어든다.

```text
T_N       = (M + N - 1)c
T_(N−1)   = (M + N - 2)c

T_N / T_(N−1) ≤ (M + N - 1) / (M + N - 2)
```

따라서 순수 bubble speedup은 `1/(M+N-2)`다. 긴 prefill에서는 `M`이 크므로 이 값은
작다. CP 통신, synchronization, nonideal scaling은 이 상한에서 다시 빼야 한다.

이 보조정리는 다음 상황에는 적용하지 않는다.

- 원래 PP가 불균형인 경우
- CP/TP가 cache, bandwidth, kernel efficiency를 바꿔 service time 자체를 낮추는 경우
- global TP처럼 dependency graph가 달라지는 경우

그러므로 이것은 “CP/TP는 항상 쓸모없다”는 정리가 아니라, **어떤 구조 변경이 prototype
비용을 감당할 만큼 큰 상한을 갖는지 먼저 거르는 gate**다.

---

## 5. Axis별 필요한 증거

| 축 | primary evidence | 주의할 교락 |
|---|---|---|
| `B` | same-session anchor A/B와 `δ_B` | chunk 수, step tax, warm-up |
| `N` | fixed feasible worker configuration | memory capacity와 node 종류 |
| `p` | `N,B,π` 고정 partition A/B와 `δ_p` | 긴 길이 thermal drift |
| `π` | 동일 node set의 순서 A/B | link group, background network traffic |
| `r` | 실제 concurrent core-split profile | DRAM/cache interference |
| `g` | capacity upper bound 후 protocol-aware cost | Q/O/KV wire bytes와 numerical equivalence |

---

## 6. 계획 선택과 평가

AxisCert가 활성이라고 판정한 축만 DerivePP의 cost model과 recurrence로 평가한다.

```text
calibration
  → axis gate
  → active candidate plans의 predicted completion time 비교
  → plan manifest
  → held-out measured best-of-grid와 regret 평가
```

평가에서는 세 수치를 분리한다.

```text
absolute accuracy : MAPE
decision accuracy : regret = T_meas(selected) / min_q T_meas(q) - 1
selection safety  : inactive로 끈 축이 실제로 낸 최고 이득
```

마지막 항목이 중요하다. AxisCert가 축을 끄려면, 그 축을 실제로 바꿨을 때 얻을 수 있는
이득이 임계값보다 작았음을 held-out 조건에서 보여야 한다.

---

## 7. 재현 artifact

paired residual은 원자료 없이는 다시 계산할 수 없다. 따라서 각 run은 다음을 보존한다.

```text
artifacts/<run-id>/
  raw/       calibration TSV, paired timing log, link log
  manifest/  git SHA, dirty patch hash, argv, model/build hash
  hosts/     worker set, affinity, governor, frequency policy
  derived/   profile, selected plan, residual bound, plot
```

원자료가 사라지면 `δ_x`와 gate 판정을 재현할 수 없으므로, planner source만 공개하는 것은
충분하지 않다.

세부 구현과 현재 run 상태는 [16-derivepp.md](../16-derivepp.md)를 따른다.
