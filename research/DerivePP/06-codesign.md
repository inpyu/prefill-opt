# 06. 계층 Co-design — 2×2 Ablation 과 Recurrence 검증

**이 문서가 논문의 핵심 표를 담는다.** 두 기법을 나열하는 것과, 두 계층이
서로를 강화한다고 보이는 것은 다르다. 그 차이를 만드는 것이 여기 있다.

---

## 6.1 핵심 표 — 2×2 ablation

| 구성 | Wave Pipeline | SharedPack | 의미 |
|---|---|---|---|
| 기존 시스템 | ✗ | ✗ | 기준 |
| Wave only | ✓ | ✗ | **클러스터 실행 기여** |
| SharedPack only | ✗ | ✓ | **로컬 연산 기여** |
| Wave + SharedPack | ✓ | ✓ | **최종 공동설계** |

측정 항목:

```
TTFT
projection time (영향받은 6개 matmul)
stage 별 service time  C_{k,j}
syncWait
네트워크 전송 시간      D_k(j)
최종 logits md5 + packed-byte 대조
```

**이 표가 있어야** SharedPack 의 1.436× 가 단순한 커널 패치인지, 아니면
Wave Pipeline 과 결합했을 때 **critical path 를 추가로 줄이는지** 구분할 수 있다.

### 예비 관측 (앵커 아님, `research/19` §8)

| wave | SP | prefillMs | logits |
|---|---|---|---|
| ✓ | ✗ | 7,742.70 | `c52e50e37e65` |
| ✓ | ✓ | 5,113.64 | `c52e50e37e65` |
| ✗ | ✗ | 34,459.12 | `8b8178a50a97` |
| ✗ | ✓ | 27,000.08 | `8b8178a50a97` |

```
wave off 에서 SP 이득  ≈ 1.276×
wave on  에서 SP 이득  ≈ 1.514×   (이 행은 앵커 값 1.436× 로 대체해야 한다)
```

**wave 가 켜져 있을 때 SP 이득이 더 크다** 는 방향성이 보인다. 이것이 co-design
주장의 씨앗이다. 다만 위 표는 각 1회이고 검증기가 켜진 상태라 **앵커 재측정이
필요하다.** → [08-status.md](10-status.md)

> **주의.** wave on 과 wave off 는 로짓 해시가 다르다. wave 경로가 활성값을
> q80 로 전송하기 때문이며 SharedPack 이전부터 있던 성질이다.
> 정확성 비교는 **같은 wave 조건 안에서만** 한다.

---

## 6.2 syncWait 를 이득으로 더하면 안 된다

관측된 성분 변화는 이렇다 (N=8, S=447, 앵커).

| 성분 | BASE | SP | 관측 비율 |
|---|---|---|---|
| prefill 전체 | 7,612 | 5,338 | **1.426×** |
| syncWait | 3,725 | 2,105 | 1.770× |
| non-wait residual | 3,887 | 3,233 | 1.202× |

**`syncWait` 는 독립적인 비용이 아니다.** stage 완료시각이 바뀐 **결과**다.
앞 스테이지의 `C_{k,j}` 가 줄면 `F_{k−1,j}` 가 당겨지고, 그 결과로 뒤 스테이지의
대기가 줄어든다. 따라서

```
"연산 1.202× 이득 + syncWait 1.770× 이득"     ← 이중 계산
```

그리고 다음 문장도 **쓰지 않는다.**

> ~~스테이지가 빨라져 wave overlap 이 좋아지고 bubble 이 줄었다~~

스케줄과 마이크로배치 수가 그대로면 bubble 의 **슬롯 수나 비율이 자동으로 준 것이
아니다.** 줄어든 것은 각 슬롯의 길이일 수 있고, 그것은 별개의 주장이다.

---

## 6.3 Recurrence 로 메커니즘을 설명한다

관측 TTFT 가 완료시각 재귀로 설명되는지 검증한다.

```
F_{k,j} = max( F_{k−1,j} + D_{k−1,k,j},  F_{k,j−1} ) + C_{k,j}
```

- `C_{k,j}` : stage `k` 가 microbatch `j` 를 계산하는 시간
- `D_{k−1,k,j}` : 앞 stage 에서 activation 을 전달하는 시간
- `F_{k,j}` : 해당 stage·microbatch 의 완료시각

**BASE/SP 의 `C_{k,j}` 를 넣어 실측 7,612 → 5,338 ms 를 재현하면**,
"pipeline amplification" 을 관찰이 아니라 **메커니즘으로** 설명할 수 있다.

### Counterfactual 네 번

| # | `C` (stage compute) | `D` (link) | 묻는 것 |
|---|---|---|---|
| 1 | BASE | BASE | 재현 기준선 — 모델이 실측을 맞히는가 |
| 2 | **영향받은 matmul 만** SP | BASE | 그 projection 들의 개선율이 N=1 평균(1.205×)보다 컸는가 |
| 3 | 모든 stage compute SP | BASE | **링크를 고정해도 1.426× 가 나오는가** |
| 4 | SP | SP | 실측과 일치하는가 (모델 타당성) |

이로써 다음을 구분한다.

- 특정 projection 의 실제 개선율이 N=1 평균보다 컸는가
- **병목 stage 가 바뀌었는가**
- stage imbalance 가 줄었는가
- `syncWait` 감소가 단순히 앞 stage compute 감소로 설명되는가
- executor 또는 통신 경로까지 달라졌는가

### 성공하면 쓸 수 있는 문장

> SharedPack 은 평균 연산량만 줄인 것이 아니라 **병목 stage 의 service time 을
> 선택적으로 줄여 pipeline critical path 를 단축**했다.

### 실패하면

3번이 1.426× 를 재현하지 못하면 원인이 compute 밖에 있다는 뜻이다.
그 경우 **TTFT 개선의 일부는 SharedPack 으로 귀속할 수 없다.**
그 가능성을 열어둔다.

### 필요한 계측

```
C_{k,j}   --stage-timing 1  (op 별 시간을 stage 로 집계)
F_{k,j}   파이프라인 로그의 fs/fe 타임스탬프
D_k(j)    syncXfer
```

---

## 6.4 이미 확보된 메커니즘 근거 — 스레드 수 실험

co-design 주장과 별개로, **SharedPack 의 메커니즘 자체는 반증 가능한 예측이
맞아떨어져 확인됐다.**

cache gate 는 이렇게 예측한다.

```
중복 packing 이 없으면 (스레드 1개) SharedPack 의 이득은 0 이어야 한다
```

측정 결과 (N=8, S=447, B=16):

| 스레드 | BASE | SP | R |
|---|---|---|---|
| 4 | 6,523.5 ms | 4,400.9 ms | **1.4823** |
| **1** | 13,588.3 ms | 13,619.5 ms | **0.9977** (−0.23%, 잡음 수준) |

**1스레드에서는 변화가 −0.23% 로 측정 잡음 수준이다.** 제거할 중복이 없기 때문이다.

단일 커널에서 관측한 전이(`1~3 스레드 1.00~1.03×`, `4 스레드 1.68×`)의
**두 끝점인 1스레드와 4스레드가 N=8 분산 E2E 에서도 재현됐다.**
(N=8 에서 2·3 스레드는 아직 재지 않았다 — 일반성 스윕 2차 항목이다.)

이것은 "그냥 빨라졌다" 가 아니라 **원인이 중복 packing 임을 보이는 증거**다.
