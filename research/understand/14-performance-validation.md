# 성능 검증의 기반지식 — SharedPack의 20%를 어떻게 확정하는가

이 문서는 [20-measurement-environment.md](../20-measurement-environment.md)를 이해하기 위한
보조자료다. 현재 진행 중인 SharedPack paired A/B를 예로 들어 다음을 설명한다.

- kernel, op, prefill wall time, TTFT가 왜 서로 다른가
- 온도·DVFS·세션 드리프트가 비율을 어떻게 오염시키는가
- anchor가 무엇을 보정하고 무엇은 보정하지 못하는가
- “더 빠르다”와 “더 안정적이다”를 어떻게 따로 검증하는가
- 어떤 결과부터 논문 수치로 고정할 수 있는가

---

## 1. 먼저 측정 층위를 분리한다

한 최적화에는 여러 성능 숫자가 생긴다. 이 숫자는 서로 대체할 수 없다.

```text
microkernel
  ↓
한 op의 시간
  ↓
prefill-only op 누적 시간
  ↓
prefill wall time
  ↓
TTFT
  ↓
분산 PP E2E
```

### 1.1 Microbenchmark

한 함수만 반복한다. SharedPack의 `dup` 대 `shared` 벤치가 여기에 해당한다.

```text
장점: 원인 하나를 통제하기 쉽다.
단점: executor boundary, graph buffer, 통신, 실제 호출 순서가 빠질 수 있다.
```

`shared=333.8 GOPS`는 packed buffer를 timed loop 밖에서 만든 kernel upper bound였다.
따라서 production speedup으로 바로 쓰지 않는다.

### 1.2 Operator time

executor가 한 op를 시작하고 모든 worker가 끝날 때까지 잰 시간이다.

```text
Down의 새 비용 = block_pack_dq + block_matmul_w2
```

pack을 matmul 밖으로 옮겼으므로 `block_matmul_w2`만 비교하면 새 경로를 유리하게 회계한다.
항상 최적화 전후의 **같은 기능 범위**를 합쳐 비교한다.

### 1.3 Prefill-only op sum

prefill 동안 실행된 모든 op 시간을 합한 값이다. 단일 노드에서 op가 순차 실행되면
`prefillMs`와 거의 같아야 한다.

과거 profiler는 decode가 끝난 뒤 한 번 출력해 prefill과 decode가 섞였다. 현재는 prefill
종료 시 snapshot을 남기고 다음 heading으로 따로 출력한다.

```text
=== op profile [PREFILL ONLY] total=... ===
```

이 값은 “어느 연산이 얼마나 줄었는가”를 설명하는 primary metric이다.

> **Op profile이 자동으로 안정적인 것은 아니다.** 진행 중 동일 바이너리 A/B에서 BASE의
> prefill-only op 합도 약 31.4→36.5초로 움직였고 `prefillMs`와 거의 같이 변했다. Op
> profile의 장점은 변동을 없애는 것이 아니라, decode를 분리하고 변동이 `w2` 같은 어느
> 연산에서 생겼는지 보여주는 것이다. 따라서 op 시간에도 paired anchor가 필요하다.

### 1.4 Prefill wall time과 TTFT

```text
prefillMs : prompt token의 forward가 끝날 때까지의 wall-clock
TTFT      : 요청 시작부터 첫 output token을 낼 수 있을 때까지의 전체 시간
```

로딩과 tokenizer 포함 여부, 첫 decode 시작점을 구현이 어떻게 정의하는지 확인해야 한다.
현재 N=1 로그에서는 prefill-only op sum과 `prefillMs`가 거의 같지만, 이 관계를 모든 실행
구조에 일반화하지 않는다.

### 1.5 분산 PP E2E

여러 stage는 동시에 실행된다. 각 worker의 op time을 단순히 더한 값은 wall time과 다르다.

```text
분산 E2E = compute critical path + pipeline fill/drain + 통신 + stall
```

따라서 단일 노드 op 20% 개선이 N=8 TTFT 20% 개선을 자동으로 보장하지 않는다. 연산 기여와
분산 시스템 기여를 따로 측정한 뒤 마지막에 결합한다.

---

## 2. 현재 클러스터의 변동 원인

### 2.1 Noise, drift, outlier, confound

| 용어 | 의미 | 예시 |
|---|---|---|
| noise | 같은 조건의 작은 무작위 흔들림 | microbenchmark ±2% |
| drift | 시간이 흐르며 기준선이 한 방향으로 이동 | 지속 부하로 baseline 증가 |
| outlier | 일부 run만 비정상적으로 큼 | attention 한 회차 폭증 |
| confound | 설정 변경과 다른 원인이 함께 바뀜 | K와 duplicate footprint가 동시에 증가 |

평균과 표준편차만 계산하면 네 현상을 구분하지 못한다. 실행 순서, 온도, 주파수, 실패 기록을
함께 봐야 한다.

### 2.2 온도와 DVFS

DVFS(Dynamic Voltage and Frequency Scaling)는 부하·온도에 따라 CPU 주파수와 전압을
바꾸는 기능이다. 현재 governor는 `ondemand`, 최대 주파수는 2.4 GHz다.

진행 중 A/B에서는 온도가 약 66°C까지 올라가고 관측 주파수가 2.0~2.4 GHz 사이에서
변했다. 같은 계산도 주파수가 낮으면 느려진다.

여기서 두 질문을 구분한다.

```text
시스템 효과:
  SharedPack을 켰을 때 실제 ondemand 환경에서 요청이 얼마나 빨라지는가?

연산 메커니즘:
  같은 주파수에서 중복 pack 제거 자체가 얼마나 빠른가?
```

첫 질문에는 SharedPack이 발열과 frequency residence를 바꾸는 효과도 포함될 수 있다.
두 번째 질문을 확인하려면 performance governor 또는 주파수 matched subset이 필요하다.

로그 끝의 한 번의 MHz snapshot은 run 전체의 frequency residence가 아니다. 가능하면 시작,
종료, 최저 주파수 또는 주기 sampling을 함께 기록한다.

### 2.3 저장장치 문제

모델과 OS가 같은 SD 카드에 있고 read error가 존재한다. `prefillMs`는 weight loading 이후에
시작하므로 SD 읽기 시간이 직접 포함되지는 않지만 다음에는 영향을 줄 수 있다.

- 실행 실패와 세션 오염 확률
- page cache와 memory pressure
- 재부팅 필요성
- 하루에 수행 가능한 안전한 run 수

따라서 10회 측정은 단순히 10개 숫자가 아니라 약 63 GB의 모델 읽기를 유발할 수 있다.
실패한 세션을 무한히 반복하는 것은 통계적으로도, 장치 수명 측면에서도 올바르지 않다.

---

## 3. 왜 동일 바이너리 A/B가 필요한가

서로 다른 commit의 로그를 나누면 최적화 외의 변화가 섞인다.

```text
컴파일러 결과
graph 순서
계측 코드
buffer allocation
환경 변수 기본값
```

현재는 다음 runtime flag를 사용한다.

```text
DLLAMA_SHARED_PACK=0   기존 private packing
DLLAMA_SHARED_PACK=1   SharedPack
```

같은 binary, model, prompt, `B`, thread 수를 유지하고 graph의 packing 경로만 바꾼다.
이렇게 해야 mode 차이를 SharedPack에 귀속할 수 있다.

정확성도 같은 방식으로 비교한다.

```text
BASE logits hash = SP logits hash = 8b8178a50a97
```

현재 진행 중 A/B의 완료된 run은 이 N=1 reference를 계속 재현하고 있다. 하지만 N=8은 기존
stage-boundary 경로 때문에 N=1과 원래 hash가 다르므로, 반드시 N=8 BASE와 N=8 SP를
서로 비교한다.

---

## 4. Anchor 설계

### 4.1 왜 단순 평균 비율이 부족한가

시간이 지나면서 baseline이 31초에서 36초로 느려졌다고 하자. 초반 BASE 평균과 후반 SP
평균을 나누면 mode 효과와 시간 drift가 섞인다.

그래서 SP를 앞뒤 BASE로 감싼다.

```text
BASE_before → SP → BASE_after
```

### 4.2 기하평균 보정

성능 drift를 곱셈적이라고 가정하면 SP 시점의 baseline은 로그 시간에서 보간한다.

```text
log T_anchor = (log T_before + log T_after) / 2
```

원래 단위로 돌아오면 기하평균이다.

```text
T_anchor = sqrt(T_before × T_after)
R_i      = T_anchor / T_SP
```

`R_i > 1`이면 SharedPack이 빠르다.

진행 중 자료의 한 예시는 다음과 같다. 이 계산은 방법 설명용이며 최종 결과가 아니다.

```text
BASE_before = 31,431.17 ms
SP          = 28,143.78 ms
BASE_after  = 36,442.48 ms

R = sqrt(31,431.17 × 36,442.48) / 28,143.78
```

모든 내부 SP에 대해 `log R_i`를 계산하고 그 평균을 exponentiate하면 전체 geometric-mean
speedup을 얻는다.

### 4.3 Anchor가 보정하지 못하는 것

Anchor는 다음 조건에서 가장 잘 작동한다.

- drift가 인접 BASE 사이에서 비교적 매끄럽다.
- mode 전환과 무관한 공통 multiplicative drift다.
- 앞뒤 BASE가 모두 정상 완료됐다.

다음은 자동으로 해결하지 못한다.

- SP 실행 중에만 주파수가 체계적으로 높거나 낮아지는 mode-correlated DVFS
- 갑작스러운 throttle 단계 전환
- crash나 I/O error 이후 세션 상태 변화
- 앞뒤 anchor 사이의 비단조 급변

따라서 “anchor가 열을 모두 흡수한다”고 표현하지 않는다. 온도·주파수 기록과 정순/역순
일관성을 함께 확인한다.

### 4.4 Endpoint와 warm-up

앞뒤 BASE가 없는 SP는 primary paired estimate에 쓰지 않는다. 첫 실행은 page/cache와
thread-pool warm-up 영향을 받을 수 있어 폐기한다.

공유 anchor를 사용하는 인접 `R_i`들은 완전히 독립된 표본이 아니다. 단순 tile-level
bootstrap으로 신뢰구간을 과도하게 좁히지 말고, run block 또는 session을 통계 단위로 둔다.

---

## 5. “빠르다”와 “안정적이다”는 다른 주장이다

### 속도 주장

```text
paired speedup의 geometric mean
정순과 역순에서 같은 방향
사전등록 실용 임계값 통과
```

### 안정성 주장

```text
raw range가 아니라 drift를 제거한 residual 분산
mode별 충분한 반복 수
주파수·온도 matched 분석
다른 session에서도 재현
```

진행 중 부분 결과에서 BASE의 `w2Ms`가 크게 흔들리고 SP가 상대적으로 좁아 보이는 것은
흥미로운 신호다. 그러나 측정이 끝나지 않았고 SP에도 2.0~2.4 GHz 변동이 있으므로 아직
“SharedPack이 안정성을 개선했다”고 확정하지 않는다.

안정성을 검증하려면 각 mode의 로그 시간에서 session drift 또는 frequency effect를 제거한
뒤 median absolute deviation, CV 또는 residual variance를 비교한다. 평균 속도 향상과
분산 감소를 별도 결과로 보고한다.

---

## 6. 현재 A/B에서 읽어야 하는 열

`artifacts/sharedpack_ab/results.tsv`의 열은 다음 뜻이다.

| 열 | 의미 | 사용처 |
|---|---|---|
| `seq` | 세션 내 실행 순서 | drift와 anchor 인접성 |
| `mode` | BASE 또는 SP | 처리 변수 |
| `prefillMs` | prefill wall time | 단일 노드 E2E primary |
| `opPrefillMs` | prefill-only op 합 | 연산 계층 primary |
| `w2Ms` | Down matmul 시간 | 핵심 병목 메커니즘 |
| `packMs` | 세 pack op 합 | 새 비용 회계 |
| `md5` | logits reference | bit-identical gate |
| `mhz` | 관측 주파수 | DVFS 공변량, 전체 residence는 아님 |
| `temp` | 시작→종료 온도 | 열 drift 진단 |

세 가지 비율을 따로 계산한다.

```text
R_wall   : anchor-adjusted prefillMs
R_ops    : anchor-adjusted opPrefillMs
R_down   : BASE w2Ms / (SP w2Ms + 필요한 pack share)
```

`R_down`에서 모든 pack 비용을 Down에 붙이면 보수적이다. 실제로 `pack_yq`, `pack_yq2`,
`pack_dq`는 서로 다른 projection family를 위해 쓰이므로 논문 표에서는 family별로 귀속하고,
전체 op 합에서는 세 개를 모두 포함한다.

---

## 7. 분산 측정에서 추가되는 교락

### 7.1 N과 node placement

이 클러스터는 switch 내부 링크와 switch 간 링크가 약 10배 다르다.

```text
N ≤ 4 : 빠른 switch 군만으로 구성 가능
N = 8 : 느린 inter-switch link를 포함
```

따라서 N=4와 N=8 비교를 순수 stage 수 효과라고 쓰지 않는다. 실제 비교 대상은 node 수와
배치 순서가 결합된 deployment plan `(N, π)`다.

### 7.2 Op sum과 critical path

N=1에서는 op가 순차적이므로 op 합과 wall time이 가까울 수 있다. N=8에서는 여러 stage가
겹쳐 실행되므로 다음 recurrence의 critical path가 TTFT를 정한다.

```text
finish[k,j]
  = max(finish[k-1,j] + transfer[k-1,k],
        finish[k,j-1])
    + compute[k,j]
```

따라서 각 노드의 줄어든 GEMM 시간을 단순 합해 분산 E2E speedup이라고 부르지 않는다.
SharedPack을 모든 stage에 배포한 뒤 동일 `(N,B,π)` BASE/SP로 다시 측정한다.

---

## 8. Artifact가 증거가 되기 위한 조건

최종 run은 다음을 함께 보존한다.

```text
raw logs와 logits dump
results.tsv
git SHA와 dirty 여부
binary/model checksum
실행 argv와 environment flag
host/node 목록
governor, 주파수, 온도
정순/역순과 invalid-session 판정
분석식 또는 script
```

최종 논문 run은 가능한 한 `dirty=false`인 clean commit에서 수행한다. dirty patch를 보존하면
복구는 가능하지만, 어떤 source가 실제 binary에 들어갔는지 독자가 한 단계 더 재구성해야 한다.

실패한 run은 삭제하지 않는다.

```text
valid/
invalid_session/
```

으로 분리하고 실패 이유를 남긴다. 원하는 결과만 남기는 selection bias를 막기 위해서다.

---

## 9. 결과를 주장하는 순서

SharedPack의 증거는 다음 계단을 순서대로 올라간다.

| 단계 | 필요한 증거 | 현재 상태 |
|---|---|---|
| 원인 | dup/shared에서 packing 정책만 변경 | 완료 |
| kernel 상한 | Down 198.7→333.8 GOPS | 완료 |
| production 통합 | pack-inclusive op 개선 | 완료 |
| 정확성 | N=1 B=16/B=32 bit-identical | 완료 |
| 동일 바이너리 연산 효과 | prefill-only paired A/B | 진행 중 |
| 단일 노드 E2E | anchor-adjusted `prefillMs` | 진행 중 |
| 분산 정확성 | N=8 BASE/SP 동일 hash | 미완료 |
| 분산 E2E | 동일 deployment plan paired A/B | 미완료 |
| 일반성 | 다른 길이·모델·CPU/thread 조건 | 미완료 |

현재 논문에 확정해서 쓸 수 있는 가장 강한 문장은 다음 수준이다.

> SharedPack은 테스트한 N=1 구성에서 bit-identical하며, production operator 계층에서
> 사전등록한 성능 기준을 통과했다. 동일 바이너리의 prefill-only 및 wall-time 효과는
> paired anchor 실험으로 재확정 중이다.

anchor가 끝난 뒤에야 “prefill을 X배 개선했다”와 “변동성을 줄였다”의 최종 수치를 쓴다.

---

## 10. 새 실험 전 최소 체크

```text
[ ] 같은 binary에서 A/B가 가능한가
[ ] prefill-only와 decode-inclusive profile을 구분했는가
[ ] warm-up 폐기와 앞뒤 anchor가 있는가
[ ] 온도·주파수·실행 순서를 기록하는가
[ ] 실패 시 session 전체 판정 규칙이 있는가
[ ] logits hash가 같은 구성의 reference와 일치하는가
[ ] 다른 benchmark가 동시에 돌지 않도록 flock을 잡았는가
[ ] run 수 × 6.3 GB의 SD 읽기를 감당할 수 있는가
[ ] 끝난 뒤 raw log와 manifest를 repository artifact로 고정하는가
```

운영 명령과 장치별 주의사항의 정본은
[20-measurement-environment.md](../20-measurement-environment.md)를 따른다.
