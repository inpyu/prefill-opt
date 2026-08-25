# 21. Cache-Triggered Graph-Hoisted Packing — SharedPack 의 일반화

`research/18`(연산 경로 분석), `research/19`(SharedPack 구현·측정)의 후속.
**현재 형태의 SharedPack 으로는 독립 알고리즘 기여를 주장하기 어렵다**는 판단과,
그것을 끌어올리는 설계를 기록한다.

---

## 1. 현재 novelty 수준 — 솔직한 평가

| 주장 | 강도 | 이유 |
|---|---|---|
| packed activation 을 스레드가 공유한다 | **약함** | 공통 입력 packing 과 재사용은 전통적 GEMM 최적화 |
| 스레드별 중복 packing 이 cache cliff 를 만든다는 진단 | 중간 | llama.cpp ARM Q4×Q8 경로에서 정량 규명 |
| pack 을 executor op 로 승격해 op 경계를 barrier 로 사용 | 중간 | 커널 내부가 아니라 graph/runtime 구조 변경 |
| 정확한 Q80×4 표현을 한 번 생성하고 모든 SDOT worker 가 소비 | 중간 | bit-identical 동적 activation packing 이라는 구체성 |
| N=8 wave PP 에서 TTFT 1.426~1.446× | **강한 시스템 증거** | 단일 커널이 아니라 실제 분산 critical path |
| "새로운 packing 알고리즘" | **약함** | packing 형식·수학 연산 자체는 새롭지 않다 |

## 2. 선행 기술 — "한 번 pack 해서 재사용" 은 구별점이 못 된다

| 선행 | 이미 하는 것 |
|---|---|
| FBGEMM | packing 을 GEMM 의 명시적 구성요소로 분리, packed 행렬 재사용, packing 에 부가 연산 결합 |
| Intel MKL packed GEMM API | 동일 입력을 여러 GEMM 호출에서 재사용하기 위한 packed API |
| llama.cpp `repack.cpp` | Q8_0 을 4행 interleave 형식으로 변환하는 코드가 이미 존재 |

- FBGEMM: https://engineering.fb.com/2018/11/07/ml-applications/fbgemm/
- MKL packed API: https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html
- llama.cpp repack: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp

따라서 **"pack once, reuse" 만으로는 선행 기술과 구별되지 않는다.**
차별점을 만들려면 *언제·어디서·누가 소유하고 언제 해제하는가*를 **자동으로 결정**하는
쪽이어야 한다.

## 3. novelty 가 아니라 검증 강점인 것 — 구분해서 쓴다

다음 둘은 기여로 내세우지 않는다.

- bit-identical 정확성
- packed bytes 2,688회 불일치 0

**정확한 커널 최적화에서 정확성 보존은 기본 요구에 가깝다.**
특히 *"다른 엣지 가속은 정확도를 희생하지만 우리는 아니다"* 라는 비교는
그 주장을 뒷받침할 측정을 갖고 있지 않으므로 **쓰지 않는다.**

이들은 §13 의 "측정 방법론" 항목에 넣는다. 다만 §19 §8.7 의
"E2E 해시가 깨진 커널을 통과시켰다" 는 **반증 사례**로서 방법론 기여에 값한다.

---

## 4. 제안 — Cache-Triggered Graph-Hoisted Packing (CTGHP)

핵심은 "항상 공유한다" 가 아니라, **실행 그래프와 cache 조건으로 pack 의
위치·소유권·재사용 범위를 자동 결정**하는 것이다.

```
기존 실행
  각 SDOT worker:  activation repack -> 자기 packed copy -> 자기 output column

제안 실행
  graph-level pack producer:
      activation -> 정확히 한 번 Q80x4 변환
                       |
                       +-- SDOT worker 0
                       +-- SDOT worker 1
                       +-- SDOT worker 2
                       +-- SDOT worker 3
```

### 4.1 단계 1 — consumer equivalence class

동일 activation 을 소비하는 GEMM 을 묶는다.

- Q / K / V projection (K = `qSlice.n` = `dim`)
- Gate / Up projection (K = `w1Slice.n` = `dim`)
- 동일 quantization block, 동일 K, 동일 SDOT layout 을 요구하는 consumer

현재 구현은 이 class 를 **사람이 손으로** 잡았다(`y_pack`, `d_pack`).
자동화하면 O projection·lm_head·MoE expert 까지 같은 규칙으로 덮인다.

### 4.2 단계 2 — 중복 footprint 계산

```
W_pack(B, K) = ceil(B/4) * ceil(K/32) * sizeof(block_q8_0x4)
```

스레드 `T` 개, class 내 consumer `G` 개일 때

```
W_dup    ~= G * T * W_pack        (기존 경로의 논리적 packed footprint)
W_shared ~= W_pack                (공유 경로)
```

### 4.3 단계 3 — cache/동기화 gate

```
SharedPack 선택
  <=>  T_dup-pack + T_cache-conflict  >  T_pack-once + T_barrier + T_shared-read
```

단순한 충분조건:

```
T * W_pack > C_eff      그리고      W_pack <= C_eff
```

**현재 실측이 이 cache transition 과 일치한다** (`research/18` §6c/§6d):

| 조건 | 관측 |
|---|---|
| 1~3 스레드 | duplicate/shared 차이 1.00~1.03× |
| 4 스레드 | **1.68×** |
| shared 경로 | panel 68~238 KiB 에서 327~334 GOPS 로 평탄 |
| duplicate 경로 | panel 증가에 따라 318 → 199 GOPS 로 붕괴 |

즉 gate 는 사후 합리화가 아니라 **이미 측정된 전이점을 설명한다.**
이것이 CTGHP 를 "항상 켜는 패치" 와 구별하는 지점이다.

### 4.4 단계 4 — executor 가 packed buffer 생명주기를 관리

- microbatch generation 별 workspace
- single writer, multiple consumer
- 마지막 consumer 종료 후 해제
- shape·capacity 를 runtime 검증 (이미 구현, `research/19` §8.5)
- 조건 불충족 시 기존 per-thread 경로로 fallback (이미 구현, `NN_NO_PREPACK`)

### 4.5 알고리즘 입출력

```
입력:
  실행 그래프
  activation shape
  quantization / layout 요구사항
  thread 수
  cache 용량과 pack 비용
  consumer 집합

출력:
  pack 을 수행할 graph 위치
  공유할 consumer 집합
  packed buffer layout
  buffer lifetime
  shared / duplicate 실행 선택
```

### 4.6 이 정도까지 가면 쓸 수 있는 기여 문장

> SharedPack 은 packing 을 GEMM 내부의 스레드 로컬 작업으로 취급하지 않고,
> quantized LLM 실행 그래프의 **공유 데이터 생성 연산**으로 승격한다.
> 캐시 상주성과 consumer reuse 를 분석하여 packing 의 위치·소유권·생명주기를
> 결정하고, 정확히 동일한 packed representation 을 여러 SDOT worker 와
> projection 이 공유하게 한다.

완전히 새로운 수학 연산은 아니지만, *"llama.cpp 비효율을 고친 패치"* 보다
훨씬 방어 가능한 **graph/runtime algorithm** 이다.

---

## 5. 1.426× 의 원인 분해 — counterfactual 설계

`research/19` §9.3 대로, 현재는 관측만 있고 설명이 없다.
`research/16` 의 completion recurrence 를 그대로 쓴다.

```
F_{k,j} = max( F_{k-1,j} + D_{k-1}(j),  F_{k,j-1} ) + C_{k,j}
```

네 번의 counterfactual 을 계산한다.

| # | C (stage compute) | D (link) | 묻는 것 |
|---|---|---|---|
| 1 | BASE | BASE | 재현 기준선 |
| 2 | **영향받은 matmul 만** SP 시간 | BASE | 그 projection 들의 실제 개선율이 N=1 평균보다 컸는가 |
| 3 | 모든 stage compute SP | BASE | 링크를 고정해도 1.426× 가 나오는가 |
| 4 | SP | SP | 실측과 일치하는가 (모델 타당성 검사) |

이로써 다음을 구분한다.

- 특정 projection 의 실제 개선율이 N=1 평균(1.205×)보다 컸는가
- 병목 stage 가 바뀌었는가
- stage imbalance 가 줄었는가
- `syncWait` 감소가 **단순히 앞 stage compute 감소로 설명되는가**
- executor 또는 통신 경로까지 달라졌는가

필요한 계측: `--stage-timing 1` 로 stage 별 `C_{k,j}`, 파이프라인 로그의
`fs/fe` 타임스탬프로 `F_{k,j}`, `syncXfer` 로 `D_k(j)`.

분해가 성공하면 모호한 "pipeline amplification" 대신 이렇게 쓸 수 있다.

> SharedPack 은 평균 연산량만 줄인 것이 아니라 **병목 stage 의 service time 을
> 선택적으로 줄여 pipeline critical path 를 단축**했다.

실패하면(즉 3번이 1.426× 를 재현하지 못하면) 원인이 compute 밖에 있다는 뜻이며,
그 경우 TTFT 개선의 일부는 SharedPack 으로 귀속할 수 없다. **그 가능성을 열어둔다.**

---

## 6. 최종 판단과 우선순위

| 수준 | novelty |
|---|---|
| 현재 구현 그대로의 SharedPack | 독립 알고리즘 기여는 **약함** |
| DerivePP 내부의 연산 기여 | **충분히 강함** |
| graph-hoisted packing + cache gate + cross-projection reuse + lifetime | **중간 이상** |
| 위 + pipeline critical-path 모델 + 자동 선택 + 다중 모델·CPU 검증 | **주요 기여 후보** |

**우선순위** (스윕을 중단할 필요는 없다 — 끝내고 순서대로):

1. stage 별 recurrence 로 1.426× 의 원인 분해 (§5)
2. packed buffer 타입·shape 검증 완성 — *완료* (`research/19` §8.5, commit `4468558`)
3. Q/K/V 와 Gate/Up 의 **cross-projection reuse** 여부 확인
   (현재 `y_pack` 은 이미 두 class 가 같은 버퍼를 쓰지만 pack 을 두 번 한다)
4. cache-triggered shared/duplicate 자동 gate 구현 (§4.3)
5. 최종 시스템을 동일 조건의 llama.cpp 와 직접 비교 (곱셈 금지 해제의 유일한 길)
