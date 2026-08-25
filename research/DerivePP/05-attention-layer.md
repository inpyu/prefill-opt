# 05. Attention 계층 — 세 번째 병목과 RoleSplit-KV 계획

> **상태: 개발 착수 단계.** 기여로 확정된 것이 아니다.
> Phase 1(내부 분해)이 끝나기 전에는 어떤 성능도 주장하지 않는다.

---

## 5.1 왜 이 계층이 필요한가 — 병목이 이동했다

`g(S)` = projection 이 prefill 에서 차지하는 비율. **직접 측정**했다
(`research/17` §7.6, 단일 노드, `--stage-timing 1`, warm-up 1 + 본측정 3).

| `S_real` | `g(S)` | projection 무한가속 상한 `1/(1−g)` |
|---|---|---|
| 447 | **0.863** | 7.30× |
| 1789 | **0.680** | 3.12× |
| 7212 | **0.439** | **1.78×** |

`S=7212` 에서 `1 − g = 0.561` 의 내역은

```
attention core / prefill  ≈ 53%
executor·norm·residual 등  ≈  3%
```

**긴 문맥에서는 attention 이 지배 병목이다.** SharedPack 이 건드리는 곳을 무한히
가속해도 TTFT 는 1.78× 에서 막힌다.

### 관측이 이를 뒷받침한다

일반성 스윕에서 SharedPack 이득이 길이에 따라 감소한다.

```
S_real=447   R = 1.4823
S_real=1789  R = 1.3773
```

`g(S)` 감소와 방향이 정확히 일치한다.

> ⚠️ 한때 `g(7212) ≈ 0.32` 로 추정했으나 **틀렸다.** att share 0.68 에서 역산한
> 값인데 그 안에 Q/K/V·O projection 이 포함돼 있었다. 직접 측정값이 **0.439** 다.

---

## 5.2 datapath 격차

같은 CPU 인데 두 계층이 다른 연산기를 쓴다.

| 계층 | 연산 | 실측 처리율 |
|---|---|---|
| projection | int8 SDOT | **327~334 GOPS** (4스레드), ~83 GOPS/core |
| attention | F32 NEON | **16.1 GFLOPS/core** (QK^T), 15.7 (AV) |

**약 5배 차이다.** 가중치는 Q4_0 라 SDOT 로 가는데 KV 캐시는 F32 라
QK^T·AV 가 F32 에 남는다.

격차를 메우려면 datapath 를 바꿔야 하는데 그것은 양자화 변경이고,
[01](01-problem.md) §1.8 의 금지 항목이다. **따라서 이 계층에서 노릴 수 있는 것은
연산기가 아니라 데이터 이동과 재사용이다.**

---

## 5.3 반드시 먼저 읽어야 할 선행 부정 결과 (in-situ)

**커널 주석에 이미 기록돼 있다.** 마이크로벤치에서 성공했는데 실제 커널에서
실패한 사례가 셋이다. 같은 함정을 다시 밟지 않기 위해 여기 옮긴다.

| 시도 | 마이크로벤치 | in-situ (`S=1789`) | 원인 |
|---|---|---|---|
| **QK 4×4 레지스터 블로킹** | 8.9 → 16.1 GFLOPS (**1.81×**), 비트 동일 | attnMs **39,050 vs 38,031** — 오히려 2.7% 손해 | "QK^T 비중보다 softmax·스크래치 왕복·온라인 누적 갱신 같은 **주변 비용**이 커서 FMA 효율 개선이 묻힌다" |
| **AV 블로킹** (i 바깥, t 안쪽 4×4) | 개선 | **39,263 → 67,292 ms (1.71배 악화)** | `kvDim0=1024 float = 4 kB` 스트라이드라 V 타일이 `128 × 4 kB = 512 kB` 로 코어당 L2 를 정확히 넘긴다 |
| **KV 레이아웃 변환** | — | 스트라이드 접근 손해가 **4% 뿐** | 변환 비용이 이득보다 크다 |
| **쿼리 G배 융합** | — | **1.00×** | `t` 바깥 루프가 이미 K 재사용을 달성 |

### 무엇을 기각했고 무엇을 기각하지 않았는가

**이 분리가 핵심이다.** 넓게 읽으면 RoleSplit 전체가 이미 기각된 것처럼 보인다.

| 과거 결과 | 실제로 기각한 것 | **기각하지 않은 것** |
|---|---|---|
| QK 4×4 in-situ −2.7% | 현재 레이아웃에서 **FMA 만** 개선 | QK/softmax **dataflow 재구성** |
| AV blocking 1.71× 악화 | 4 kB stride 를 둔 채 `i→t` **루프 교환** | **append 시점 blocked-V** |
| stride 손해 4% | **K** 를 읽는 QK 경로의 layout 변환 | **V** 의 feature-major AV layout |
| query G 융합 1.00× | `t`-outer QK 의 **cache-level** K 재사용 | AV 의 **register-level** V 공유 |

> 세 번째 행이 특히 중요하다. 그 실험은 사실상 **K-stride** 결과다.
> "KV 레이아웃 변환" 이라고 넓게 쓰면 V layout 까지 기각된 것으로 읽힌다.

### "V 타일 512 KiB" 의 정확한 의미

```
유효 V 데이터  = 128 tokens × 128 floats × 4B      =  64 KiB
주소 범위      = 128 tokens × kvDim0(1024) × 4B    ≈ 512 KiB
```

**V 데이터가 512 KiB 인 것이 아니다.** 다른 KV head 가 사이에 끼어 있어
**64 KiB 를 읽으려고 약 512 KiB 주소 범위를 건드린다.**

blocked-V 는 이 주소 범위를 64 KiB 에 가깝게 **압축하는** 설계다.
이것이 novelty 의 중요한 근거이고, 과거 실패와 RoleSplit 을 가르는 지점이다.

### 함의 셋

**(a) 옛 AV 마이크로벤치에 설계 결함이 있었다.** V 를 **연속(512 B stride)** 으로
뒀다. 스트라이드를 QK 에만 넣고 AV 에는 안 넣은 것이다.

> **새 AV 벤치는 production 주소식을 그대로 써야 한다.** 협상 대상이 아니다.

**(b) blocked-V 의 근거가 바로 이 실패에 있다.** 주석은 이렇게 끝난다 —
*"블록화하려면 V 타일을 연속 스크래치로 팩킹해야 하고, 그 복사 비용을 따로
재야 한다."* RoleSplit 의 **append 시점 직접 기록**이 정확히 그 복사 비용을
없앤다. 이 방향은 기각된 것이 아니라 **미완**이다.

**(c) 그러나 softmax 가 지배적일 가능성이 있다.** QK 블로킹 실패의 원인 설명이
"softmax·스크래치 왕복·온라인 누적 갱신이 더 크다" 였다.
**그렇다면 V 레이아웃만 고쳐도 전체가 거의 안 움직인다.**

## 5.4 Phase 1 — 내부 분해

지금 아는 것은 *"attention 이 53%"* 까지다. **그 안을 모른다.**
재보지 않고 V 레이아웃부터 바꾸면 개선하고도 전체가 안 움직일 수 있다.

### 5.4.1 계측기가 병목의 모양을 바꾸면 안 된다

`DLLAMA_ATT_PHASE=1` (`multiheadAttFused_F32`). 두 가지를 지켰다.

**(1) 핫 경로에 atomic 이 없다.** 스레드별 cache-line 정렬 슬롯에 로컬 누적하고
전역 반영은 함수 종료 시 한 번이다. tile 마다 전역 atomic 을 치면 네 스레드가
같은 라인을 다투어 **coherence traffic 이 측정 대상 자체를 느리게 만든다.**

**(2) `wall` 은 thread 시간의 합이지 벽시계가 아니다.** 셋을 구분해 낸다.

| 값 | 용도 |
|---|---|
| `thread_sum` | phase **비중** 분석 |
| `thread_max` | 멀티코어 **service time** 근사 — executor op wall 과 비교할 값 |
| executor op wall | 최종 검산 기준 |

> thread-sum 을 executor wall 과 그냥 비교하면 스레드 수만큼(약 4배) 어긋난다.

### 5.4.2 회계 누락을 막는다

phase 를 여섯으로 나누고 나머지는 역산한다.

```
setup      scratch/state 준비, output memset, maxPos 계산
qk         QK^T 내적
softmax    running max/sum, exp, 이전 누적분 rescale
av         p × V 누적
finalize   1/l 정규화·store
other      = wall − 위 다섯   ← 제어 루프 등 잔여
```

**`|phase 합 − wall| ≤ 5%` 를 실패 판정으로 쓰면 안 된다.** 실제 작업이
`wall − phase 합` 에 남아 있기 때문이다. 그래서 `other` 를 명시적으로 역산하고
`−1% ≤ other ≤ 15%` 를 회계 정상 범위로 본다.

**KV append 는 이 함수 밖의 별도 op 다.** 여기서 잡히지 않는다.
`DLLAMA_OP_PROFILE` 의 op 별 시간에서 읽는다.

### 5.4.3 계측기 오버헤드 A/B (필수 선행)

같은 조건에서 교차로 잰다.

```
DLLAMA_ATT_PHASE=0 → 1 → 0 → 1
```

| 결과 | 판정 |
|---|---|
| attention wall 변화 ≤ 2% | 계측 사용 가능 |
| > 2% | timer granularity 축소 |

### 5.4.4 pilot 을 먼저 한다

처음부터 `3 길이 × 2 B × 4 스레드 × 3회` 를 돌 필요가 없다.

```
S_real=447   B=32  threads=4     warm-up 1 + 본측정 3
S_real=1789  B=32  threads=4     warm-up 1 + 본측정 3
S_real=7212  B=32  threads=4     warm-up 1 + 본측정 3
```

pilot 으로 판단할 것:

- phase 회계가 맞는가
- softmax 가 실제로 지배적인가
- AV share 가 충분한가
- 길이에 따라 phase 비중이 어떻게 변하는가

그 뒤에 `B`·thread 축을 확장한다.

### 5.4.5 Gate — phase 비율이 아니라 예측 속도로 판정한다

*"AV ≥ 35%"* 같은 기준은 방향성은 맞지만 **attention 1.5× 를 보장하지 않는다.**
측정된 share 와 현실적인 phase 별 가속률을 넣어 직접 계산해야 한다.

```
1 / r_attention = f_qk/r_qk + f_softmax/r_sm + f_av/r_av + f_final/r_fin + f_other
```

예를 들어

```
f_qk = 0.30, r_qk = 1.20
f_sm = 0.30, r_sm = 1.10
f_av = 0.35, r_av = 1.70
f_other = 0.05

r_attention = 1 / (0.30/1.20 + 0.30/1.10 + 0.35/1.70 + 0.05) ≈ 1.28×
```

**AV 가 35% 를 넘어도 attention 전체 1.5× 에는 못 미친다.**

| optimistic `r_attention` | 판정 |
|---|---|
| ≥ 1.5× | RoleSplit 전체 진행 |
| 1.3~1.5× | 20% E2E 는 어렵다. 보조 기여로 가능 |
| < 1.3× | **V layout 중심 방향 중단** |

## 5.5 RoleSplit-KV Attention — 설계 (미구현)

K 와 V 는 **소비 방식이 다르다.**

```
QK^T   한 query 와 한 token 의 K 벡터 전체를 내적
       → token 별 연속 K 가 유리                (현재 레이아웃이 이미 맞다)

AV     같은 output feature 에 대해 많은 token 의 V 를 누적
       → feature 별로 token 방향이 연속인 V 가 유리   (현재 레이아웃과 어긋난다)
```

그래서 역할별로 분리한다.

**현재 K 의 물리 구조는 문서에 적었던 `[KV head][token][feature]` 가 아니다.**
코드를 보면

```cpp
hKc  = keyCache + headIndex * headDim;
posK = hKc + t * kvDim0;
```

이므로 실제로는 **token 이 바깥**이다.

```
K 현재   [token][KV head][feature]      — 유지 (QK^T 내적에 이미 맞다)
V 제안   [KV head][token tile][feature block][token in tile][lane]
```

**`token tile` 을 `feature block` 보다 앞에 둔다.** 그래야 새 token 하나를
append 할 때 현재 tile 의 약 64 KiB 영역 안에서 끝난다. 반대 순서면 feature
block 들이 cache 전체에 넓게 흩어진다.

```
tokenTile    = token / TILE
tokenInTile  = token % TILE
featureBlock = feature / 4
lane         = feature % 4

offset = ((((kvHead * numTokenTiles + tokenTile)
             * numFeatureBlocks + featureBlock)
             * TILE + tokenInTile)
             * 4 + lane)
```

이 레이아웃은 셋을 동시에 만족한다.

- append 가 현재 token tile 안에서 끝난다
- AV 에서 고정 feature block 에 대한 **token 방향 SIMD vector 가 연속**이다
- online-softmax 의 token tile 과 자연스럽게 대응한다

네 요소를 하나의 알고리즘으로 묶어야 novelty 가 생긴다.

1. K/V 역할별 **비대칭 레이아웃**
2. V projection 결과를 **cache append 시점에 직접** blocked layout 으로 기록
   (매 attention 마다 transpose 하면 §5.3 의 복사 비용이 그대로 돌아온다)
3. **GQA head 사이 V SIMD load 공유** — `kvMul` 개 query head 가 NEON 레지스터에
   올린 V 하나를 공유
4. cache·register 용량으로 microkernel 형태를 **유도** (magic number 금지)

### 5.5.1 GQA 공유의 차별점을 엄밀히 정의한다

**현재 커널도 이미** GQA head 를 그룹으로 처리하고, `t` 를 바깥 루프로 두며,
V 를 cache 에서 재사용한다. 따라서

> ~~GQA head 가 V 를 공유한다~~

만으로는 기존 코드와 **구분되지 않는다.** 정확한 차이는 이것이다.

> **하나의 V SIMD vector load 를 여러 query-output accumulator 가
> register 수준에서 동시에 소비한다.**

기존은 cache 수준 재사용(같은 라인을 다시 읽음)이고, 새 방식은 register 수준
재사용(한 번 올린 값을 여러 누적기가 씀)이다. **query G배 융합이 1.00× 였던 것은
cache 수준 재사용이 이미 달성돼 있었기 때문**이며, register 수준은 다른 문제다.

증명에 필요한 것:

```
retired load instruction 감소
L1D access / refill 감소
같은 V tile 에 대한 load 횟수 정적 분석
kvMul=1 에서 register-sharing 이득 소멸
kvMul=2/4/8 에 따른 이득 변화
```

### 5.5.2 microkernel ablation 은 4단계다

locality 이득과 GQA 공유 이득을 **분리해야** 한다.

| 단계 | 레이아웃 | register 공유 | 무엇을 잰다 |
|---|---|---|---|
| A | 기존 V | 없음 | 기준선 |
| B | blocked V | 없음 | **레이아웃만의 locality 이득** |
| C | blocked V | query row 공유 | row 방향 register 재사용 |
| D | blocked V | **GQA head 공유** | 최종 후보 |

`D − B` 가 register 공유의 순수 기여다. `B` 만으로 대부분이 설명되면
novelty 는 레이아웃 쪽이고, 그건 선행 기술에 더 가깝다.

### 반증 가능한 사전 예측

> RoleSplit 의 이득은 prompt 가 길고 `kvMul` 이 클수록 증가하며,
> **`kvMul=1`(MHA)에서는 blocked-V 의 locality 이득만 남아야 한다.**

이 예측을 **측정 전에** 기록한다. 그래야 결과가 fitting 으로 보이지 않는다.

---

## 5.6 상한 계산 — 20% E2E 가 필요로 하는 것

`S=7212` 에서 attention 비중 0.53 을 쓰면

```
attention 전체 1.5× 가속  →  E2E  1/((1−0.53) + 0.53/1.5)  ≈  1.215×
```

**20% E2E 를 위해서는 attention 전체가 약 1.5× 빨라져야 한다.**
AV 만 1.7× 로 올린다면 AV 가 attention 시간의 약 76% 를 차지해야 도달한다.
현실적으로 그만큼 높지 않을 수 있으므로 **QK·softmax 도 함께 다뤄야 한다.**

이것이 Phase 1 분해가 선행되어야 하는 이유다.

---

## 5.7 중단 기준 (사전등록)

| 단계 | 통과 | 실패 시 |
|---|---|---|
| 내부 분해 | phase 합이 wall 의 ±5% | 계측 수정 |
| 계측기 오버헤드 | on/off 차이 ≤2% | granularity 축소 |
| **Phase 1 gate** | optimistic `r_attention` ≥1.5× | 1.3~1.5 보조 기여 / <1.3 **중단** |
| AV microkernel | ≥1.5×, bit-identical | **RoleSplit 중단** |
| GQA 공유 (`D` 대 `B`) | `kvMul=4` 에서 ≥1.15× | locality 기여로 축소 |
| V direct append | append <1%, 메모리 증가 ≤5% | layout 재설계 |
| attention 전체 | `S=7212` 에서 ≥1.5× | **20% E2E 주장 철회** |
| 긴 입력 E2E | `Wave+SP` 대비 ≥1.20× | 보조 기여로 축소 |
| decode | 악화 ≤3% | decode accessor 최적화 |
| 일반성 | 최소 두 모델에서 같은 방향 | 단일 모델 최적화로 한정 |

**RoleSplit 이 실패해도 Wave+SharedPack 논문은 그대로 유지된다.**
성공했을 때만 세 번째 핵심 기여로 올린다.

---

## 5.8 정확성 기준 — 비교 대상을 틀리지 않는다

현재 online fused kernel 은 과거 non-fused attention 과 **bit-identical 하지 않다**
(온라인 소프트맥스는 연산 순서가 다르다). 따라서

> **비교 기준은 반드시 현재 online fused kernel 이다.**

검증 계층은 SharedPack 과 같은 이유로 5단이다 ([09](09-measurement.md) §9.2).

```
1  V cache logical value     native[t,i] == blocked accessor[t,i]
2  Tile level                한 V 타일의 raw/logical 대조
3  Attention phase           QK score / running max / running sum / AV partial
4  Layer output              attention sublayer output
5  Serving output            logits, 생성 token
```

**token sequence 동일만으로 통과시키지 않는다.** SharedPack 에서 깨진 커널이
일부 구성의 최종 해시를 통과한 전례가 있다 ([04](04-sharedpack.md) §3.7).
