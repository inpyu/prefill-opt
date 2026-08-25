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

### 이것이 계획에 주는 함의

**(a) 마이크로벤치 설계 결함이 있었다.** AV 블로킹이 벤치에서 안 잡힌 이유는
거기서 V 를 **연속(512 B stride)** 으로 뒀기 때문이다. 스트라이드를 QK 에만
넣고 AV 에는 안 넣은 설계 결함이다.

> **새 AV 벤치는 production 주소식을 그대로 써야 한다.** 이건 협상 대상이 아니다.

**(b) blocked-V 의 근거가 바로 이 실패에 있다.** 주석은 이렇게 끝난다 —
*"블록화하려면 V 타일을 연속 스크래치로 팩킹해야 하고, 그 복사 비용을 따로
재야 한다."* RoleSplit 의 **append 시점 직접 기록**은 정확히 그 복사 비용을
없애는 설계다. 즉 이 방향은 기각된 것이 아니라 **미완**이다.

**(c) 그러나 softmax 가 지배적일 가능성이 있다.** QK 블로킹 실패의 원인 설명이
"softmax·스크래치 왕복·온라인 누적 갱신이 더 크다" 였다.
**그렇다면 V 레이아웃만 고쳐도 전체가 거의 안 움직인다.**

---

## 5.4 Phase 1 — 내부 분해 (진행 중)

지금 아는 것은 *"attention 이 53%"* 까지다. **그 안을 모른다.**
재보지 않고 V 레이아웃부터 바꾸면 개선하고도 전체가 안 움직일 수 있다.

`DLLAMA_ATT_PHASE=1` 계측을 넣었다(`multiheadAttFused_F32`).

```
qk        QK^T 내적
softmax   running max/sum, exp, 이전 누적분 rescale
av        p × V 누적
finalize  1/l 정규화·store
wall      커널 전체
```

**필수 검증: `|phase 합 − wall| / wall ≤ 5%`.** 넘으면 계측부터 고친다.

### 사전등록 판정 기준

| 결과 | 다음 행동 |
|---|---|
| AV ≥ 35% | blocked-V 진행 |
| QK+AV ≥ 70% | attention 전체 1.5× 가능성 있음 |
| softmax+rescale ≥ 30% | **V 레이아웃만으로 20% E2E 불가.** softmax 도 함께 |
| KV append ≥ 5% | direct append 설계를 더 엄격히 평가 |

측정 조건: `S_real` 447/1789/7212 × `B` 16/32 × threads 1/2/3/4, 최소 3회,
같은 세션 교차.

---

## 5.5 RoleSplit-KV Attention — 설계 (미구현)

K 와 V 는 **소비 방식이 다르다.**

```
QK^T   한 query 와 한 token 의 K 벡터 전체를 내적
       → token 별 연속 K 가 유리                (현재 레이아웃이 이미 맞다)

AV     같은 output feature 에 대해 많은 token 의 V 를 누적
       → feature 별로 token 방향이 연속인 V 가 유리   (현재 레이아웃과 어긋난다)
```

그래서 역할별로 분리한다.

```
K cache   [KV head][token][feature]                        — 유지
V cache   [KV head][feature block][token tile][token][lane] — 변경
```

네 요소를 하나의 알고리즘으로 묶어야 novelty 가 생긴다.

1. K/V 역할별 **비대칭 레이아웃**
2. V projection 결과를 **cache append 시점에 직접** blocked layout 으로 기록
   (매 attention 마다 transpose 하면 §5.3 의 복사 비용이 그대로 돌아온다)
3. **GQA head 사이 V SIMD load 공유** — `kvMul` 개 query head 가 NEON 레지스터에
   올린 V 하나를 공유
4. cache·register 용량으로 microkernel 형태를 **유도** (magic number 금지)

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
| AV microkernel | ≥1.5×, bit-identical | **RoleSplit 중단** |
| GQA 공유 | `kvMul=4` 에서 blocked-only 대비 ≥1.15× | locality 기여로 축소 |
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

검증 계층은 SharedPack 과 같은 이유로 5단이다 ([09](09-measurement.md) §7.2).

```
1  V cache logical value     native[t,i] == blocked accessor[t,i]
2  Tile level                한 V 타일의 raw/logical 대조
3  Attention phase           QK score / running max / running sum / AV partial
4  Layer output              attention sublayer output
5  Serving output            logits, 생성 token
```

**token sequence 동일만으로 통과시키지 않는다.** SharedPack 에서 깨진 커널이
일부 구성의 최종 해시를 통과한 전례가 있다 ([04](04-sharedpack.md) §3.7).
