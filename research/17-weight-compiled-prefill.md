# WCEP — 실제 Q4 가중치를 CPU 산술 회로로 컴파일하는 Exact Prefill

**상태: 새 연구 가설 / 구현 전 feasibility 단계**  ·  작성 2026-08-21  
관련: [16-derivepp.md](16-derivepp.md), [02-baseline-dotprod-fix.md](02-baseline-dotprod-fix.md),
[12-findings.md](12-findings.md)

> 이 문서는 DerivePP의 스케줄 설정을 하나 더 고르는 계획이 아니다. 현재 Q4_0×Q8_0
> GEMM이 모든 weight tile을 같은 방식으로 계산한다는 전제를 버리고, **모델에 실제로
> 저장된 Q4 값으로부터 그 tile을 정확하게 계산하는 ARM 명령 DAG를 오프라인 합성**하는
> 새 연산 알고리즘을 검토한다.

---

## 0. 먼저 읽는 결론

### 0.1 목표

다음 세 조건은 협상하지 않는다.

1. **정확도 손실 없음.** 현재 Q4_0 weight와 Q8_0 activation, block scale, INT32 dot을
   그대로 계산한다. 최종 목표는 결정적 baseline 대비 bit-identical output이다.
2. **CPU 연산 알고리즘.** 요청·노드·microbatch를 고르는 스케줄러가 아니라 Q4×Q8
   내적의 실행 방법을 바꾼다.
3. **현재 강한 baseline 대비 end-to-end TTFT 1.20× 이상.** 현재 baseline은 이미
   Q4_0x4 repack과 ARM SDOT으로 약 330 GFLOPS를 내므로 llama.cpp의 옛 dequant/GEMV
   경로를 비교 대상으로 삼지 않는다.

### 0.2 제안

가칭 **WCEP(Weight-Compiled Exact Prefill)** 는 Q4 weight tile마다 다음 문제를 푼다.

\[
P^*(W_t)=
\arg\min_{P\in\mathcal P(\mathcal I)}
\operatorname{Cost}_{\mathrm{CPU}}(P)
\]

단,

\[
\forall x\in\mathbb Z_8^{32},\qquad
P(x)=x^\top W_t
\]

- \(W_t\): 고정된 Q4_0 weight tile
- \(x\): 런타임 Q8_0 activation block
- \(\mathcal I\): Pi 5가 제공하는 `SDOT`, `TBL`, widening add/sub, shift 등의
  exact integer primitive
- \(P\): 실제 weight 값에 맞게 합성된 산술 DAG

기존 커널은 모든 tile에 같은 프로그램을 적용한다. WCEP는 **실제 weight 값에 따라
SDOT, LUT, shift-add, output 간 공통 부분합을 조합**한다.

### 0.3 20%를 넘는 경로

`S=448` 웜 실측에서 `ffnMs + attnProjMs`는 전체의 약 86%다. 모든 Q4 projection에
동일한 kernel speedup \(r\)이 적용된다고 단순화하면:

\[
\operatorname{Speedup}_{E2E}(r)
=\frac{1}{(1-g)+g/r},\qquad g\approx0.86
\]

| Q4×Q8 kernel speedup `r` | E2E 예상 | 연구 판단 |
|---:|---:|---|
| 1.20× | 1.16× | 논문 목표 미달 |
| **1.25×** | **1.20×** | 최소 통과선 |
| 1.30× | 1.24× | 현실적 목표 |
| 1.50× | 1.40× | 강한 결과 |
| 2.00× | 1.75× | stretch, 현재 근거 없음 |

따라서 개발 목표를 다음처럼 둔다.

```text
최소 목표      weighted Q4 GEMM 1.25×  → TTFT ≥1.20×
주 목표        weighted Q4 GEMM 1.50×  → TTFT 약 1.40×
stretch 목표   weighted Q4 GEMM 2.00×  → TTFT 약 1.75×
```

이 표는 가능 영역이지 예측 결과가 아니다. 실제 weight에 공유 가능한 구조가 없으면
WCEP는 구현 전에 기각한다.

### 0.4 현재 novelty 판단

다음은 **새롭다고 주장하지 않는다.**

- LUT 기반 low-bit GEMM — T-MAC이 이미 수행
- Q/K/V 또는 Gate/Up weight 연결 — 일반 fused projection
- constant matrix common-subexpression elimination — 오래된 MCM/CSE 분야
- shape별 JIT kernel — FBGEMM 등에 존재

novelty 후보는 이 네 요소의 단순 병렬 나열이 아니라 다음 결합이다.

> 실제 Q4_0×Q8_0 tile을 대상으로, block quantization의 exact scale 경계를 보존하면서
> TBL과 SDOT을 함께 쓰는 ISA-aware 산술 DAG를 합성하고, Transformer FFN의 정확한
> neuron permutation symmetry로 같은 DAG template를 연속 배치하여 prefill panel에서
> 실행한다.

이 문장은 아직 **가설**이다. 관련 연구에서 같은 결합이 있었는지 정식 선행연구 조사가
필요하며, weight audit와 kernel 결과가 없으면 논문 기여로 승격하지 않는다.

---

## 1. 왜 새 계산법이 필요한가

### 1.1 현재 병목

Llama-3 8B의 레이어당 토큰 FLOPs는 대략 다음과 같다.

| 연산 | FLOPs 비중 |
|---|---:|
| K/V projection | 3.9% |
| Q/O projection | 15.4% |
| SwiGLU FFN의 세 projection | **80.7%** |

현재 웜 실측(`S=448`, 단일 노드)은 다음과 같다.

| 성분 | 시간 |
|---|---:|
| prefill 전체 | 약 30.0 s |
| FFN | 약 21.1 s |
| attention projection | 약 4.7 s |
| attention core | 약 2.4 s |
| 기타 | 약 1.1 s |

즉 스케줄이나 activation buffer만 고쳐서는 20%가 나오기 어렵다. 전체의 대부분을 차지하는
Q4×Q8 projection 자체를 더 빠르게 해야 한다.

### 1.2 현재 baseline은 약하지 않다

기존 distributed-llama 커널은 매 GEMM마다 Q4 nibble을 풀어 약 105 GFLOPS에 머물렀다.
현재 코드는 Q4_0 block 네 개를 `block_q4_0x4`로 미리 repack하고 ARM SDOT을 사용해
마이크로벤치 약 330 GFLOPS를 낸다.

따라서 다음 비교는 금지한다.

```text
WCEP vs repack 없는 옛 distributed-llama
WCEP vs scalar/GEMV fallback
WCEP vs 오래된 llama.cpp
WCEP vs llamafile_sgemm 경로        ← production 이 아니다 (§7.5.1, 2.6배 느림)
```

필수 비교 대상은 다음이다.

```text
1. 현재 nn-repack Q4_0x4×Q8_0 SDOT
2. 같은 compiler/threads의 최신 llama.cpp ARM Q4 kernel
3. upstream T-MAC exact 설정
4. WCEP pure-template ablation
5. WCEP full(weight synthesis + permutation clustering)
```

---

## 2. 무엇이 기존 LUT 공유와 다른가

### 2.1 PF-LUT가 핵심 novelty가 될 수 없는 이유

Q/K/V는 하나의 큰 행렬로 연결할 수 있다.

\[
[Q\;K\;V]=H[W_Q\;W_K\;W_V]
\]

Gate/Up도 같다.

\[
[G\;U]=H_2[W_G\;W_U]
\]

따라서 activation LUT를 projection 사이에 공유하는 것은 기존 LUT-GEMM의 output-column
재사용으로 환원된다. 구현 최적화로는 유효하지만 새 핵심 수식은 아니다.

### 2.2 T-MAC과의 경계

T-MAC은 n-bit weight를 고정된 bit-plane으로 분해한다.

\[
W=\sum_{b=0}^{n-1}2^bW_b
\]

WCEP는 bit-width만 보지 않는다. 실제 tile을 여러 exact basis의 합으로 표현한다.

\[
W_t=\sum_{j=1}^{J_t}c_jB_j
\]

- `SDOT basis`: 연속적인 dense 4-byte dot
- `TBL basis`: 반복되는 subset/sign pattern
- `shift basis`: 2·, 4·, 8· partial sum
- `shared basis`: 여러 output column이 함께 쓰는 실제 부분식

목표는 다음이다.

\[
\min_{\{c_j,B_j\}}
\sum_j\operatorname{ISA\_cost}(B_j)
\quad\text{s.t.}\quad
W_t=\sum_jc_jB_j
\]

고정된 LUT 알고리즘이 아니라 **weight-dependent exact program synthesis**라는 점이
핵심 차이다.

### 2.3 일반 constant-matrix CSE와의 경계

상수행렬의 공통 부분식 제거 자체는 새롭지 않다. WCEP가 기여가 되려면 일반적인
add/sub CSE를 실행한 것 이상이어야 한다.

필요한 차별점은 다음과 같다.

1. Q4_0×Q8_0의 32원소 block scale 경계를 보존한다.
2. scalar adder count가 아니라 ARM A76의 실제 SIMD 명령·port·register 비용을 최소화한다.
3. `TBL`과 `SDOT`을 같은 exact DAG 안에서 공동 사용한다.
4. 합성 결과를 FFN permutation symmetry로 재배치해 instruction-cache 문제까지 푼다.
5. `B=16/32` prefill panel에서 여러 activation row를 동시에 처리한다.

이 중 1~3이 빠지면 기존 CSE 또는 T-MAC 변형으로 보일 가능성이 높다.

---

## 3. 정확한 연산 정의

### 3.1 Q4_0×Q8_0 block

한 block에서:

\[
w_i=d_wq_i^w,\quad q_i^w\in[-8,7]
\]

\[
x_i=d_xq_i^x,\quad q_i^x\in[-127,127]
\]

그러므로 output에 더해지는 값은:

\[
\Delta y=d_wd_x
\left(\sum_{i=0}^{31}q_i^wq_i^x\right)
\]

WCEP가 바꾸는 것은 괄호 안 INT32 dot의 실행 방법뿐이다.

### 3.2 overflow 안전성

\[
\left|\sum_{i=0}^{31}q_i^wq_i^x\right|
\le32\cdot8\cdot127=32{,}512
\]

정상적인 INT32 범위를 크게 밑돈다. 중간 DAG node도 symbolic range analysis를 통과해야
하며, 초과 가능성이 있으면 widening primitive를 강제한다.

### 3.3 bit-identical 조건

아래를 모두 지켜야 한다.

1. block별 INT32 dot이 동일하다.
2. weight와 activation scale의 F16→F32 변환이 동일하다.
3. K-block 방문 순서가 동일하다.
4. scale product와 FMA 순서가 동일하다.
5. output column의 thread 소유권을 baseline과 동일하게 둔다.
6. compiler reassociation을 허용하지 않는다.

새 DAG의 integer 연산 순서는 달라도 결과 정수는 정확히 같을 수 있다. 그러나 block 사이
F32 누적 순서를 바꾸면 bit-identical이 깨지므로 이 경계는 변경하지 않는다.

---

## 4. WCEP 컴파일 알고리즘

### 4.1 입력과 출력

```text
입력
  실제 Q4_0 weight tile W_t
  CPU primitive set I
  primitive별 실측 cycle/throughput
  register 수와 load/store 비용
  baseline 누적 순서

출력
  exact arithmetic DAG P_t
  template ID
  packed operand/index descriptor
  symbolic equality certificate
  register/range certificate
```

### 4.2 단계 A — 후보식 생성

각 32×4 tile에서 다음 후보를 만든다.

1. 현재 `Q4_0x4 SDOT` 그대로 계산하는 후보
2. signed bit-plane LUT 후보
3. 2-bit 또는 sign/magnitude basis 후보
4. output column 사이에서 반복되는 2항·4항 부분합 후보
5. 기존 후보의 shift/add/sub 조합

baseline 후보를 항상 포함하므로 합성기가 느린 표현만 찾았을 때 안전하게 기존 커널로
돌아갈 수 있다. 이 fallback은 런타임 탐색이 아니라 model-load 시 정적 결정이다.

### 4.3 단계 B — symbolic canonicalization

각 node는 길이 32의 정수 coefficient vector로 표현한다.

```text
x0             → [1,0,0,...]
x1             → [0,1,0,...]
2*x0 - x1      → [2,-1,0,...]
```

같은 coefficient vector는 동일한 부분식이므로 하나로 합친다. 이 표현은 equality
certificate이자 CSE key다.

### 4.4 단계 C — 비용 제한 beam/e-graph 탐색

전역 최소 프로그램 합성은 조합 폭발한다. 첫 구현은 다음 제한을 둔다.

- tile: 32 K-elements × 4 output columns
- DAG depth 상한
- live vector register 상한 28개
- 후보 coefficient 절댓값 상한
- beam width `K_search`
- baseline 비용보다 싼 partial state만 유지

비용 함수:

\[
C(P)=
\sum_u n_u c_u
+\lambda_{spill}N_{spill}
+\lambda_{load}N_{load}
+\lambda_{dep}D_{critical}
\]

`c_u`는 문서나 peak 값이 아니라 Pi 5에서 dependency-chain/independent-chain
microbenchmark로 잰다.

### 4.5 단계 D — exact verifier

최종 DAG의 네 output coefficient vector가 weight tile 네 column과 같은지 검사한다.

\[
V(P_t)=W_t
\]

추가 검증:

- 각 node의 정수 범위
- descriptor index 범위
- random Q8 block 10만 개와 scalar reference 대조
- adversarial `{-127,0,127}` 패턴

symbolic equality가 1차 증명이고 random test는 구현 버그를 찾는 2차 gate다.

---

## 5. 20%를 넘어가기 위한 세 단계

### 5.1 Level 1 — intra-tile exact synthesis

한 32×4 tile 안에서 다음을 줄인다.

- Q4 nibble expand
- 독립적으로 반복되는 dot/lookup
- horizontal reduction
- 동일 partial sum 재계산

이 단계만으로 weighted GEMM 1.25×가 나오면 최소 목표를 달성한다.

### 5.2 Level 2 — TBL–SDOT cooperative DAG

pure LUT와 pure SDOT 중 하나를 고르는 것만으로는 novelty가 약하다. 한 tile의 서로 다른
exact basis를 두 primitive가 나눠 계산하도록 한다.

```text
dense/irregular 부분       → SDOT
반복 sign/subset pattern   → TBL
2·/4·/8· 계수 조합        → SHIFT + ADD/SUB
공유되는 결과              → 여러 output accumulator에 fan-out
```

필수 측정은 `TBL`과 `SDOT`이 A76에서 실제로 병렬 발행되거나 dependency gap을 숨길 수
있는지다. 두 명령이 같은 실행 자원을 완전히 경쟁한다면 이 level의 상한은 작아진다.

판정:

```text
혼합 DAG가 pure-SDOT/pure-LUT 최선보다 10% 이상 빠름 → 독립 기여 후보
차이 <5%                                      → 혼합 주장은 철회
```

### 5.3 Level 3 — FFN symmetry-aware template clustering

FFN neuron 순서는 수학적으로 모델 함수를 바꾸지 않고 바꿀 수 있다. permutation
matrix를 \(P\)라 하면:

\[
W'_g=PW_g,\qquad W'_u=PW_u,\qquad W'_d=W_dP^\top
\]

원소별 SwiGLU는 permutation과 교환된다.

\[
\operatorname{SiLU}(Pg)\odot(Pu)
=P(\operatorname{SiLU}(g)\odot u)
\]

따라서:

\[
W'_d[
\operatorname{SiLU}(W'_gx)\odot W'_ux]
=W_d[
\operatorname{SiLU}(W_gx)\odot W_ux]
\]

다만 현재 Q4_0 포맷에서는 Down weight의 입력 32개가 scale 하나를 공유한다. 서로 다른
scale block에서 neuron 하나씩 꺼내 새 block을 만들면 원래 값을 같은 Q4_0 block으로
표현할 수 없으므로, **수학적으로 가능한 임의 \(P\)를 그대로 사용해서는 안 된다.**

재양자화 없이 허용되는 permutation 집합을 다음처럼 제한한다.

\[
\mathcal P_{Q4}=
\{\text{32-neuron block permutation}\}
\ltimes
\{\text{각 block 내부 permutation}\}
\]

- 32-neuron block 전체를 옮길 때 Down weight code와 scale을 함께 옮긴다.
- 같은 block 안의 neuron 순서는 공통 scale이므로 자유롭게 바꿀 수 있다.
- Gate/Up의 대응 output row에도 정확히 같은 permutation을 적용한다.

이 제한된 자유도를 사용해 같은 WCEP template를 쓰는 neuron을 가능한 범위에서 연속
배치한다.

```text
변경 전  SDOT, LUT, MIX, SDOT, MIX, LUT, ...
변경 후  SDOT, SDOT, ... | LUT, LUT, ... | MIX, MIX, ...
```

목표:

- indirect branch 제거
- instruction-cache locality 향상
- descriptor load 연속화
- vector lane/template 이용률 향상

주의: permutation이 실제 Q4 값을 바꾸면 안 된다. 기존 quantized row와 scale을 그대로
이동하고, Gate/Up의 동일 neuron과 Down의 대응 input column을 함께 옮긴다. 기본 구현은
위 \(\mathcal P_{Q4}\)만 허용한다. 더 일반적인 permutation을 위해 per-element scale 같은
새 포맷을 도입하는 것은 연산량과 모델 표현을 함께 바꾸므로 본 연구 범위에서 제외한다.

### 5.4 supporting optimization — fused projection과 activation materialization

Q/K/V, Gate/Up 연결과 SwiGLU intermediate 축소는 WCEP와 함께 구현할 수 있다. 하지만
이는 다음 이유로 핵심 novelty에서 제외한다.

- fused QKV/Gate-Up은 일반적이다.
- activation traffic은 Q4 weight traffic보다 작다.
- 이전 분석상 이것만으로 20%를 설명하기 어렵다.

최종 성능에는 포함하되 ablation에서 별도 항으로 보고한다.

---

## 6. template와 코드 크기

### 6.1 tile마다 machine code를 만들면 실패한다

Llama-3 8B에는 매우 많은 Q4 tile이 있다. tile마다 고유 코드를 생성하면 model weight보다
코드가 커지고 instruction cache가 무너진다.

따라서 합성 결과를 canonical template와 descriptor로 나눈다.

```text
P_t = Template[type_t](descriptor_t, activation)
```

- `Template`: 명령 topology
- `descriptor`: weight index, permutation, sign/shift 정보
- `scale`: 기존 Q4_0 scale

### 6.2 template coverage

다음 곡선을 먼저 측정한다.

\[
\operatorname{Coverage}(K)
=\frac{\text{상위 K개 template가 처리하는 tile 수}}
{\text{전체 tile 수}}
\]

사전 gate:

```text
K ≤ 64  에서 coverage ≥ 90%   → 진행
K ≤ 128 에서 coverage ≥ 90%   → 조건부 진행
그 외                         → code locality 위험, 기각 우선
```

### 6.3 static grouping

런타임에 tile마다 template를 분기하지 않는다.

1. model load 때 template ID를 확정한다.
2. 가능한 projection은 같은 template tile을 연속 pack한다.
3. FFN은 exact neuron permutation으로 그룹화한다.
4. 바꿀 수 없는 Q/O/Down output 순서는 짧은 homogeneous run만 묶는다.

---

## 7. 성능 상한과 적용 범위

### 7.1 Amdahl 상한

짧은·중간 prompt에서는 projection 비중이 높아 WCEP의 영향이 크다. 긴 prompt에서는
F32 attention core 비중이 커지므로 동일 kernel speedup의 E2E 효과가 줄어든다.

따라서 “모든 S에서 20%”와 “평가 길이 geometric mean 20%”를 구분한다.

필수 보고:

```text
S ∈ {447, 1789, 7212}
각 길이의 Q4 projection share g(S)
kernel speedup r(S,B)
예측 1 / (1-g+g/r)
실측 TTFT speedup
```

### 7.2 목표 등급

| 등급 | 조건 | 논문에서의 위치 |
|---|---|---|
| 실패 | TTFT <1.10× | negative result |
| 보조 | 1.10~1.19× | 엔진 최적화 |
| 최소 성공 | **≥1.20×** | 독립 연산 기여 후보 |
| 강함 | ≥1.35× | 논문 핵심 기여 가능 |
| 획기적 | ≥1.50× | headline 후보, 강한 재현 필요 |

### 7.3 50% 이상을 내려면

현재 `g≈0.86`에서 TTFT 1.50×를 내려면 Q4 projection이 약 1.68× 빨라야 한다.

\[
1.50=\frac{1}{0.14+0.86/r}
\quad\Rightarrow\quad r\approx1.68
\]

이는 단순한 LUT build 공유로는 어렵다. 다음이 동시에 필요할 가능성이 높다.

- 실제 dot/lookup µop 30~40% 이상 감소
- register spill 없음
- template dispatch 거의 0
- B=16에서 현재 SDOT의 output-lane 이용률보다 높은 이용률
- quantize/repack/materialization 보조 비용 감소

따라서 50% TTFT는 stretch로만 두며 feasibility 결과 전에는 초록·계획의 약속값으로 쓰지
않는다.

---

## 7.4 사전 gate — 물리적 상한부터 확인한다 (§20 보다 먼저)

Phase 1(weight scanner) 이전에 **더 값싼 반증 경로**가 있다. 각 반나절이며, 하나만
실패해도 6~10주를 아낀다.

### 7.4.1 Q4 weight traffic (정정)

레이어 하나의 주요 projection parameter 수:

    4096^2 + 2*(4096*1024) + 4096^2 + 3*(4096*14336) = 218,103,808

Q4_0 은 32 weight 마다 `4-bit code 16 B + FP16 scale 2 B = 18 B` 이므로

    218,103,808 * 18/32 ≈ 117 MiB / layer
    32 layer            ≈ 3.66 GiB

(embedding·lm_head 별도. 전체 `.m` 6.32 GB 와 일관.)

### 7.4.2 산술강도와 machine balance — `B=16` 이 ridge point 근처다

Q4 weight byte 당 이상적 연산량:  `I(B) = 2B / (18/32) ≈ 3.56 B`

| Batch | 이상적 연산강도 |
|---|---|
| 4 | 14.2 ops/B |
| 8 | 28.4 ops/B |
| **16** | **56.9 ops/B** |
| 32 | 113.8 ops/B |

machine balance = `330 GOPS / 6.36 GB/s ≈ 51.9 ops/B`
(6.36 GB/s 는 오늘 affinity·barrier 검증된 4스레드 STREAM 실측, §7.15(e))

- `B=8` — 대역폭 바운드 가능성 큼
- **`B=16` — ridge point 근처**
- `B=32` — 연산 바운드 가능성 큼

scale·activation·output traffic 을 넣으면 `B=16` 은 대역폭 쪽으로 더 이동한다.
**즉 0a 측정은 필수지만, 이 계산만으로 WCEP 를 미리 기각할 근거는 아니다.**

### 7.4.3 기각 조건은 batch 별이며 조건부다

WCEP 가 현재 형태처럼 weight byte 를 줄이지 않고 정수 명령만 줄인다면, 실행이
대역폭에 완전히 포화된 조건에서는 효과가 거의 없다. 그러나 **단일 `B` 결과로 전체를
기각하면 안 된다.**

    실용 B ∈ {16,32,64} 전부에서 memory-bound
      + WCEP 가 weight bytes 를 줄이지 못함
      → WCEP 기각

    B=16 만 memory-bound 이고 B=32 이상은 compute-bound
      → B 이동을 포함한 E2E 상한을 계산한 뒤 판단

기각하면 안 되는 경우: ① `B` 최적점이 16→32 로 이동 ② 현재 커널이 STREAM 보다
낮은 이유가 nibble unpack dependency 나 register stall ③ WCEP 의 packed descriptor
가 실제 weight traffic 도 줄임.

**판정은 세 증거가 함께 맞을 때만 강하다.**

1. achieved bandwidth 가 application-specific ceiling 에 근접
2. thread 증가에 따라 연산량이 아니라 bandwidth 와 성능이 같이 포화
3. CPU clock 또는 정수 instruction 수를 바꿔도 성능이 거의 변하지 않음

`BW_achieved ≈ BW_STREAM` 만으로 판정하지 않는다 — 접근 패턴이 다르고, 4-thread
STREAM 자체가 비정상적으로 감소한다(9.59 → 6.36 GB/s, §7.15(e)).

### 7.4.4 `g(S)` 는 att share 에서 역산하면 안 된다

§0.3 의 `g ≈ 0.86` 은 `S=448` 한 점이다. 긴 프롬프트에서는 attention core 비중이
커지므로 `g` 가 떨어진다. 다만 `16-derivepp.md` 의 `att share = 0.68` 에서
`g = 0.32` 로 역산하면 **틀린다** — att share 안에 Q/K/V·O projection(WCEP 대상)이
포함돼 있기 때문이다. `0.32` 는 FFN 만 센 보수적 하한이다.

직접 측정해야 한다.

    g(S) = ( T_FFN(S) + T_QKVO(S) ) / T_prefill(S)

동일 계측기의 `ffnMs`, `attnProjMs`, `attnCoreMs` 를 쓴다.

### 7.4.5 사전등록 성공 기준 (데이터 보기 전 확정)

| 지표 | 기준 |
|---|---|
| 1차 성공 | 세 길이 **geometric mean ≥ 1.20×** |
| 안전성 | 어떤 길이에서도 5% 이상 slowdown 없음 |
| 짧은/중간 | S=447, 1789 각각 ≥ 1.20× 권장 |
| 긴 길이 | S=7212 는 별도 보고 |

"모든 prompt 길이에서 20%" 로 정의하면 훨씬 엄격하다 — attention core 를 건드리지
않는 WCEP 가 `S=7212` 에서도 1.20× 를 내려면 projection 가속률이 매우 커야 한다.

### 7.4.6 null 은 histogram-matched 여야 한다

단순 uniform random Q4 는 **약한 대조군**이다. Q4_0 은 16개 값만 가지므로 랜덤에서도
CSE 가 상당히 작동한다. 네 종류 null 과 비교한다.

1. Uniform Q4
2. projection 별 nibble histogram 보존 random
3. tile 별 histogram 보존, 위치만 셔플
4. 레이어·projection·block scale 분포까지 보존한 shuffled null

핵심 비교값:  `Δ_t = Saving(W_t^real) − median_r Saving(W_{t,r}^null)`

수백만 tile 이므로 **p-value 는 거의 항상 작다. 효과크기를 사전등록한다.**

    절대 weighted µop 절약           ≥ 25%
    matched-null 대비 추가 절약      ≥ 10 %p
    layer-cluster bootstrap 95% 하한 ≥ 5 %p
    template 64~128개 coverage       ≥ 90%

**두 판정을 분리한다.**

    actual ≈ null, 둘 다 빠름  → generic exact mixed kernel.
                                  "weight-aware" novelty 는 철회
    actual > null              → WCEP 핵심 가설 지지

### 7.4.7 정확성 gate 는 동일 실행 구성끼리

공식 gate:

    N=1 baseline ↔ N=1 WCEP
    N=8 baseline ↔ N=8 WCEP
    (동일 B·thread·partition·node order·compiler flags)

`N=1 vs N=8` 비교는 **baseline invariance audit** 이지 WCEP 통과 조건이 아니다 —
stage boundary 의 cast·pipe bridge·buffer 표현이 다를 수 있다. 같으면 전역
bit-identical 주장 가능성이 열리고, 다르면 기존 차이로 기록한다.

단 **Q4×Q8 block 과 GEMM 수준은 분산과 무관하므로 무조건 bit-identical** 이어야 한다.

### 7.4.8 수정된 작업량

| 작업 | 추정 |
|---|---|
| 0a~0c 사전 gate | 1일 |
| weight scanner / null 실험 | 2~4일 |
| symbolic IR·CSE·verifier | 1~2주 |
| ISA cost model·합성기 | 1~2주 |
| microkernel·code generation | 1~2주 |
| permutation·loader | 1주 |
| executor·PP 통합·검증 | 1~2주 |
| **총합** | **약 6~10주** |

§18 의 20~36일은 낙관적이다. Phase 0~1 에서 며칠 안에 기각되는 것이 가장 좋은 실패다.

### 7.4.9 실행 순서

    0a. B별 Q4 GEMM roofline 판정      arithmetic-only WCEP 의 물리적 상한
    0b. S별 g(S) 직접 측정             필요 kernel speedup 계산
    0c. 정확성 기준선 audit            공식 gate 확정
    0d. null 설계·효과크기 사전등록    결과 보기 전 threshold 고정
    1.  actual-weight scanner          실제 weight vs matched null
    2.  합성기/커널 구현 여부 결정

---

## 7.5 0a 결과 — roofline 판정: **기각 조건 미충족, 진행 가능**

`prefill_bench/bench_roofline.cpp`, `artifacts/wcep_0a/roofline_repack.tsv`.
production 과 동일 경로: `block_q4_0x4` × `block_q8_0x4` → `ggml_gemm_q4_0_4x4_q8_0`,
출력 열 4의 배수 분할, 스레드별 활성화 중복 repack. reps=5, best-of.

### 7.5.1 ⚠️ 커널 오인 — baseline 을 2.6배 낮게 잴 뻔했다

첫 판에서 `llamafile_sgemm` 을 벤치했는데 **production 경로가 아니다.**

| 경로 | 1스레드 GOPS |
|---|---|
| `llamafile_sgemm` | 18~27 |
| **`ggml_gemm_q4_0_4x4_q8_0` (production)** | **65~86** |

2.6배 차이다. §1.2 가 "옛 dequant 경로를 비교 대상으로 삼지 말라" 고 경고했는데
정확히 그 실수를 할 뻔했다. **§1.2 의 금지 목록에 `llamafile_sgemm` 경로를 추가한다.**

비교 기준을 명시한다:

    ggml_gemm_q4_0_4x4_q8_0  (block_q4_0x4 × block_q8_0x4)
      + 출력 열 4의 배수 분할
      + 스레드별 활성화 중복 repack

4스레드 최대 **271~326 GOPS** 로, §1.2 의 "약 330 GFLOPS" 주장이 확인된다.

### 7.5.2 측정 결과

| shape | B | GOPS@4t | BW GB/s | STREAM(6.36) 대비 | 4t/1t |
|---|---|---|---|---|---|
| Q/O 4096→4096 | 16 | 274.6 | 5.00 | **79%** | 3.38 |
| | 32 | 271.3 | 2.55 | 40% | 3.23 |
| | 64 | 301.0 | 1.51 | 24% | 3.60 |
| K/V 4096→1024 | 16 | 270.8 | 5.03 | **79%** | 4.05 |
| | 32 | 288.4 | 2.83 | 44% | 4.07 |
| Gate/Up 4096→14336 | 16 | 297.5 | 5.39 | **85%** | 3.52 |
| | 32 | 301.9 | 2.81 | 44% | 3.54 |
| Down 14336→4096 | 16 | 322.0 | 5.75 | **90%** | 3.75 |
| | 32 | 326.4 | 2.96 | 47% | 3.81 |
| | 64 | 325.2 | 1.52 | 24% | 3.82 |

### 7.5.3 세 증거 판정 (§7.4.3)

1. **ceiling 근접도** — `B=16` 은 STREAM 의 **79~90%** 로 경계에 있다.
   `B=32` 는 40~47%, `B=64` 는 23~25% 로 멀다.
2. **thread scaling** — `B≥16` 에서 3.2~4.1× 로 잘 확장된다. 대역폭 완전 포화라면
   꺾여야 하는데 그렇지 않다.
3. **`B` 에 따른 포화** — `B=16` 에서 275~322 GOPS 도달 후 `B=32,64` 에서 거의
   오르지 않는다. **연산 자원이 한계**라는 신호다.

**판정: 기각 조건("B ∈ {16,32,64} 전부에서 memory-bound") 미충족.**
`B=32` 이상은 명확히 연산 바운드이므로 산술 µop 감소가 그대로 이득이 된다.
**WCEP 는 물리적 상한에서 막히지 않는다 — 진행 가능.**

### 7.5.4 함의 — 최적 `B` 가 이동할 것이다

현재 DerivePP 의 선택 `B=16` 은 대역폭 경계에 있다. WCEP 가 산술을 줄이면 그 지점의
병목이 대역폭으로 완전히 넘어가므로, **최적 `B` 가 32 쪽으로 이동할 가능성이 크다.**
§13.3(“새 kernel 은 batch 효율 곡선을 바꾸므로 `B=16` 을 그대로 가져오면 안 된다”)이
정확히 이 상황이며, 통합 시 `B` 재측정은 선택이 아니라 필수다.

### 7.5.5 부산물 — 작은 배치에서 4스레드가 2스레드보다 느리다

| shape | B=4: 2t → 4t | B=8: 2t → 4t |
|---|---|---|
| Q/O | 129.8 → 106.2 | 125.5 → 111.0 |
| Gate/Up | 127.9 → **97.5** | 134.4 → **98.4** |
| Down | 134.6 → 105.3 | 137.3 → 105.1 |

`B≤8` 에서 4스레드가 2스레드보다 **20~27% 느리다.** `16-derivepp.md` §7.15(e)의
STREAM 관측("4스레드 대역폭이 1스레드보다 34% 낮다")과 같은 방향이며, 이 기기의
메모리 서브시스템이 동시 스트림에 취약하다는 것이 **두 번째로 재현**됐다.
원인은 여전히 미규명이며 논문 주장으로 올리지 않는다.

---

## 7.6 0b 결과 — `g(S)` 직접 측정과 목표 재보정

`artifacts/wcep_0b/`. 단일 노드, `--stage-timing 1`, wave 끔(op breakdown 은 wave 에서
집계되지 않는다), `nBatches=32`. warm-up 1 + 본측정 3.

    g(S) = (ffnMs + attnProjMs) / prefillMs        ← attnMs(core)는 WCEP 비대상

### 7.6.1 측정값

**표기 주의: 현재 값은 `g(S; B=32, N=1, 현재 baseline)` 이다.**
최종 시스템 baseline 은 `N=8, B=16` 이므로 정확한 E2E 예상에는 해당 조건의 `g` 가
필요하다. 최소한 `B=16` vs `B=32` 의 component share 차이를 확인해야 하며,
WCEP 가 최적 `B` 를 바꿀 가능성(§7.5.4)이 있으므로 최종 분석은

    E(S,B) = 1 / ( 1 − g(S,B) + g(S,B)/r(S,B) )

형태여야 한다.

| S | g (median, 기본) | g (clean pooled, 보조) | 무한가속 상한 `1/(1−g)` |
|---|---|---|---|
| 447 | **0.863** | 0.863 | 7.30× |
| 1789 | **0.680** | 0.685 | 3.12× |
| 7212 | **0.439** | 0.439 | 1.78× |

`S=447` 의 0.863 은 §0.3 의 `g ≈ 0.86` 가정과 **정확히 일치**한다.

⚠️ **`g(7212) ≈ 0.32` 은 틀렸다.** `att share = 0.68` 에서 역산한 값인데, att share
안에 Q/K/V·O projection(WCEP 대상)이 포함돼 있다. 실측은 **0.439** 다(§7.4.4 참조).

### 7.6.2 `S=7212` 의 하드 실링

    1 − g(7212) = 0.561
      = attnCore/prefill ≈ 53%  +  executor·norm·residual 등 비대상 ≈ 3%

> `S=7212` 에서는 attention core 와 기타 비대상 연산이 약 56% 를 차지하므로,
> Q4 projection 을 무한히 가속해도 TTFT speedup 은 약 **1.78×** 로 제한된다.

단 이것이 "WCEP 가 긴 입력에서 무의미하다" 는 뜻은 아니다. `r=1.5` 만으로도 약
**17%** 향상이 예측된다.

### 7.6.3 필요 커널 가속률 — 두 양을 혼동하지 않는다

**틀린 계산:** 길이별로 각각 1.20× 를 내는 `r_S` 의 기하평균 = 1.38×.
이는 "공통 `r` 로 기하평균 1.20× 를 만드는 값" 이 **아니다.**

**옳은 조건:**

    [ Π_S  1/(1 − g(S) + g(S)/r) ]^(1/3)  ≥  1.20      →   r ≥ 1.334

| 공통 `r` | S=447 | S=1789 | S=7212 | **E2E geo평균** |
|---|---|---|---|---|
| 1.30 | 1.249 | 1.186 | 1.113 | 1.181 |
| 1.33 | 1.273 | 1.203 | 1.122 | 1.198 |
| **1.34** | 1.280 | 1.208 | 1.125 | **1.203** |
| 1.40 | 1.327 | 1.241 | 1.144 | 1.235 |
| 1.50 | 1.404 | 1.293 | 1.172 | 1.286 |

    최소 kernel 목표   약 1.34×
    안전한 목표        1.40×
    강한 목표          1.50× 이상

§0.3 의 "커널 1.25× → E2E 1.20×" 는 `g=0.86` 단일값 가정이라 **`S=447` 에서만**
성립한다. 세 길이 기준으로는 1.34× 가 필요하다.

### 7.6.4 성공 기준은 바꾸지 않는다

"짧은/중간 길이만" 기준으로 완화해도 실익이 없다.

    기준 A (세 길이 geo평균 ≥1.20×)     필요 r = 1.334
    기준 B (447·1789 각각 ≥1.20×)       필요 r = 1.325
    차이 0.7% — 개발 난도가 실질적으로 낮아지지 않는다

**따라서 §7.4.5 의 사전등록을 유지하고 구조만 명확히 한다.**

| 층 | 기준 |
|---|---|
| Primary | 세 길이 E2E geometric mean ≥ **1.20×** |
| Secondary | S=447, 1789 각각 ≥ 1.20× |
| Long-context | S=7212 결과와 Amdahl ceiling(1.78×) 별도 보고 |
| **Kernel gate** | 주요 projection 가중평균 ≥ **1.35×** |
| 안전성 | 어떤 길이에서도 5% 이상 slowdown 없음 |

### 7.6.5 Phase 1 진입 기준 상향

Phase 1 scanner 의 예상 projection speedup 진입선을 **≥1.35×** 로 올린다.
1.30× 예측이면 합성기를 만들지 않고 기각한다 — 위 표에서 1.30× 는 geo평균 1.181×
로 목표 미달이기 때문이다. (§9.3 kill criterion 2 의 `1.30×` 를 `1.35×` 로 갱신.)

### 7.6.6 `S=1789` 이상치 처리

| rep | prefillMs | ffnMs | attnProjMs | attnCoreMs | g |
|---|---|---|---|---|---|
| 1 | 146,725 | 82,806 | 18,443 | 36,094 | 0.690 |
| 2 | **353,905** | 80,719 | 18,444 | **242,346** | **0.280** |
| 3 | 143,333 | 79,361 | 18,071 | 37,252 | 0.680 |

rep2 의 `attnCore` 가 정상 회차의 **6.6배**다. `ffnMs`·`attnProjMs` 는 정상이므로
attention core 만 튀었다.

**처리 방침** — 단순 삭제하지 않는다.

    기본값  전 회차 median            g = 0.680
    보조값  이상치 제외 pooled        g = 0.685
    원자료와 제외 사유를 모두 공개하고, 가능하면 정상 회차를 1~2회 추가한다

⚠️ 이 이상치가 OS stall 인지 profiler attribution 오류인지 **확인되지 않았다.**
`16-derivepp.md` §7.15(f)의 node 113 attention 이상과 원인을 연결해서는 안 된다.

---

## 7.7 0c 결과 — 정확성 기준선 audit 과 공식 gate 확정

`artifacts/wcep_0c/`. logits 덤프를 위해 `DLLAMA_DUMP_LOGITS=<path>` 를 추가했다
(`src/dllama.cpp`, prefill 종료 직후 float32 raw 1회 기록 — 성능 경로 무영향).

### 7.7.1 1층 — baseline 자체 결정성: **통과**

동일 prompt·model·thread, greedy(temp 0), `S=447`, 각 3회.

| config | rep 1~3 logits md5 |
|---|---|
| N=1, B=16 | `8b8178a50a97` (3회 동일) |
| N=1, B=32 | `8b8178a50a97` (3회 동일) |
| N=8, B=16 | `c52e50e37e65` (3회 동일) |

**세 구성 모두 회차 간 완전히 결정적이다.** 분산 실행도 비결정성을 도입하지 않는다 —
스레드 스케줄링·네트워크 타이밍이 결과를 바꾸지 않는다. WCEP 의 bit-identical gate 를
정의할 수 있다.

### 7.7.2 `B` 불변성 — gate 조합이 절반으로 줄어든다

`N=1` 에서 **`B=16` 과 `B=32` 의 logits 가 bit-identical** 이다(둘 다 `8b8178a5`).

마이크로배치 폭이 FP32 누적 순서를 바꾸지 않는다는 뜻이다. causal attention 에서
토큰 `i` 의 출력은 `0..i` 에만 의존하고 청크 분할이 그 순서를 바꾸지 않기 때문이다.

⚠️ **구조적 보장으로 일반화하면 안 된다.** 현재 구현에서 두 batch 구성이 같은
logits 를 냈을 뿐이며, WCEP 가 `B` 별로 다른 template 이나 reduction 경로를 쓰면
결과가 달라질 수 있다.

    reference 는 N 별로 하나만 보존   →  2개
    gate 실행은 여전히 B 별로 전부    →  4개

    N=1 reference  8b8178a50a97
      ├─ B=16 WCEP 비교
      └─ B=32 WCEP 비교
    N=8 reference  c52e50e37e65
      ├─ B=16 WCEP 비교
      └─ B=32 WCEP 비교

### 7.7.3 2층 — `N=1` vs `N=8`: **다르다** (예상대로, WCEP gate 아님)

    N=1  8b8178a50a97
    N=8  c52e50e37e65

> `N` 에 따른 차이는 **q80 stage-boundary quantization 경로와 일관된다.**
> 정확한 최초 divergence 지점은 layer-boundary dump 로 추후 확인한다.

(`--buffer-float-type q80` 이므로 stage 경계마다 활성값이 양자화 왕복을 거치는 반면
단일 노드는 FP32 를 그대로 넘긴다 — 코드 구조상 가장 유력한 설명이지만 **경계 덤프로
확인하기 전까지 단정하지 않는다.**)

**이는 기존 시스템의 특성이며 WCEP 통과 조건이 아니다.** 다만 논문에서
"bit-identical" 을 말할 때 **범위를 명시**해야 한다 — 전역이 아니라 동일 실행 구성
안에서다.

### 7.7.4 3층 — WCEP 공식 정확성 gate (확정)

**항상 동일 실행 구성끼리 비교한다.**

    N=1, B=16 baseline ↔ N=1, B=16 WCEP
    N=8, B=16 baseline ↔ N=8, B=16 WCEP
    (B 는 §7.7.2 로 불변이 확인됐으나, WCEP 는 커널을 바꾸므로 재확인한다)

| 계층 | 정확성 기준 |
|---|---|
| Q4×Q8 block | **bit-identical** (분산과 무관) |
| 전체 GEMM | **bit-identical** (분산과 무관) |
| layer/executor | 동일 구성에서 bit-identical |
| logits | 동일 구성에서 bit-identical |
| token sequence | greedy 에서 동일 |
| `N=1` vs `N=8` | **audit 만 — WCEP gate 아님** |

⚠️ **정수 내적이 수학적으로 같다는 것만으로 부족하다.** block 별 scale 적용 순서와
FP32 누적 순서를 바꾸면 bitwise 결과가 달라진다. 따라서 reference 에는 현재
baseline 의 **accumulation order 까지** 포함한다. 위 logits 파일이 그 역할을 한다.

### 7.7.5 산출물

    artifacts/wcep_0c/
      hashes.tsv                       config × rep × logits md5 × prefillMs
      raw/logits_n1_B{16,32}_r{1..3}.bin
      raw/logits_n8_B16_r{1..3}.bin
      raw/*.log

**0c 상태 분리**

| | 내용 | 상태 |
|---|---|---|
| 0c-a | logits 결정성 + gate 정의 | **완료** |
| 0c-b | block/layer reference hook | **미완 — WCEP 커널 검증 전 필수** |
| | Phase 1 정적 scanner 진행 | 영향 없음 |

`reference_q4q8.bin`(block 수준)과 `reference_layer_outputs.bin`(layer 수준)은
아직 없다. 각각 Q4×Q8 GEMM 과 layer 경계에 덤프 훅이 필요하며, WCEP 커널 구현
직전에 추가한다. 현재 logits 수준 reference 만으로 Phase 1 진행에는 지장이 없다.

### 7.7.6 운영 메모 — 재배포 자동화 필요

logits 덤프 추가로 root 를 재빌드하자(`7723c5d4` → `fb66e60f`) 워커가 구버전이 되어
`start_workers.sh` 의 md5 가드가 실행을 차단했다. **이 가드가 다섯 번째로 값을 했다** —
없었다면 구버전 워커와 신버전 root 가 섞인 상태로 정확성 audit 을 돌려, logits 차이가
`N` 축 때문인지 바이너리 불일치 때문인지 구분할 수 없었을 것이다.

WCEP 는 커널을 반복적으로 고치므로 **빌드 직후 재배포를 자동화**한다.

---

## 7.8 0d — null 설계·표본·통계 사전등록 (**scanner 코드 작성 전 확정**)

이 절은 측정이 아니라 **규칙 고정**이다. 결과를 보기 전에 확정한다.

### 7.8.1 tile 정의 — 셔플 단위는 컴파일 단위와 일치해야 한다

    WCEP tile = 4개 output row × 32개 input weight

null 생성의 셔플 단위를 이와 다르게 잡으면 "구조가 있다/없다" 판정이 무의미해진다.

### 7.8.2 네 종류 null

| # | null | 보존하는 것 | 검사하는 것 |
|---|---|---|---|
| 1 | **Uniform** | 없음 | 값 분포 자체의 기여 |
| 2 | **Projection-histogram** | layer·projection 전체 nibble 개수 | tile 배치의 기여 |
| 3 | **Tile-histogram** | 각 tile 의 16개 값별 빈도 | **값 빈도 효과 vs 공간 배열 효과 분리** |
| 4 | **Cross-output** | 각 output row 의 histogram | **output row 간 공통 부분식이 실제 구조인가** |

3번이 가장 중요하다 — 값 빈도만으로 설명되는 절약과 배열에서 오는 절약을 가른다.
4번은 WCEP 가 §4.2(4)에서 주장하는 "output column 사이 반복 부분합" 이 실재하는지를
직접 검사한다.

**scale 분포 보존** 은 µop 구조 분석에는 직접 영향이 없으나 descriptor 크기와 실제
kernel 측정에는 영향을 준다 → **runtime 단계의 보조 null** 로 둔다.

### 7.8.3 표본 — 값싼 것은 전수, 비싼 것만 층화

    전수 조사    nibble histogram, entropy, 반복 pattern, 단순 CSE 상한
    층화 표본    beam search (비쌈)
                 layer × projection 마다 동일 개수의 tile
                 고정 random seed
                 actual tile 마다 null tile 을 **paired** 생성

### 7.8.4 통계 단위는 tile 이 아니라 **layer**

수백만 tile 을 독립 표본으로 취급하면 신뢰구간이 부당하게 작아진다.

    Δ(ℓ,p) = Saving_actual(ℓ,p) − median_r Saving_null(ℓ,p,r)

layer-cluster bootstrap 으로 95% 신뢰구간을 계산한다.

### 7.8.5 효과크기 사전등록 (§7.4.6 재확인)

    절대 weighted µop 절약            ≥ 25%
    matched-null 대비 추가 절약       ≥ 10 %p
    layer-cluster bootstrap 95% 하한  ≥ 5 %p
    template 64~128개 coverage        ≥ 90%

p-value 는 쓰지 않는다 — 표본이 커서 거의 항상 유의하다.

### 7.8.6 합성기 과적합 방지 — 두 결과를 분리한다

template 을 발견한 데이터와 평가 데이터가 같으면 coverage 가 낙관적으로 나온다.

| 결과 | 정의 |
|---|---|
| **Operational** | 전체 weight 로 template 생성 → 같은 모델 전체 적용 |
| **Generalization** | 짝수 layer 에서 template 생성 → **홀수 layer 에서 측정** |

최종적으로는 **다른 모델 held-out** 이 더 강한 검증이다.

### 7.8.7 합성기 파라미터는 actual 과 null 에 **동일 적용**

    beam width, DAG depth 상한, live register 상한(28),
    coefficient 절댓값 상한, baseline 후보 포함 여부

하나라도 다르면 비교가 무효다.

---

## 7.9 Phase 1 재구성 — 구조 gate 와 성능 gate 를 분리한다

§9.3 의 "예상 speedup ≥1.35×" 는 **µop 개수만으로 판정할 수 없다.** 명령마다
throughput·latency 가 다르고 register spill 도 있다. 세 단계로 나눈다.

| 단계 | 내용 | 통과 조건 |
|---|---|---|
| **1a** | scanner — 구조적 절약 가능성 | weighted µop 감소 ≥25%, matched-null 대비 ≥10 %p, template coverage ≥90% |
| **1b** | 작은 ISA primitive benchmark | SDOT·TBL·add·shuffle, dependency chain 과 독립 명령 각각, spill penalty 포함 |
| **1c** | ISA 비용을 적용한 예상 kernel speedup | **weighted predicted speedup ≥1.35×** 일 때만 본 합성기 구현 |

1a 는 구조가 있는지, 1c 는 그 구조가 **실제 하드웨어에서 이득이 되는지**를 본다.
1a 통과 + 1c 미달이면 "구조는 있으나 ISA 에서 회수되지 않는다" 는 negative result 다.

---

## 8. Phase 0 — 현재 결과 보존

### 목적

WCEP가 실패해도 DerivePP의 4.57~4.68× 결과와 논문 artifact가 손상되지 않게 한다.

### 작업

1. 현재 commit, submodule, dirty patch, compiler 정보를 snapshot한다.
2. `wcep` 별도 branch/worktree 또는 clone을 만든다.
3. current repack binary와 benchmark raw log를 고정한다.
4. accuracy reference vector를 생성한다.

### 산출물

```text
artifacts/wcep_baseline/
  manifest.txt
  current_repack_bench.tsv
  reference_q4q8_vectors.bin
  reference_layer_outputs.bin
  reference_logits.bin
```

---

## 9. Phase 1 — actual-weight feasibility audit

**이 단계가 전체 계획의 최우선이다. 커널을 먼저 작성하지 않는다.**

### 9.1 weight scanner

Llama-3 8B Q4_0의 다음 projection을 전부 스캔한다.

```text
Q, K, V, O
FFN Gate, Up, Down
lm_head (분리 보고)
```

tile별로 기록한다.

- nibble histogram과 entropy
- 1/2/4항 pattern 빈도
- output-column 간 동일 부분식 수
- pure SDOT 예상 µop
- pure T-MAC 예상 lookup/add
- WCEP beam search 예상 µop
- live register와 DAG depth
- canonical template ID

### 9.2 보고할 분포

평균 하나로 판정하지 않는다.

```text
p10 / median / p90 instruction saving
projection별 saving
레이어별 saving
상위 K template coverage
baseline fallback tile 비율
```

### 9.3 Phase 1 kill criterion

다음 중 하나면 WCEP 핵심 구현을 중단한다.

1. weighted median 예상 µop 감소가 25% 미만
2. FFN 세 projection의 가중 예상 speedup이 **1.35× 미만** (§7.6.5 로 상향, 원래 1.30×)
3. 128개 template로 90% tile을 처리하지 못함
4. 대부분 후보가 register 28개를 초과
5. 실제 weight가 random Q4 synthetic baseline과 차이가 없음

5번은 중요하다. 실제 weight에서만 나타나는 구조가 없다면 “weight-aware compilation”의
필요성과 novelty가 모두 약해진다.

---

## 10. Phase 2 — ISA characterization과 합성기

### 10.1 Pi 5 primitive benchmark

다음을 dependency-chain과 independent-chain으로 각각 잰다.

- `SDOT` lane/vector variants
- `TBL` 1/2-table variants
- widening add/sub
- shift + add
- vector load와 descriptor load
- `TBL`과 `SDOT` interleave
- horizontal reduction
- register 수 증가에 따른 spill knee

출력은 instruction별 latency 하나가 아니라 혼합 sequence별 cycles/iteration이다.

### 10.2 합성기 v0

1. coefficient-vector IR
2. pure SDOT/pure LUT baseline generator
3. pairwise CSE
4. bounded beam search
5. symbolic range/equality verifier
6. cost-ranked template canonicalizer

### 10.3 selftest

- 작은 `K=4/8`은 brute force 최적값과 비교
- random Q4 tile 10만 개
- actual weight tile 전수 symbolic equality
- scalar reference와 random Q8 실행 비교

---

## 11. Phase 3 — 32×4 microkernel

### 11.1 순서

```text
A. pure SDOT 재현       현재 kernel과 성능/출력 기준선
B. pure LUT             T-MAC 계열 기준선
C. mixed WCEP           합성 DAG
D. mixed + template run 연속 실행
```

### 11.2 실제 형상

```text
B ∈ {4, 8, 16, 32}
4096  → 4096    Q/O
4096  → 1024    K/V
4096  → 14336   Gate/Up
14336 → 4096    Down
```

`B=16`이 현재 DerivePP의 핵심이므로 batch 1 결과만으로 통과시키지 않는다.

### 11.3 Phase 3 gate

```text
대표 형상 가중평균 <1.25×      → 전체 목표 미달, 중단
FFN 가중평균 <1.30×            → 20% TTFT 가능성 낮음, 중단 우선
어느 주요 형상도 >5% slowdown  → fallback/grouping 수정 전 통합 금지
bit mismatch                  → 성능과 무관하게 실패
```

---

## 12. Phase 4 — FFN permutation clustering

### 12.1 loader transform

레이어마다 하나의 허용 neuron permutation \(P\in\mathcal P_{Q4}\)를 계산한다.

목적함수:

\[
\min_P
\left[
N_{switch}(P)+alpha N_{template}(P)+\beta C_{descriptor}(P)
\right]
\]

제약:

- Gate/Up row에 같은 \(P\)
- Down input column에 \(P^{-1}\)
- Q4 value와 scale bit pattern 보존
- Down의 32-column scale block을 분해하지 않음
- output tensor의 의미 동일

첫 구현은 optimal assignment가 아니라 template ID stable-sort로 충분하다. 그 뒤 필요할 때만
전환 비용을 포함한 clustering을 쓴다.

### 12.2 정확성 gate

```text
dequantized Gate/Up/Down matrix의 permutation 대응 완전 일치
FFN 단독 F32 output bit-identical
한 layer output bit-identical
```

허용 집합 안에서도 loader의 byte layout 때문에 exact column permutation이 불가능한 경우
re-quantize하지 않는다. 그 경우 permutation 확장은 기각하고 WCEP core만 유지한다.

---

## 13. Phase 5 — executor와 DerivePP 통합

### 13.1 단계별 통합

1. 단일 projection
2. FFN 한 block
3. Transformer 한 layer
4. 32-layer 단일 노드
5. DerivePP stage 하나
6. PP8 wave pipeline

각 단계에서 bit gate를 통과한 binary만 다음 단계로 넘긴다.

### 13.2 PP와의 관계

WCEP는 pipeline 식의 `C(B)`를 줄인다.

\[
T_{PP}\approx
\left(\left\lceil\frac{S}{B}\right\rceil+N-1\right)C(B)
\]

통신량, stage 수, microbatch release 시점은 바꾸지 않는다. 그러므로 기존 query fusion처럼
실제 pipeline 단위가 `G·B`로 커지는 문제가 없다.

### 13.3 B 재검증

새 kernel은 batch별 효율 곡선을 바꾸므로 현재 `B=16` 결과를 그대로 가져오면 안 된다.
이는 AxisCert를 논문 중심으로 되돌린다는 뜻이 아니라, 새 연산 kernel의 성능 knee를 다시
재는 정상적인 통합 절차다.

```text
B ∈ {8,16,32,64}
같은 세션 anchor 교차 측정
새 B_min 기록
```

---

## 14. 정확성 검증 계획

### 14.1 계층별 gate

| 단계 | 비교 | 통과 조건 |
|---|---|---|
| block | INT32 dot | 모든 값 동일 |
| GEMM | F32 output | bit-identical |
| FFN | residual output | bit-identical |
| layer | hidden state/KV | bit-identical |
| model | logits | bit-identical |
| serving | token sequence | 동일 |

### 14.2 데이터

- synthetic random
- saturation/adversarial Q8
- 실제 calibration activation
- S=447/1789/7212 prompt
- 3B/8B/13B 모델

### 14.3 금지되는 타협

- perplexity만 같으면 통과
- top-1 token만 같으면 통과
- 작은 logit tolerance로 bit mismatch 은폐
- weight 재양자화
- table quantization
- approximate fast aggregation

bit-identical을 구조적으로 달성할 수 없다는 증거가 나오면 정확도 조건을 조용히 낮추지
말고 연구 조건 불충족으로 기록한다.

---

## 15. 성능 평가 계획

### 15.1 baseline

동일한 다음 조건을 고정한다.

- 모델 파일과 Q4 format
- thread 수와 affinity
- governor=`performance`
- compiler와 flags
- prompt token IDs
- B/N/PP placement
- warm-up과 anchor 순서

### 15.2 kernel 지표

- cycles/Q4 block
- effective GOPS
- instructions, IPC
- `SDOT`, `TBL`, load, branch 수
- L1/L2/LLC miss
- spill load/store
- energy/prefill 가능 시 측정

### 15.3 end-to-end 지표

- TTFT
- `ffnMs`, `attnProjMs`, `attnMs`, `otherMs`
- 단일 노드와 PP8
- speedup geometric mean
- worst-case slowdown
- 예측 Amdahl 값과 실측 차이

### 15.4 ablation

| 구성 | 질문 |
|---|---|
| current SDOT | 현재 강한 기준선 |
| pure LUT | LUT 자체의 효과 |
| weight-aware CSE only | 실제 weight 구조의 효과 |
| mixed TBL–SDOT | 공동 primitive의 효과 |
| + template clustering | code/branch locality 효과 |
| + FFN permutation | exact symmetry 효과 |
| + supporting fusion | materialization 감소 효과 |

---

## 16. Novelty와 관련 연구 경계

### 16.1 반드시 비교할 연구

1. **T-MAC, EuroSys 2025** — bit-wise LUT low-bit CPU GEMM, ARM `TBL`, multi-batch
   prefill을 이미 지원한다.  
   <https://arxiv.org/abs/2407.00088>
2. **FBGEMM** — low-precision CPU inference와 shape-specific runtime code generation.  
   <https://arxiv.org/abs/2101.05615>
3. **constant matrix CSE / Multiple Constant Multiplication** — 실제 상수의 shift-add/CSE는
   오래된 최적화 문제다.  
   <https://doi.org/10.1016/j.jcp.2013.03.042>
4. **Radix-2^r CSE/MCM** — radix 표현과 공통 부분합으로 addition 수를 줄이는 선행.  
   <https://doi.org/10.1049/iet-cds.2020.0213>

### 16.2 주장 가능한 것과 없는 것

| 문장 | 현재 상태 |
|---|---|
| “최초의 LUT low-bit CPU GEMM” | 주장 불가 |
| “최초의 weight-specific constant-matrix CSE” | 주장 불가 |
| “Q4×Q8 block-exact TBL–SDOT 공동 합성” | 후보, 정식 조사 필요 |
| “FFN permutation으로 합성 template를 clustering” | 후보, 정식 조사 필요 |
| “강한 Q4 SDOT baseline 대비 ≥20% TTFT” | 실험 전 가설 |

### 16.3 논문 기여 문장 초안

> WCEP compiles the actual Q4 weights, rather than only their shape or bit width,
> into certified Q4×Q8 arithmetic DAGs that jointly use table-lookup and dot-product
> instructions. It then exploits the exact permutation symmetry of SwiGLU FFNs to
> cluster compatible DAG templates, preserving the original quantized model bit for
> bit while accelerating CPU prefill.

이 문장은 Phase 1~5가 모두 통과했을 때만 사용한다.

---

## 17. 리스크와 반증 가능성

### 리스크 1 — 실제 weight가 너무 random하다

Q4 code가 거의 독립·균등이면 CSE가 절약할 부분이 없다. 이 경우 synthetic random과 actual
weight의 차이가 없어지고 WCEP의 weight-aware 근거가 무너진다.

**대응:** Phase 1에서 먼저 기각한다.

### 리스크 2 — SIMD는 scalar addition count와 다르다

부분식 수가 30% 줄어도 TBL 한 명령이 이미 16 lane을 처리하므로 실제 instruction이 줄지
않을 수 있다.

**대응:** scalar op가 아니라 emitted AArch64 µop와 cycle을 비용으로 쓴다.

### 리스크 3 — TBL과 SDOT이 같은 port를 경쟁한다

혼합했지만 병렬성이 생기지 않을 수 있다.

**대응:** Phase 2 interleave microbenchmark로 먼저 판정한다.

### 리스크 4 — template/code locality

actual-weight 특화가 너무 강하면 template가 폭발한다.

**대응:** coverage gate와 FFN permutation clustering. 그래도 128개를 넘으면 기각한다.

### 리스크 5 — 현재 kernel이 이미 ceiling에 가깝다

330 GFLOPS baseline은 약하지 않다. WCEP의 줄어든 산술이 load/scale 병목에 가려질 수 있다.

**대응:** pure instruction ceiling과 weight/load floor를 따로 측정한다.

### 리스크 6 — 긴 prompt에서 attention이 지배한다

WCEP는 dense projection을 줄이지 attention의 \(O(S^2)\) 항을 줄이지 않는다.

**대응:** 길이별 speedup을 정직하게 분리한다. 긴 길이 하나에서 20% 미만이라고 짧은 길이
결과를 부풀리지 않는다.

---

## 18. 개발 순서와 예상 작업량

| 순서 | 작업 | 예상 | 중단 가능 지점 |
|---:|---|---:|---|
| 0 | baseline/artifact freeze | 0.5일 | — |
| 1 | Q4 actual-weight scanner | 1~2일 | **첫 kill gate** |
| 2 | symbolic IR + 단순 CSE | 2일 | 구조 없음 시 중단 |
| 3 | Pi 5 ISA benchmark | 1일 | 혼합 자원 이득 없음 |
| 4 | beam/e-graph 합성기 v0 | 3~5일 | template 폭발 |
| 5 | 32×4 reference microkernel | 3~5일 | **1.25× 미달 시 중단** |
| 6 | B=16 panel kernel | 3~5일 | shape 가중평균 미달 |
| 7 | FFN permutation loader | 2~4일 | exact format 불가 |
| 8 | executor/DerivePP 통합 | 3~5일 | bit gate 실패 |
| 9 | 3B/8B/13B 평가 | 3~5일 | 일반성 미달 |

전체 구현을 한 번에 승인하지 않는다. Phase 1의 weight 구조와 Phase 3의 1.25× kernel gate가
통과한 뒤에만 loader/executor를 수정한다.

---

## 19. 최종 go/no-go 표

| 질문 | Go | No-go |
|---|---|---|
| 실제 weight가 random보다 구조적인가? | µop 절약 유의 | 차이 없음 |
| template 수가 감당 가능한가? | 64~128개로 ≥90% | coverage 미달 |
| exactness가 가능한가? | block~logit bit-identical | tolerance 필요 |
| current SDOT보다 빠른가? | weighted ≥1.25× | 미달 |
| 전체 TTFT가 개선되는가? | ≥1.20×, slowdown 없음 | 미달/길이 악화 |
| T-MAC/CSE와 구별되는가? | 혼합 DAG+symmetry 기여 입증 | 단순 LUT/CSE로 환원 |

**하나라도 핵심 No-go면 WCEP를 DerivePP의 새 중심 기여로 만들지 않는다.** 결과는 exact
low-bit CPU kernel의 한계를 보여주는 negative result로 보존하고, 기존 DerivePP 논문은
wave-PP 시스템 기여를 중심으로 유지한다.

---

## 20. 바로 다음 작업

1. 현재 Q4_0 model 파일을 읽는 **read-only weight scanner**를 작성한다.
2. `Gate/Up/Down/Q/K/V/O`별 32×4 tile 통계를 artifact로 저장한다.
3. random Q4 synthetic tile과 같은 통계를 나란히 낸다.
4. 아직 assembly kernel은 작성하지 않는다.
5. Phase 1 표를 채운 뒤 WCEP 계속 여부를 결정한다.

첫 산출물은 speedup 숫자가 아니라 다음 표여야 한다.

| projection | median 예상 µop 절약 | p90 | template-64 coverage | fallback 비율 |
|---|---:|---:|---:|---:|
| Q | TBD | TBD | TBD | TBD |
| K/V | TBD | TBD | TBD | TBD |
| O | TBD | TBD | TBD | TBD |
| Gate/Up | TBD | TBD | TBD | TBD |
| Down | TBD | TBD | TBD | TBD |

이 표가 WCEP가 실제 알고리즘 연구가 될지, 흥미로운 수식에 그칠지를 결정한다.
