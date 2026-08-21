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
2. FFN 세 projection의 가중 예상 speedup이 1.30× 미만
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
