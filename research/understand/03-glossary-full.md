# 상세 용어 사전 — 00~08 전체

[08-glossary](../08-glossary.md) 의 확장판이다. 08 은 **짧게 읽히도록** 쓰였고,
여기는 **막혔을 때 찾아보도록** 쓰였다.

각 항목의 구성:

> **뜻** → **왜 이 연구에서 중요한가** → **우리 측정값** → **코드 위치** → **자주 틀리는 것**

- 08 과 중복되는 항목도 다시 쓴다 (한 곳에서 다 찾을 수 있어야 하므로).
- **모든 수치에 출처 문서를 붙였다.** 출처가 없는 수치는 쓰지 않았다.
- 코드 위치는 [01-code-map](01-code-map.md) 에 더 자세히 있다.

**찾아보기**: [Part 1 추론](#part-1-추론의-두-단계) · [Part 2 연산](#part-2-연산과-하드웨어) ·
[Part 3 분산](#part-3-분산) · [Part 4 우리 알고리즘](#part-4-우리-알고리즘의-수학) ·
[Part 5 검증](#part-5-검증과-정확도) · [Part 6 연구 방법론](#part-6-연구-방법론-용어) ·
[Part 7 선행연구](#part-7-선행연구) · [부록 숫자표](#부록-a-숫자-한-곳에)

---

# Part 1. 추론의 두 단계

## prefill / decode

**뜻.** LLM 추론은 성격이 완전히 다른 두 단계로 나뉜다.

| | prefill | decode |
|---|---|---|
| 하는 일 | 입력 프롬프트 전체를 한 번에 처리 | 토큰을 하나씩 생성 |
| 한 번에 처리하는 토큰 | S개 (우리 실험 96~2048) | 1개 |
| 병목 | **연산** (compute-bound) | **메모리 대역폭** (memory-bound) |
| 행렬 연산 형태 | GEMM (행렬×행렬) | GEMV (행렬×벡터) |
| 산출물 | KV 캐시 전체 + 마지막 로짓 | 다음 토큰 |

**왜 성격이 다른가.** 같은 가중치를 몇 번 쓰느냐의 차이다.

```
prefill : 가중치 6.3 GB 를 한 번 읽어  →  447 토큰분 계산   → 연산이 병목
decode  : 가중치 6.3 GB 를 한 번 읽어  →    1 토큰분 계산   → 읽기가 병목
```

decode 는 가중치를 읽는 시간이 곧 토큰 시간이다. 6.3 GB ÷ 8 GB/s ≈ 0.8초 —
실측 토큰당 약 0.5초([08](../08-glossary.md))와 같은 자릿수다.

**왜 이 연구에서 중요한가.** 이 프로젝트가 얹혀 있는 distributed-llama 는
**decode 를 텐서 병렬로 가속하는 것이 목표**였고 prefill 은 부산물이었다([03 §4](../03-attention-batching.md)).
발견 #1·#2 가 전부 여기서 나온다 — decode 용으로 맞춰진 코드가 prefill 경로에서 그대로 돈다.

**코드.** prefill 루프 `dllama.cpp:414~530`, 구분 스위치 `nnCpuOpsSetDecodePhase`
(`nn-cpu-ops.cpp:1280`), 다중 노드 전파 `controlPacket.phase` (`app.cpp:817`).

**자주 틀리는 것.** "prefill 이 compute-bound" 는 **배치가 충분할 때만** 참이다.
배치가 32 미만이면 prefill 도 memory-bound 다 → [machine balance](#machine-balance-균형점) 참조.

---

## TTFT (Time To First Token)

**뜻.** 프롬프트를 넣고 **첫 글자가 나오기까지의 시간**. 사용자가 체감하는 반응성.

```
TTFT = prefill + (첫 decode 스텝 1회)
prefill = Σ_chunks ( 연산 + 통신 + 동기 대기 )
```

**왜 중요한가.** 이 연구의 **목표 지표**다. prefill 시간이 거의 전부를 차지한다.

**우리 실측** ([06 §2](../06-baseline-status.md), S=447, llama3-8b_q40, Pi5 4스레드 단일 노드):

| 단계 | prefill | llama.cpp 대비 |
|---|---|---|
| 원본 | 496 ms/tok | 12 % |
| + dotprod (발견 #1) | 78,150 ms | 35 % |
| + attention 배치화 (발견 #2) | 72,938 ms | 38 % |
| + Q4_0 repack (발견 #4) | 35,889 ms | 77 % |
| + lm_head prefill 제한 | **32,417 ms** | **85 %** |

누적 **6.9×**. 네 항목 모두 출력 **byte-for-byte 동일**을 확인했다.

**코드.** `dllama.cpp:723` `ttftMs`. `hasFirstPredToken` 이면 실제 TTFT, 아니면 prefill wall.

**자주 틀리는 것.** 문서마다 단일 노드 prefill 값이 다르다(37,939 / 36,740 / 32,417 ms).
**측정 회차가 다르다** — [README §4](README.md#4-원본-문서-간-알려진-불일치) 참조.
표끼리 직접 빼서 비율을 내면 안 된다.

---

## KV 캐시

**뜻.** Attention 이 "이전 토큰들"을 참조할 때 쓰는 저장소.
각 토큰 t, 각 레이어 l 마다 `K_l[t]`, `V_l[t]` 벡터를 보관한다.

**크기** (Llama-3-8B, F32, 1노드 — [05 A-3](../05-kvcache-ttft-guide.md)):

| | 계산 | 값 |
|---|---|---|
| 토큰 1개 · 레이어 1개 | 2 × kvDim(1024) × 4 B | **8 KB** |
| 토큰 1개 (32 레이어) | × 32 | **256 KB** |
| S=2048 프롬프트 | × 2048 | **512 MB** |

**메모리 레이아웃.** `keyCache[headIndex * headDim + t * kvDim0]` —
**position-major, head-minor**. 한 위치의 모든 KV 헤드가 연속으로 놓인다.

**왜 중요한가.** **TTFT 의 산출물은 정확히 두 개다** — 전체 KV 캐시 + 마지막 토큰의 로짓
([05 B-2](../05-kvcache-ttft-guide.md), [00 §1](../00-RESEARCH-PLAN.md)).
그래서 이 연구는 "KV 캐시를 어떻게 빨리 만드는가"의 문제로 환원된다.

토큰 `0..S-2` 의 상위 레이어 활성화는 **오직 그 레이어의 KV 를 만들기 위해서만** 존재한다.
이 사실이 lm_head 제거, 깊이 분해(SwiftKV 계열), PrefillOnly 논문의 공통 출발점이다.

**코드.** 할당 `llm.cpp:306~307`, 쓰기 `OP_SHIFT` (`llm.cpp:411/417`),
슬라이싱 `nn-core.cpp:215 sliceKvCache`.

**자주 틀리는 것 (실제로 OOM 을 냈다).**
KV 버퍼는 실제 프롬프트 길이가 아니라 **`--max-seq-len` 전체로** 할당된다.
8B / seqLen 8192 이면 KV 만 2.1 GB, 모델 6.3 GB 와 합쳐 8.4 GB.
→ [06 §6 OOM #3](../06-baseline-status.md), 가드는 `prefill_bench/run_one.sh:9~13`.

> **이 코드베이스는 KV 를 F32 로 저장한다.** llama.cpp 기본은 F16.
> 메모리 2배 + **attention 메모리 트래픽도 2배**. 무손실에 가까운 개선 후보로 남아 있다
> ([06 §5](../06-baseline-status.md)).

---

## GQA (Grouped-Query Attention)

**뜻.** 쿼리 헤드는 32개인데 KV 헤드는 8개만 두고 4개씩 공유하는 구조.
`kvMul = nHeads / nKvHeads = 4`.

```
hidden h = 4096,  headDim = 128
  Q 헤드 32개 → qDim  = 4096
  KV 헤드 8개 → kvDim = 1024   ( = h/4 )
```

**왜 중요한가 — 두 방향으로 작용한다.**

1. **유리하게**: KV 캐시가 1/4 로 줄고, 노드 간 KV 교환도 1/4 이 된다.
   CP(context parallel)의 교환 대상이 hidden(4096)이 아니라 kvDim(1024)이라
   **CP 통신량이 4배 싸진다** ([05 D-2](../05-kvcache-ttft-guide.md)).
2. **불리하게**: 커널이 **쿼리 헤드 단위로 순회**하면 같은 KV 헤드를 `kvMul`=4번
   반복해서 읽는다. 이것이 **발견 #2** 의 절반이다 ([03 §2](../03-attention-batching.md)).

**코드.** 수정본 `multiheadAttBatch_F32` 의 (A) GQA 그룹화 — 같은 KV 헤드를 공유하는
4개 쿼리 헤드를 함께 처리해 KV 읽기를 `1/kvMul` 로.

---

## RMSNorm / φ(·)

**뜻.** 벡터 크기를 정규화하는 연산. 08 §4 수식의 `φ(·)` 가 이것이다.

**코드.** 두 op 로 쪼개져 있다 — `OP_INV_RMS`(`block_norm_pre_*`)가 역제곱근을 계산하고
`OP_RMS_NORM`(`block_norm_*`)이 곱한다 (`llm.cpp:330/336`).

**왜 φ 인가.** 깊이 분해 수식 `K_l = A_l·φ(x_k)` 에서 사영의 입력은 **정규화된** 활성화다.
`x_k` 가 아니라 `φ(x_k)` 를 쓰는 이유는 실제 `block_matmul_k` 의 입력이 그것이기 때문이다.

---

## RoPE

**뜻.** 위치 정보를 **회전**으로 넣는 기법. Q 와 K 에 적용된다.

**왜 중요한가 — 두 가지.**

1. **깊이 분해의 덤프 지점을 정한다.** RoPE 는 위치에 의존하므로, **위치 무관** 선형 사상을
   적합하려면 **RoPE 적용 전** 값을 써야 한다. 그래서 캘리브레이션 덤프를
   `block_matmul_k` 직후 · `block_rope_k` 이전에 뜬다.
   → [코드 맵 §3](01-code-map.md#3-레이어-하나의-연산-순서--05-a-1--08-4-1)
2. **KV 재사용을 어렵게 한다.** 캐시에 들어가는 K 는 RoPE 가 적용된 값이라,
   위치를 바꿔 재사용하려면 RoPE 를 되돌려야 한다. APE 계열 기법이 여기를 건드린다
   ([05 A-1](../05-kvcache-ttft-guide.md)).

**코드.** `llm.cpp:395/403`, 커널 `nn-cpu-ops.cpp:1663`.

---

## SwiGLU / FFN

**뜻.** Llama 계열의 FFN. 행렬이 **3개**다.

```
FFN(x) = w2( silu(w1·x) ⊙ w3·x )
  w1, w3 : 4096 → 14336
  w2     : 14336 → 4096
```

**왜 중요한가 — 이것이 실제 병목이다.**

토큰 1개·레이어 1개당 FLOPs ([05 A-2](../05-kvcache-ttft-guide.md), [00 §1](../00-RESEARCH-PLAN.md)):

| 연산 | FLOPs | 비중 |
|---|---|---|
| K projection (2·h·kvDim) | 8.4 M | 1.9 % |
| V projection | 8.4 M | 1.9 % |
| Q projection (2·h·h) | 33.6 M | 7.7 % |
| O projection | 33.6 M | 7.7 % |
| **FFN SwiGLU (3·2·h·d_ff)** | **352.3 M** | **80.7 %** |
| 합계 | 436 M | 100 % |

실측 시간 비중도 같은 방향이다 (S=1789, 발견 #2 수정 후 — [05 B-1](../05-kvcache-ttft-guide.md)):

| 항목 | 시간 | 비중 | S 스케일링 |
|---|---|---|---|
| **FFN** | 201.3 s | **63.8 %** | O(S·h·d_ff) 선형 |
| attention projection | 47.6 s | 15.1 % | O(S·h²) 선형 |
| attention core | 41.9 s | 13.3 % | **O(S²·h) 이차** |
| lm_head | 18.0 s | 5.7 % | O(S·h·V) 선형 |
| norm / 기타 | ~7 s | 2.2 % | 선형 |

> **어떤 최적화든 FFN 을 건드리지 않으면 상한이 36 % 다.**
> 그리고 FFN 은 이미 sgemm 포화 상태라 커널로는 더 못 짜낸다 →
> **알고리즘 레벨 접근(토큰 분할 / 깊이 분해)이 필요한 이유가 여기 있다** ([03 §8](../03-attention-batching.md)).

**코드.** `llm.cpp:502/508/514` (`block_matmul_w1`, `w3`, `block_act`).

---

## lm_head / 로짓

**뜻.** 마지막 레이어 출력을 어휘 크기(128256)로 사영해 다음 토큰 확률을 만드는 행렬.
`2·h·V = 1.05 GFLOP/token` — **레이어 하나의 2.4배**다.

**왜 중요한가.** prefill 에서는 **완전한 낭비**다. TTFT 에 필요한 로짓은 마지막 토큰 1개뿐인데
모든 위치에 대해 계산하고 있었다. 예비 측정에서 **5.9 %** ([01 §5.5](../01-EXP1-cost-breakdown.md)),
S=1789 에서 **5.7 %** ([05 B-1](../05-kvcache-ttft-guide.md)).

**수정** ([06 §4](../06-baseline-status.md)). 그래프 형태와 sync 는 그대로 두고
**행렬곱의 행 범위만 제한**했다. 결과 32,417 ms (+11 %).

**남은 기회.** 현재는 청크마다 마지막 행 1개를 계산하는데, 그 한 행 때문에
**lm_head 가중치 295 MB 를 청크마다 스트리밍**한다 (청크당 ~32 ms).

| | 연산 절감 | logits sync 절감 |
|---|---|---|
| 행 제한 (현재) | 97 % | ✘ |
| **세그먼트 통째 스킵** | 100 % | ✔ 청크당 ~16 MB (1GbE 에서 0.14초) |

**다중 노드에서 승격할 것.** 단 wave 모드는 `drainPrefillLogits` 가 로짓을 읽으므로 예외 처리 필요.

**코드.** `lmHeadRowRange` (`nn-cpu-ops.cpp:1293`), 사용처 1311·1382.

---

## 토큰 임베딩 (발견 #3)

**뜻.** 토큰 ID → 벡터 룩업 테이블.

**문제.** `llm.cpp:196` 에서 **F32 로 저장**된다.

| | dllama `.m` | GGUF Q4_0 |
|---|---|---|
| 레이어 가중치 | Q4_0 | Q4_0 |
| lm_head | Q4_0 | Q4_0 |
| **토큰 임베딩** | **F32 (2.10 GB)** | Q4_0 (0.30 GB) |
| 합계 | **6.32 GB** | 4.34 GB |

**왜 중요한가 — 성능이 아니라 설계 공간의 문제다.**
임베딩은 행 복사 룩업이라 **prefill 연산 성능에는 영향이 없다**(llama.cpp 와의 tok/s 비교
공정성도 유지된다). 그런데 **CP·블록 병렬은 각 노드가 전체 가중치를 복제해야 한다.**
Pi5 8GB 에 6.32 GB 면 KV 캐시(S=2048 에 512 MB)를 올릴 여유가 없다.
**4.34 GB 로 줄이면 8 GB 노드가 실제로 쓸 수 있게 된다** ([03 §6.5](../03-attention-batching.md), [07 §5](../07-design-block-parallel.md)).

---

## 청크 (chunk) / chunked prefill

**뜻.** 프롬프트를 한 번에 다 넣지 않고 잘라서 순차 처리하는 것. 원조는 SARATHI.

**왜 중요한가.** 청크 크기가 **배치 크기**를 결정하고, 배치 크기가
compute-bound / memory-bound 를 가른다 → [machine balance](#machine-balance-균형점).

**우리 규칙.** 배치 32 이상에서 GEMM 이 포화하므로
**"청크 ≥ 32 토큰"이 하한**이고, 그 위로는 연산 효율 이득이 없다.
→ **청크 크기는 이제 순수하게 통신/파이프라인 관점에서만 정하면 된다.**
설계 공간이 그만큼 좁아진다 ([02 §5(c)](../02-baseline-dotprod-fix.md)).

**코드.** `--prefill-chunk-size`, `--prefill-chunk-threshold` (`dllama.cpp:954~955`).

---

# Part 2. 연산과 하드웨어

## GEMM / GEMV

- **GEMM** (General Matrix-**M**atrix Multiply): 행렬 × 행렬. **prefill** 이 여기.
- **GEMV** (General Matrix-**V**ector Multiply): 행렬 × 벡터. **decode** 가 여기.

GEMM 이 훨씬 효율적이다 — 가중치를 한 번 읽고 여러 토큰을 처리하니까.
**발견 #1 은 정확히 "prefill 이 GEMM 을 못 쓰고 GEMV 를 반복하고 있었다"는 문제였다.**

---

## 산술 강도 (arithmetic intensity)

```
산술 강도 = 연산량 / 메모리 접근량   [FLOP/byte]
```

이 값이 기계의 균형점보다 크면 compute-bound, 작으면 memory-bound.

**우리 맥락의 두 공식** ([05 B-3](../05-kvcache-ttft-guide.md), [03 §5](../03-attention-batching.md)):

| 대상 | 산술 강도 | 균형점(11~14) 도달 조건 |
|---|---|---|
| GEMM (FFN, projection) | ≈ **B** (배치 크기) | B ≥ 12~14 |
| attention | ≈ **T/2** (타일 크기, F32 KV) | T ≈ 22~28 |

두 조건이 거의 같은 자리에 온다. 그래서 하나의 규칙으로 묶인다:

> **Pi5 에서 prefill 청크는 32 이상이어야 GEMM 과 attention 이 동시에 memory-bound 를 벗어난다.**

---

## machine balance (균형점)

**뜻.** 기계의 연산 능력 ÷ 메모리 대역폭. 이 기계가 "byte 하나당 몇 FLOP 을 감당하는가".

**우리 Pi5 (Cortex-A76 ×4) 실측:**

```
연산 ~110 GFLOPS  ÷  메모리 대역폭 ~8–10 GB/s  ≈  11~14 FLOP/byte
```

**왜 중요한가 — 이것이 논문 논거의 뿌리다.**
machine balance 의 **함수**로 표현되므로 **다른 하드웨어로 전이 가능한 규칙**이 된다.
[00 §3](../00-RESEARCH-PLAN.md) 의 "체제 인지 플래너"의 첫 번째 구성요소이자,
[07 §4.5](../07-design-block-parallel.md) 의 설계 제약 "블록 크기 ≥ 32 토큰"의 근거다.

**파생 제약** ([07 §4.5](../07-design-block-parallel.md)):

```
블록 크기 = S / N ≥ 32   →   N ≤ S / 32

S=447 → N ≤ 14   (8노드까지 안전)
S=224 → N ≤ 7
S=112 → N ≤ 3    ← 8노드 클러스터에서 5대가 논다
```

> **프롬프트 길이가 병렬화 가능한 노드 수를 제한한다.** 이 제약이 07 의 하이브리드 2D 분할
> (블록 × TP)과 "GEMM 배치와 블록 경계 분리"를 낳았다.

**관련.** [roofline](#roofline) — machine balance 를 그림으로 그린 것.

---

## roofline

**뜻.** x축 산술 강도, y축 달성 성능의 로그-로그 그래프.
대각선(대역폭 한계)과 수평선(연산 한계)이 만나는 지점이 machine balance.

**왜 중요한가.** [00 §3](../00-RESEARCH-PLAN.md) 의 **phase diagram 은 roofline 의 변형**이다.
[04 Stage 0](../04-reading-path.md) 이 이를 명시한다 — 원논문(Williams et al., CACM 2009)을
정독 대상으로 지정한 이유.

---

## compute-bound / memory-bound

연산 유닛이 노는지, 메모리가 노는지의 문제.

**우리 사례** — 같은 하드웨어에서 두 경로가 정반대였다 ([03 §2](../03-attention-batching.md)):

| 경로 | 실효 성능 | 판정 |
|---|---|---|
| GEMM (int8 dotprod) | **110 GFLOPS** | compute-bound |
| attention (수정 전) | **2.4 GFLOPS** | 심하게 memory-bound |

**왜 그랬나.** 수정 전 attention 은 KV 캐시를 `(batchSize × kvMul)` 번 스트리밍했다.
S=1789 기준 추정 트래픽:

```
쿼리 토큰 1개당 : 32헤드 × 128 × 4B × (K+V) = 32.8 KB × pos
레이어당 Σ      ≈ 32,768 × S²/2 = 52 GB
× 32 레이어     ≈ 1.7 TB
÷ 8 GB/s        ≈ 210 초        (실측 355초 — 같은 자릿수)
```

**교훈.** "attention 이 병목이다"라는 관찰이 **깨진 커널이 만든 인공물**일 수 있다.
→ [H2 재정정](#h2--attention-비지배) 참조. **측정 전에 구현부터 확인해야 한다.**

---

## 양자화 (F32 / F16 / Q8_0 / Q4_0)

| 표기 | 비트 | 용도 |
|---|---|---|
| F32 | 32 | 원본 정밀도. 우리 KV 캐시·임베딩 |
| F16 | 16 | 절반. llama.cpp 의 KV 기본값 |
| **Q8_0** | 8 (+블록당 스케일) | **활성화** |
| **Q4_0** | 4 (+블록당 스케일) | **가중치** |

`Q4_0` 은 값 32개를 한 블록으로 묶어 4비트 정수 + fp16 스케일 하나로 저장한다
(`NnBlockQ40`, 18 B/블록). 8B 모델이 4.5 GB 로 줄어든다.

우리 실행 조합은 **Q4_0 가중치 × Q8_0 활성화**다. 이 조합이 int8 dotprod 경로를 탄다.

---

## dotprod (SDOT) — 발견 #1

**뜻.** ARM v8.2 의 명령어. int8 4개씩의 내적을 한 번에 계산한다(`vdotq_s32`).

**무슨 일이 있었나** ([02](../02-baseline-dotprod-fix.md)):

```
1. Makefile 이 aarch64 에서 -march=native -mtune=native 를 함께 지정
2. aarch64 에서 -mtune=native 를 붙이면 -march=native 가 확장한 기능셋이
   기본 armv8-a 로 되돌아가 __ARM_FEATURE_DOTPROD 가 사라진다
3. sgemm.cpp:978 의 #elif defined(__ARM_FEATURE_DOTPROD) 분기가 컴파일에서 제외
4. llamafile_sgemm 이 Q40×Q80 조합에 항상 false 반환
5. matmulForward_Q80_Q40_F32 가 폴백 진입 → 배치를 토큰 하나씩 matvec 으로 처리
6. prefill 이 배치 이득을 전혀 못 받음
```

| 플래그 | `__ARM_FEATURE_DOTPROD` |
|---|---|
| `-march=native` | 정의됨 |
| `-march=native -mtune=native` | **사라짐** |
| `-mcpu=native` | 정의됨 ← 수정 |

**증상.** prefill 2.01 tok/s vs decode 1.90 tok/s — **배치 이득이 0**.
prefill 은 decode 보다 몇 배 빨라야 정상이다.

**수정 후 마이크로벤치** (`bench_sgemm.cpp`, 4스레드, GFLOPS):

| shape | b=1 | b=8 | b=32 | b=128 |
|---|---|---|---|---|
| qkv/o (4096×4096) | 31.9 | 72.7 | **105.4** | 109.6 |
| ffn_w1 (14336×4096) | 33.7 | 90.5 | **106.3** | 109.5 |
| ffn_w2 (4096×14336) | 26.4 | 85.3 | **104.6** | 101.9 |

수정 전에는 위 표의 **모든 칸이 `FALLBACK`** 이었다.
바이너리 크기 459,312 → 527,088 B (경로가 실제로 컴파일된 증거).

**연구에 갖는 의미** — 세 갈래다:

1. **baseline 신뢰성.** 이 트리의 기존 prefill 수치가 전부 3~4배 느린 상태에서 나왔다.
2. **비용 모델 갱신.** 00 §1 의 가정 30 GFLOPS → 실측 **110**. 3.7배 차이.
   연산이 3.7배 빨라지면 통신/연산 비율이 그만큼 나빠진다 → **H1 에 유리해진다.**
3. **설계 공간 단순화.** 배치 32 포화 → 청크 크기를 통신 관점으로만 정하면 된다.

**대조 근거.** llama.cpp 는 `-mcpu=cortex-a76+crc+crypto` 를 쓰고 dotprod 지원을
**컴파일 타임에 명시적으로 테스트**한다(`GGML_MACHINE_SUPPORTS_dotprod`).
발견 #1 이 distributed-llama 고유 문제라는 직접 근거 ([03 §6.6](../03-attention-batching.md)).

---

## attention 배치화 — 발견 #2

**문제.** `multiheadAtt_F32` 의 시그니처가 `pos` 를 **하나**만 받는다.
**쿼리 위치 1개를 전제로 설계된 decode 용 커널**인데, prefill 이 이것을
`for (batchIndex)` 루프로 배치 크기만큼 반복 호출했다.
게다가 커널 내부가 쿼리 헤드 단위로 순회해 같은 KV 헤드를 `kvMul`=4번 읽는다.

→ KV 캐시가 **(batchSize × kvMul)번** 스트리밍된다.

**수정** — `multiheadAttBatch_F32` 신설. 연산 순서를 유지하므로 수치 결과는 동일해야 한다.

| | 기법 | 효과 |
|---|---|---|
| (A) | **GQA 그룹화** — 같은 KV 헤드를 공유하는 kvMul 개 쿼리 헤드를 함께 처리 | KV 읽기 1/kvMul |
| (B) | **쿼리 타일링** — t 루프를 바깥으로 빼 배치 전체가 KV 한 번 읽기를 공유 | KV 읽기 1/batchSize |

스레드 분할은 KV 그룹 수가 스레드 수 이상일 때만 그룹 단위(=A 적용), 아니면 헤드 단위(=B 만).
TP 로 헤드가 쪼개진 구성에서 스레드가 놀지 않게 하기 위함이다.
`batchSize == 1`(decode)은 **기존 경로 그대로** — PiPP 의 기존 측정치는 영향받지 않는다.

**결과** ([03 §7](../03-attention-batching.md)):

| | S=447 | S=1789 |
|---|---|---|
| attention | 10.4 s → **2.8 s (3.65×)** | 355.6 s → **41.9 s (8.49×)** |
| prefill 전체 | 78.2 → 72.9 s | 624.2 → **315.7 s (1.98×)** |
| ms/tok | 175 → 163 | 349 → **176** |
| attn 비중 | 13.3 % → 3.9 % | 57.0 % → 13.3 % |

정확성: seed 42 / temp 0 에서 출력 **byte-for-byte 동일**(md5 일치).

> **가장 중요한 신호**: ms/tok 이 S=447(163)과 S=1789(176)에서 거의 평평해졌다.
> **prefill 비용이 S 에 대해 다시 선형에 가까워졌다**는 뜻이고, 이것이 정상 동작이다.

**코드.** `nn-cpu-ops.cpp:1695` `multiHeadAttForward_F32_F32` 안의 `if (batchSize > 1u)` 분기.

---

## repack — 발견 #4

**뜻.** 가중치를 **SIMD 가 먹기 좋은 순서로 미리 재배치**하는 것.
llama.cpp 는 로드 시점에 Q4_0 을 이렇게 바꾼다(Cortex-A76 은 `q4_0_4x4_q8_0` 경로).
distributed-llama 가 벤더링한 llamafile sgemm 에는 이 개념이 없었다.

**기존 커널의 명령어 예산** ((i,j,l) 조합마다 — [06 §3](../06-baseline-status.md)):

| 용도 | 명령어 |
|---|---|
| 4비트 언팩 (load/and/shr/sub) | ~6 |
| 실제 연산 (vdotq_s32) | 2 |
| 누적 (cvt, mla) | 2 |
| 스케일 (fp16 스칼라 곱) | ~1 |

유효 64 FLOP / 명령어 11개 ≈ 5.8 FLOP/instr → 이론 111 GFLOPS.
**실측 97~110 GFLOPS 와 정확히 일치한다.** 즉 **명령어의 절반 이상이 언팩이었다.**

**repack 이 바꾸는 것 3가지:**

1. **zero-point(−8)를 pack 시점에 XOR(`0x88888888`)로 흡수** → 런타임 `vsubq_s8` 제거
2. **출력 행 4개를 인터리브** → 로드 1회가 4행을 커버 (`vdotq_laneq_s32`)
3. 스케일 `d[4]` 가 연속 → `vmulq_laneq_f32` 로 벡터화

**마이크로벤치** (`bench_repack.cpp`, 4스레드, batch 32, GFLOPS):

| shape | 기존 | repack | 배율 |
|---|---|---|---|
| qkv/o (4096×4096) | 105.2 | 317.6 | 3.02× |
| ffn_w1 (14336×4096) | 106.2 | 307.0 | 2.89× |
| ffn_w2 (4096×14336) | 103.2 | 331.3 | **3.21×** |

정확성 (`test_repack.cpp`): gemm/gemv 모두 L2 상대오차 ~1e-6
(양자화 차이가 아니라 fp32 누적 순서 차이 수준).

**통합 설계 판단 3가지** — 여기가 이 항목의 실질이다:

1. **in-place repack.** `NnBlockQ40 4개(72 B) == block_q4_0x4(72 B)` 라 추가 메모리 0.
   `static_assert` 로 강제한다(`nn-repack.hpp:39`). 행 그룹 g 의 원본/목적지 바이트 범위가
   정확히 겹치므로 그룹 하나 분량만 임시로 잡으면 된다.
   > 초기 구현은 op 마다 가중치 전체 사본을 잡았다 — lm_head 는 295 MB 로 Pi5 8GB 에서 위험.
2. **로드 완료 감지.** `NnCpuDeviceSegment::loadWeight` 에서 `loadedBytes` 누적.
   root/worker 가 모두 이 경로를 지나므로 훅이 한 곳이면 된다 (`nn-cpu.cpp:224~268`).
3. **배리어 없는 스레드 분할.** 실행기는 op 경계에서만 동기화한다.
   출력 **열**로 나누고 각 스레드가 활성화 변환을 자기 `thread_local` 스크래치에 중복 수행.
   - 중복 비용: batch 32 / k 4096 기준 스레드당 ~139 kB 셔플
   - **행**으로 나누면 배리어는 불필요하지만 모든 스레드가 가중치 전체(FFN w1 28 MB)를 읽어
     트래픽이 nThreads 배. 연산 11.4 ms vs 트래픽 14 ms 로 **역전되어 기각**.

**폴백.** repack 미지원 형상(d 가 4의 배수 아님)이나 비-aarch64 는
기존 `matmulForward_llamafile` 로 자동 폴백. decode(batch=1)는 gemv 가 평범한
`NnBlockQ80` 을 그대로 받으므로 활성화 변환 없이 동작한다.

**남은 갭.** 마이크로벤치 3.1× 인데 실측 GEMM 은 2.2× 다. 후보 두 개 —
활성화 변환의 스레드별 중복 / GEMM 이 3배 빨라지며 드러난 메모리 대역폭 한계.

---

## online softmax / 타일링

**online softmax.** softmax 를 **한 번의 순회로** 계산하는 기법. 최대값과 누적합을 함께 갱신한다.
FlashAttention 의 두 축(타일링 + online softmax) 중 하나.

**우리 맥락.** 현재 `multiheadAttBatch_F32` 는 `att` 중간 버퍼를 메모리에 **실체화**한다:

```
batchSize 32 × nHeads 32 × seqLen 2048 × 4 B = 8.4 MB / 청크
쓰고 → 읽고(softmax) → 다시 읽는다(AV)
```

llama.cpp 는 `ggml_flash_attn_ext` 로 융합한다. **추가 2~3× 여지** ([03 §8](../03-attention-batching.md)).

**남은 갭의 전모** — attention 41.9 s 는 이론 하한(110 GFLOPS 기준 7.6 s)의 약 5.5배:

1. `att` 중간 버퍼 실체화 → 2~3× 여지
2. KV 캐시 F32 (llama.cpp 는 F16) → 트래픽 2배
3. attention 이 fp32 NEON(피크 76.8 GFLOPS)인데 GEMM 은 int8 dotprod(110)

> 단 attention 이 이미 13.3 % 이므로 2.5× 더 개선해도 prefill 전체로는 8 % 남짓이다.
> **FFN 63.8 % 가 더 큰 레버다.**

---

# Part 3. 분산

## 분산 축 네 가지

| 축 | 무엇을 쪼개나 | 레이어당 통신 | KV 가 남는 모양 | decode 인계 |
|---|---|---|---|---|
| **TP** (텐서) | 헤드/hidden 차원 | all-reduce ×2, `2·B·S·h` | `kvDim/N` × **전체 위치** | 같은 TP 면 그대로 |
| **PP** (파이프라인) | 레이어 | P2P ×1, `B·S·h` | **자기 레이어** × 전체 위치 | 같은 PP 면 그대로 |
| **CP/SP** (시퀀스) | 토큰 위치 | ring KV 교환, `B·S/N·kvDim` | **자기 위치 구간** × 전체 레이어 | partial attention + LSE reduce 필요 |
| **블록 병렬** (근사) | 토큰 위치, 블록 간 attention 생략 | **0** (인코딩 중) | CP 와 동일 | 단일 merge |

**통신량 비교** (S=512, N=4, F32, 추정 — [05 D-2](../05-kvcache-ttft-guide.md)):

| 축 | prefill 총 통신량 |
|---|---|
| TP | ~768 MB |
| CP (ring, GQA) | ~100 MB |
| PP | ~25 MB |
| 블록 병렬 | ~1 MB |

**우리 실측** (447토큰, llama3-8b_q40, Pi5, 1GbE — [07 §1](../07-design-block-parallel.md)):

| 구성 | prefill | syncWait | syncXfer | gemm | 단일 대비 |
|---|---|---|---|---|---|
| 단일 노드 | 37,939 ms | — | — | **33,079** | 1.00× |
| **TP 2노드** | **33,633** | 14,611 | 1,965 | **13,942** | **1.13×** |
| PP 4노드+wave | 39,825 | 23,102 | 829 | — | 0.95× |
| SP 2노드 | 43,620 | 1,785 | 537 | **36,502** | 0.87× |
| SP 4노드 | 47,374 | 8,178 | 1,687 | **32,766** | 0.80× |

**여기서 읽어야 할 세 가지** — 07 전체의 근거다:

**(a) 연산이 87 % 이고, 나누면 정확히 나뉜다.** gemm 33.1초는 prefill 37.9초의 87 %.
TP2 에서 13.9초로 떨어졌다(2.37×, 2× 초과분은 캐시 효과).
→ **연산 분할은 잘 된다. 남는 문제는 동기뿐이다.**

**(b) 깊이 축(PP)의 대기는 구조적이라 튜닝으로 안 줄어든다.** 발견 #5 의 poll 수정으로
TP2 는 42 % 빨라졌는데 PP4 는 3 % 뿐. PP4 는 전송량이 TP2 의 절반 이하인데 대기는
더 크다(23.1 vs 14.6초). **청크가 스테이지를 순서대로 통과해야 하기 때문이다.**

**(c) dllama 의 `--sp-size` 는 토큰 축 병렬이 아니다.** ↓

---

## SP 의 두 가지 뜻 (필독)

| 표기 | 뜻 |
|---|---|
| SP (문헌) | Sequence Parallel — 토큰을 나눠 **연산도** 나눔 |
| `--sp-size` (dllama) | KV **저장**만 시퀀스 방향으로 쪼갬. 연산은 전 노드가 중복 |

`sliceKvCache()` 의 `localSeqStart`/`localSeqLen` 은 KV 버퍼를 쪼갤 뿐,
모든 노드가 같은 토큰을 계산한다. **측정이 이를 확인한다** —
SP2/SP4 의 gemm 이 단일 노드와 같다(36.5 / 32.8 vs 33.1초).

> **메모리 최적화이지 병렬화가 아니다.**
> prefill 에서 본질적으로 병렬인 축은 토큰 축인데(레이어 안의 모든 토큰은 서로 독립),
> **그 축으로 연산을 나누는 기능이 없다.** 이것이 07 이 만들려는 것이다.

또한 현재 `SYNC_SP_KV` 는 **allgather** 다 — KV 를 전 노드에 복제한다.
ring(pass-KV)으로 바꾸면 통신량이 크게 준다. **exact CP 구현의 출발점이 여기다**
([05 D-3](../05-kvcache-ttft-guide.md)) — 신규 구현이 아니라 교체다.

---

## barrier (배리어)

**뜻.** 여러 노드가 **서로를 기다리는 지점**. 가장 느린 노드에 전부 맞춰진다.

**얼마나 자주인가.** TP 는 레이어마다 2번 — `llm.cpp:454`(O-proj 뒤)와 `605`(FFN 뒤).
32레이어면 **청크당 64번**. S=447 을 청크로 나누면 실행당 수백 회다
(TP2 실측: **896 배리어**).

**비용.** 1회 **16.3 ms** = syncWait 14,611 ms ÷ 896 ([07 §4](../07-design-block-parallel.md)).

**왜 하필 O-proj 뒤와 FFN 뒤인가.** 그 두 지점이 TP 로 쪼갠 조각을 합쳐야 하는 곳이다.
[00 §1](../00-RESEARCH-PLAN.md) 이 지적하듯, KV 생성의 본질적 연산은 3.9 % 인데
**나머지 96 % 가 정확히 all-reduce 를 유발하는 부분**이다.

**코드.** 호출 `nn-executor.cpp:277`, 실체 `nn-network.cpp:1867`
`NnNetworkNodeSynchronizer::sync`, 단일 노드 무동작판 `nn-executor.cpp:55`.

---

## 발견 #5 — 배리어의 1 ms sleep

> **주의: 이 발견을 서술한 연구 문서가 없다.** 07 이 `06 §7` 로 링크하지만 06 §7 은
> 다른 버그(트래픽 요약 무한 출력)다. 유일한 근거는 `nn-network.cpp:95~105` 주석이다.

**문제.** 논블로킹 소켓이 `EAGAIN` 을 반환했을 때 **무조건 1 ms 를 잤다.**
리눅스 타이머 해상도상 실제로는 1~2 ms 이고, 하나의 집합통신에서 재시도가 수십 회
발생하므로 배리어 1회가 수십 ms 로 부풀었다.

```
실측 (TP 2노드, S=447) : 배리어 896회에 syncWait 39.3초 = 배리어당 43.9 ms
  그중 회선 시간         :  6.9 ms
  나머지                 : 37 ms  ← 전부 이 sleep
```

**수정.** `poll()` 로 교체. 데이터가 준비되면 마이크로초 단위로 깨어나므로 sleep 해상도에
묶이지 않는다. 동시에 **스핀이 아니라 커널 대기라 CPU 코어를 뺏지 않는다** —
코어가 4개뿐인 SBC 에서는 이 점이 중요하다(→ [H5](#h5--overlap-이-zero-sum)).

**결과.** 배리어 43.9 → **16.3 ms**. TP2 가 42 % 개선, **PP4 는 3 % 뿐**
→ [07 §1(b)](../07-design-block-parallel.md) 의 "구조적 대기" 논거.

---

## all-reduce / allgather / ring / P2P

| 방식 | 동작 | 배리어인가 |
|---|---|---|
| **all-reduce** | 모두가 값을 내고 합쳐서 모두가 받음 | **예** (전원 참여) |
| **allgather** | 각자 조각을 내고 모두가 전체를 가짐 | **예** |
| **P2P** | 한 노드가 다른 한 노드에게만 | 아니오 |
| **ring** | 원형으로 이웃에게만 넘김. P2P 의 연쇄 | 아니오 — 연산과 겹칠 수 있음 |

**우리 측정의 핵심 결론:**

```
TP2 : syncWait 14.6초  vs  syncXfer 2.0초   →  7:1
```

> **통신 시간의 대부분은 전송이 아니라 대기였다.**
> 즉 **바이트를 줄이는 것보다 배리어 수를 줄이는 것이 중요하다.**
> 이것이 GPU 문헌의 최적화 방향(통신량 압축)과 갈리는 지점이다.

**계측 정의** (이걸 모르면 숫자 해석이 틀린다 — `nn-network.cpp:877`, [01 §1](../01-EXP1-cost-breakdown.md)):

- `readMany`/`writeMany` 의 busy-poll 루프에서
- **바이트가 실제로 움직인 패스** → `syncXferUs`
- **진행이 없던 패스와 재시도 대기** → `syncWaitUs`

> 실행기 단계(`STEP_SYNC_NODES`)를 쪼개는 대신 **소켓 계층에서** 나눴다.
> 실제 대기가 발생하는 지점이 거기이고, 스텝 타입을 늘리면 `N_STEP_TYPES` 의존 코드가
> 전부 흔들리기 때문이다.

---

## straggler (스트래글러)

**뜻.** 집합 통신에서 **가장 느린 노드**. 배리어는 전원을 여기 맞춘다.

**우리 계측에서** `syncWait` 가 곧 straggler 비용이다.

**원인 후보 구분** ([01 §4](../01-EXP1-cost-breakdown.md)):
`monitor_node.sh` 가 수집한 **온도/주파수**와 노드별 `T_gemm` 을 대조해
열 스로틀링인지 OS 스케줄링 jitter 인지 가른다.
노드별 `T_gemm` 편차가 15 % 를 넘으면 **이종/열 인지 스케줄링이 별도 기여점으로 성립**한다.

**측정 위생 경고** ([06 §5](../06-baseline-status.md)):
워커에 구버전 바이너리가 남으면 워커가 6.9× 느려 **전부 straggler** 가 되고
`wait_frac` 이 오염된다. 다중 노드 측정 전 root/worker `md5sum` 대조 필수.

---

## wave / drainPrefillLogits

**뜻.** `--wave-pipeline 1`. 청크를 파이프라인에 **비동기로** 흘려보내 PP 버블을 줄인다.

**결과.** PP4 가 73.7 → 41.0초로 개선됐지만 **여전히 단일 노드보다 느렸다**
([08](../08-glossary.md), [07 §1](../07-design-block-parallel.md)).

**부작용 두 개:**

1. `forward()` 가 비동기로 끝나지 않으므로 **op 프로파일링 집계에서 제외된다**
   (`dllama.cpp:477`). wave 로 측정한 로그에는 gemm/attn 분해가 없다.
2. 마지막에 밀린 로짓을 걷어내는 `drainPrefillLogits`(`app.cpp:1035`)가
   **로짓을 실제로 읽는 유일한 경로**라, lm_head 세그먼트 스킵 최적화의 예외가 된다.

---

## back-pressure

수신 측이 아직 읽지 않아 송신이 막히는 상태. `writeMany` 는 이 시간을
`xfer` 가 아니라 **`wait` 로 분류한다**(`nn-network.cpp:877`).
발견 #5 의 sleep 이 여기에 얹혀 있었다.

---

## anchor (앵커)

**뜻.** 블록 병렬에서 **모든 노드가 공통으로 보는 앞부분 토큰 구간**.
블록을 독립 인코딩하면 블록 간 attention 이 사라져 정확도가 떨어지는데,
프롬프트 맨 앞 A 토큰을 모든 블록의 접두로 붙여 그 손실을 완화한다.
[Star Attention](https://arxiv.org/pdf/2411.17116) 의 핵심 장치.

**우리 쟁점 — 재계산 vs 브로드캐스트** ([07 §4](../07-design-block-parallel.md)):

Star 는 anchor 를 **모든 노드가 중복 계산**한다. 노드당 처리 토큰이 `S/N + A` 가 되고,
`A = S/N` 이면 연산이 2배 → speedup 이 N → N/2 로 반토막.

우리 클러스터 실측값으로 두 선택지를 비교하면 (N=4, S=447, L=32):

| 방식 | 비용 |
|---|---|
| anchor 재계산 | 노드당 **+9.5초** (연산 2배) |
| **anchor KV 브로드캐스트** | 전송 32 MB ≈ 0.3초 + 배리어 32회 × 16.3 ms ≈ 0.8초 = **1.1초** |

**약 12배 차이.** 이것이 07 의 고유 기여 주장이다.

**의존성 처리.** 노드 0 이 레이어 l 의 anchor KV 를 만들어야 나머지가 레이어 l 을 계산할 수 있다
→ **노드 0 을 한 레이어 앞서 달리게 한다.** 스큐는 1 레이어(전체의 1/L = 3 %)뿐이고,
전송이 단방향이라 연산 뒤로 숨길 수 있다. anchor 는 어차피 블록 0 이므로
노드 0 이 자기 일을 하면서 부산물로 만든다.

> 이 논거는 이전 초안에서 "GPU 대비 반전"으로 서술했다가 **근거 부족으로 철회**했다.
> 지금은 우리 하드웨어의 측정값만으로 성립한다 — GPU 와 비교할 필요가 없다.
> ([refs.md 인용 주의 §1](../refs.md) 참조)

---

## anchor amortization

**뜻.** Star Attention 의 **숨은 전제**. 호스트 하나가 블록을 **여러 개** 맡으면
anchor 를 한 번 계산해 여러 블록에 재사용할 수 있어 오버헤드가 희석된다.

**S=128K 라서 성립한다.** 우리 체제(S ≤ 4K, 노드당 블록 1개)에서는
희석할 대상이 없어 anchor 비용이 그대로 **2×** 로 남는다.

**왜 중요한가.** 이것이 가설 **H4** 이고, [refs.md](../refs.md) 가 명시하듯
**살아남는 논거가 이것뿐**이다:

> "GPU 는 재계산이 싸고 SBC 는 통신이 싸다"는 단순 반전 주장은 **성립하지 않는다.**
> 재계산 vs 브로드캐스트 비용비는 Pi5+1GbE ≈ 200, A100+NVLink ≈ 110 으로
> **양쪽 다 브로드캐스트가 유리**하다. Star 가 재계산을 택한 이유는 비용이 아니라
> **의존성 제거**(완전 독립 실행)다.

---

## LSE (log-sum-exp) merge / 분산 softmax

**뜻.** 각 노드가 자기 KV 구간에 대해서만 attention 을 계산한 뒤,
부분 결과를 **수학적으로 정확하게** 합치는 방법.

```
노드 i : partial output O_i,  그 구간의 log-sum-exp  L_i
병합   : softmax 분모를 L_i 들로 재정규화 → 전체 attention 과 동일
```

교환량은 레이어당 헤드당 `(head_dim + 1)` float — 무시할 수준이다.
**근사가 아니라 exact 다.**

**왜 중요한가.** [07 §2](../07-design-block-parallel.md) 의 "KV 를 통합할 필요가 없다"의 근거.

| decode 구성 | KV 통합 비용 |
|---|---|
| **토큰 축 (prefill 과 동일)** | **0** |
| 한 노드로 수집 | S=447 → 117 MB ≈ 1초 / S=2048 → 537 MB ≈ 4.8초 |
| TP (현행 dllama) | 헤드 축으로 재분배 → all-to-all, 가장 비쌈 |

> **설계 제약: prefill 축 = decode 축.**
> 현행 dllama decode 는 TP 이므로 최종적으로는 decode 도 토큰 축으로 바꿔야 한다(Phase C).
> v1(Phase A)에서는 수집으로 우회한다.

근거: [vLLM CP RFC](https://github.com/vllm-project/vllm/issues/26133),
[Meta CP](https://arxiv.org/pdf/2411.01783) 의 pass-Q, [Tree Attention](https://arxiv.org/html/2408.04093v2).

---

## KV affinity / KV migration

`--strict-kv-affinity`, `--allow-kv-migration` 플래그.

**왜 있나.** [05 D-1](../05-kvcache-ttft-guide.md) 이 지적하듯,
분산 축을 고르는 것은 **KV 캐시가 어디에 어떤 모양으로 남을지를 고르는 것**이다.
decode 는 마지막 토큰 1개가 **전체 KV** 를 봐야 한다.

3번(decode 인계 비용)을 빼먹으면 **"prefill 은 빨라졌는데 KV 옮기느라 TTFT 는 그대로"** 가 된다.
**이 플래그들이 존재한다는 것 자체가 그 함정을 이미 겪었다는 증거다.**

---

# Part 4. 우리 알고리즘의 수학

> **주의**: 이 Part 는 [07 의 블록 병렬](#블록-병렬-요약)과 **별개의 아이디어**다.
> 07 은 **토큰**을 나누고, 여기는 **깊이 사슬**을 줄인다. 둘은 곱해진다.

## 4-1. 문제 — 깊이는 순차다

```
x_{l+1} = x_l + Attn_l(x_l) + FFN_l(x_l)
```

레이어 l+1 을 계산하려면 레이어 l 이 끝나야 한다. **32번 순차.**
토큰끼리는 독립이라 병렬이 되지만, **깊이는 안 된다.** 이것이 모든 문제의 뿌리다.

## 4-2. 관찰 — KV 는 단순한 곱셈이다

```
K_l = W_k^l · φ(x_l)        ← 행렬 하나 곱하는 것이 전부
```

레이어 비용의 **3.9 %** (K 1.9 % + V 1.9 %). 나머지 96 % (Q/O projection, FFN)는
**오직 x_l 을 만들기 위해서만** 존재한다.

## 4-3. 아이디어 — 중간 지점에서 끊는다

레이어 k 까지만 정상 계산하고, 그 위 레이어들의 KV 는 **x_k 에서 직접** 만든다.

```
K_l = A_l · φ(x_k)          (l > k)
```

`A_l` 은 "x_k 를 보고 레이어 l 의 K 를 맞추는 행렬".

**효과.** 레이어 k+1..31 이 **서로 독립**이 되어, 노드에 아무렇게나 나눠줘도 통신이 0이다.
깊이 사슬이 32 → k 로 줄어든다.

### 선형 사상 (linear map)

`A_l · φ(x_k)` 처럼 **행렬 곱 하나로 표현되는 변환**. 가장 단순하고, 그래서 빠르다.
실제 관계가 비선형이면 오차가 생긴다 — **그게 우리가 검증하는 것**이다.

### 최소제곱 적합 (least squares fit)

`A_l` 을 어떻게 정하나? **실제 값과 가장 가깝게** 맞춘다.

```
A_l* = argmin_A  ‖ A·φ(x_k) − K_l ‖²
```

읽는 법: "A 를 곱한 결과와 진짜 K_l 의 차이(제곱합)를 최소로 만드는 A".

> **학습(training)이 아니다.** 연립방정식을 푸는 것이라 **닫힌 형태 해**가 있고,
> 몇 분이면 끝난다. GPU 도 역전파도 필요 없다.
> → SwiftKV 는 distillation(증류 학습)이 필요해 GPU 가 있어야 한다. **우리의 차별점.**

**코드.** `prefill_bench/fit_kv_projection.py` 의 `ls_fit(X, Y, ridge)` — 6줄이 전부다.

```python
def ls_fit(X, Y, ridge=1e-3):
    """min_A ||X A - Y||_F  (X: n x h, Y: n x d)  ->  A: h x d"""
    XtX = X.T @ X
    XtX[np.diag_indices(h)] += ridge * np.trace(XtX) / h   # ← ridge
    return np.linalg.solve(XtX, X.T @ Y)
```

## 4-4. dense / 랭크 / 공유 기저

**08 에서 가장 헷갈리는 부분이다. 천천히 간다.**

### dense (조밀) 적합

`A_l` 을 **아무 제약 없는 꽉 찬 행렬**로 두는 것.

```
A_l : 4096 × 1024  =  4.2 M 개의 숫자
상위 16레이어의 K,V 를 모두 → 134 M 개
Q4_0 으로 저장해도 75 MB
```

가장 정확하다. 그런데 **CPU 에서는 이 75 MB 를 매번 메모리에서 읽어야 한다.**
Part 2 에서 봤듯 SBC 는 메모리가 병목이라 **치명적**이다.

### 랭크 (rank)

행렬이 담고 있는 **실질적인 정보의 차원 수**.

```
4096×1024 행렬이라도, 실제로는 256차원 정보만 담고 있을 수 있다
→ "랭크가 256이다"
```

랭크가 낮으면 **작은 행렬 두 개의 곱으로 쪼갤 수 있다**:

```
A (4096×1024)  ≈  P (4096×256) · B (256×1024)
   4.2 M 개              1.05 M + 0.26 M = 1.3 M 개   → 3.2배 절약
```

이것을 **저랭크 근사(low-rank approximation)** 라 한다.

### SVD / 특이값 / 에너지

랭크를 찾는 표준 도구.

**SVD**(특이값 분해)는 행렬을 "방향들 + 각 방향의 세기"로 분해한다.
그 세기가 **특이값(singular value)** 이다.

```
특이값이 큰 순서로: σ₁ ≥ σ₂ ≥ σ₃ ≥ ...

에너지 비율 = (상위 r개 특이값의 제곱합) / (전체 제곱합)
```

에너지가 0.95 라면 "상위 r개 방향이 정보의 95 %를 담는다"는 뜻이고,
나머지를 버려도 5 % 만 잃는다. **랭크를 얼마로 할지 정하는 기준**이 된다.

### 공유 기저 (shared basis) ← 우리 아이디어

저랭크를 **레이어별로 따로** 하면 여전히 레이어마다 `P_l`, `B_l` 두 행렬이 필요하다.

> **핵심 가설**: 상위 레이어들의 KV 는 **같은 부분공간**에 산다.

그렇다면 기저를 레이어끼리 공유하고, 레이어별 차이는 **대각 스케일링**만 두면 된다.

```
A_l  ≈  U · diag(d_l) · V

  V      : 공유 (256 × 4096)   — x_k 를 256차원으로 압축
  d_l    : 레이어별 (256개)     — 레이어마다 다른 가중치
  U      : 공유 (1024 × 256)   — 256차원을 K 로 복원
```

**계산 순서가 바뀐다** — 이것이 이득의 원천이다:

```
z   = V·φ(x_k)            ← 한 번만 (모든 레이어 공통)
K_l = U·(d_l ⊙ z)         ← 레이어마다 원소곱 + 작은 행렬 곱
```

`⊙` 는 원소별 곱(element-wise product). 벡터 두 개를 같은 자리끼리 곱하는 것.

**절감 효과** (상위 16레이어, K+V):

| | 파라미터 | 스트리밍량 | 연산 |
|---|---|---|---|
| dense | 134 M | 75 MB | 268 MFLOP/token |
| **공유 기저** | **1.58 M** | **0.9 MB** | **18.9 MFLOP/token** |
| 절감 | **85×** | **83×** | **14×** |

> **CPU 에서는 83× 메모리 절감이 결정적이다.**
> GPU 연구가 이 구조를 안 쓰는 이유는 GPU 에서는 메모리 대역폭이 넉넉해
> dense 로도 충분하기 때문이다. **우리 체제에서만 필요한 설계다.**

**가설의 방증.** KVSharer / CommonKV 가 "레이어 간 KV 가 서로 닮았다"를
training-free 로 이용한다 → 공유 기저 가설과 같은 방향.

**코드.** `fit_kv_projection.py --k 16 --rank 256`.
캘리브레이션 덤프는 `DLLAMA_CALIB_DIR` 환경변수 (`nn-cpu-ops.cpp:1425~1469`),
`block_matmul_k`/`v` 직후·RoPE 이전.

---

## 블록 병렬 (요약)

Part 4 와 짝을 이루는 **다른 축**의 아이디어. 상세는 [07](../07-design-block-parallel.md).

```
입력 S 토큰을 N 블록으로 분할, 노드당 1블록

각 레이어 l 에서:
  노드 0   : 자기 블록 계산 → layer-l anchor KV 를 브로드캐스트
  노드 i>0 : [anchor KV ⊕ 자기 블록] 에만 attention
             그 외 노드 간 통신 없음

종료 후: KV 는 토큰 구간별로 샤딩된 채 유지, 쿼리만 LSE merge
```

**연산량:**

| 항목 | 단일 노드 | 블록 병렬 (노드당) |
|---|---|---|
| gemm | O(S·h²) | **O(S/N·h²)** |
| attention | O(S²·h) | **O((S/N + A)²·h)** |
| 통신 | 0 | anchor KV 브로드캐스트만 |

**attention 은 N 배보다 더 줄어든다** — O(S²) 이므로 총 연산량 자체가 감소한다:

| | 단일 447토큰 | 4블록 × 112토큰 |
|---|---|---|
| attention 총합 | 2,770 ms | 4 × ~220 = **880 ms** (3.1× 감소) |

**TP/PP 는 attention 연산량이 그대로다. 블록 병렬만 이 이득이 있다.**

**선형성 근거** ([07 §1(d)](../07-design-block-parallel.md)) — 토큰을 N 등분하면 시간도 1/N 이 된다는 직접 증거:

| 토큰 | prefill | gemm | attn | ms/tok |
|---|---|---|---|---|
| 96 | 7,224 ms | 6,666 | 161 | **75.2** |
| 193 | 15,462 | 13,765 | 887 | 80.1 |
| 447 | 36,740 | 32,298 | 2,770 | **82.2** |

토큰이 4.6배 늘어도 ms/tok 은 9 % 만 증가. 그 증가분이 O(S²) attention 항이다.
96 토큰에서도 75.2 ms/tok 이 유지되므로 **블록이 작아져도 고정 오버헤드는 크지 않다.**

**이론 상한:**

```
TTFT_ideal ≈ gemm/N + attention + merge
N=4 →  33.1/4 + α ≈ 9.5초   (3.99×)
N=8 →  33.1/8 + α ≈ 5.1초   (7.4×)
```

**해소책 (블록이 32 토큰 미만이 될 때):**

1. **하이브리드 2D 분할** — `N = N_block × N_tp` 로 인수분해. 남는 노드를 TP 에 쓴다.
   2노드 그룹 안의 all-reduce 는 전체 all-reduce 보다 훨씬 싸다(참여 노드도 적고 데이터도 1/N_block).
2. **GEMM 배치와 블록 경계를 분리** — FFN·projection 은 토큰 간 의존이 전혀 없다.
   블록 경계는 **오직 attention 에만** 의미가 있다.
   → **"노드당 토큰 수 ≥ 32"** 만 지키면 되고 "블록당 토큰 수"는 자유로워진다.
   이미 발견 #2 에서 attention 을 배치 커널로 분리해 뒀으므로 **블록별 마스크만 추가하면 된다.**

---

# Part 5. 검증과 정확도

## 재구성 오차 (relative error)

`A_l·φ(x_k)` 가 진짜 `K_l` 과 얼마나 다른가.

```
rel.err = ‖ 진짜 − 예측 ‖ / ‖ 진짜 ‖
```

0 이면 완벽, 0.1 이면 "평균적으로 10 % 틀림".
**우리 판정 기준: 0.10 미만이면 유망.**

**코드.** `fit_kv_projection.py` 의 `rel_err(Y, Yhat)`.

---

## 과적합 / holdout / 일반화

**과적합(overfitting).** 연립방정식의 미지수가 관측보다 많으면,
주어진 데이터는 완벽히 맞추면서 새 데이터에서는 엉망이 된다.

```
A_l 의 미지수: 4096 × 1024
필요한 관측 : 최소 4096 토큰
우리 첫 시도 : 460 토큰  ← 부족해서 다시 수집
```

**holdout.** 데이터의 일부(20 %)를 **적합에 쓰지 않고 남겨** 두었다가 거기서 오차를 재는 것.
이래야 **일반화**(새 입력에서도 되는가)를 알 수 있다.

> holdout 오차가 적합 오차보다 훨씬 크면 과적합이다. **적합 오차만 보고 판단하면 안 된다.**

---

## ridge (정칙화)

관측이 부족할 때 해가 불안정해지는 것을 막는 보정.
`(XᵀX + λI)` 처럼 대각선에 작은 값을 더한다.

우리 구현은 스케일 불변 형태다 — `XtX[diag] += ridge * trace(XtX) / h`.
행렬의 전체 크기에 비례해 더하므로 `ridge` 값(기본 1e-3)이 데이터 스케일에 안 휘둘린다.

---

## 정확도 벤치마크

| 이름 | 무엇 |
|---|---|
| **RULER** | 장문 컨텍스트 종합 벤치. 길이별 난이도 스케일링 |
| **LongBench** | 장문 태스크 모음 |
| **needle-in-haystack** | 긴 문서에 심어둔 한 문장을 찾아내는가 |

**우리에게 유리한 조건** ([07 §6](../07-design-block-parallel.md)):
**우리는 블록 수가 4~8 로 적다.** 초장문을 겨냥한 기존 연구들은 수십 블록을 쓰므로
블록 간 attention 손실이 훨씬 크다. Star Attention 은 4~8블록에서 97~100 % 를 보고한다.

---

## matched-accuracy 비교

**뜻.** 근사 기법과 exact 기법을 비교할 때, **정확도를 같은 지점에 맞춰 놓고** 속도를 재는 것.

**왜 필요한가.** 우리가 근사(블록 병렬)를 쓰므로
**"근사인데 exact 와 비교 불가"** 라는 리뷰어 공격이 온다.
대비: exact Ring CP 를 **직접 구현**해 동일 정확도 지점에서 비교한다
([00 §3 Baseline](../00-RESEARCH-PLAN.md), [07 §6](../07-design-block-parallel.md)).

**필수 baseline 4종:**

| Baseline | 목적 |
|---|---|
| TP (dllama 현행) | 현재 시스템 |
| Ring CP (pass-KV, exact) | exact 상한 |
| **Star Attention 직접 포팅** | "포팅 아니냐" 질문을 그래프 하나로 종결 |
| 제안 기법 | — |

---

## byte-for-byte 동일

**뜻.** 최적화 전후 출력이 **바이트 단위로 같은지** 확인하는 것 (seed 고정, temp 0, md5 대조).

**왜 이 프로젝트에서 특히 중요한가.** 발견 #1·#2·#4 와 lm_head 제한은
**전부 "빠르게만 하고 결과는 안 바꾼다"** 는 주장이다. 네 항목 모두 확인했다
([06 §2](../06-baseline-status.md)).

repack 만은 예외적으로 L2 상대오차 ~1e-6 이 남는데,
**양자화 차이가 아니라 fp32 누적 순서 차이 수준**이다.

---

# Part 6. 연구 방법론 용어

## 체제 (regime)

**뜻.** 하드웨어·워크로드 파라미터가 만드는 **동작 영역**.
같은 알고리즘이라도 체제가 다르면 최적 구성이 달라진다.

**이 연구의 한 줄 요약이 정확히 이것이다** ([00 §0](../00-RESEARCH-PLAN.md)):

> 기존 병렬 컨텍스트 인코딩 기법은 **초장문 · 고대역폭 · 연산 풍부** 체제에서 설계되었다.
> SBC 커머디티 클러스터(CPU only, 1GbE, 프롬프트 0.5~4K)는 그 체제에 속하지 않으며,
> 이 체제에서는 최적 구성이 **정량이 아니라 정성적으로** 달라진다.

**논문 클레임 초안:**

> **We do not propose a new attention mechanism.** We show that existing parallel
> context-encoding designs are tuned for a regime — ultra-long prompts, high-bandwidth
> interconnects, abundant compute — that commodity SBC clusters do not occupy, and that in
> the short-prompt, low-bandwidth, compute-scarce regime their optimal configuration changes
> **qualitatively**, not merely quantitatively.

**우리가 제안하는 것은 attention 메커니즘이 아니라 "무엇을 어떻게 나눌지 결정하는 규칙"이다.**

---

## phase diagram / ρ

**뜻.** (대역폭 × 프롬프트 길이 × 노드 수) 축에서 **최적 구성이 바뀌는 경계**를 그린 지도.
[roofline](#roofline) 을 분산 축으로 확장한 것.

`ρ` 는 체제를 가르는 **무차원 파라미터** — [00 §3](../00-RESEARCH-PLAN.md) 이 기여 2번으로 예고했지만
**아직 정의되지 않았다.**

**리스크 대비.** "체제 차이가 사소하다"는 공격에는
phase diagram 에서 **선택이 뒤바뀌는 경계가 실사용 구간(1GbE, 4~8노드, 0.5~4K)을 관통함**을
보이는 것으로 답한다.

---

## 가설 H1 ~ H5

[00 §2](../00-RESEARCH-PLAN.md). **EXP-1 이 검증/반증할 대상이다.**

### H1 — 통신+대기가 25~45 %, 그중 대기가 전송보다 크다

**반증되면**: 통신 최적화 노선 폐기, 연산 축(H3)만 추진.

**현재 상태**: **다중 노드 스윕 미실행 → 미판정.**
단 07 의 예비 측정에서 TP2 의 wait:xfer 가 **7:1** 로 나왔다(14.6 vs 2.0초).
판정 기준은 `T_wait` 비중 > 20 % ([01 §5](../01-EXP1-cost-breakdown.md)).

### H2 — attention 비지배

S ≤ 4K 에서는 O(S²) 항이 지배적이지 않고 **MLP/projection 이 병목**.

**현재 상태**: **성립.** 단 두 번 뒤집혔다 — **이 이력이 중요하다.**

```
1차 관찰 (S=110)      : attn 0.5 %  → H2 성립
2차 관찰 (S=1789)     : attn 57 %   → "S≈1400 에서 역전, H2 는 S<500 에서만"
3차 (커널 수정 후)     : attn 13.3 % → 2차 관찰은 깨진 커널의 인공물이었다
```

> **H2 재정정** ([03 §7](../03-attention-batching.md)): "S≈1400 에서 attention 이 역전된다"는
> 관찰은 **발견 #2 의 버그가 만든 인공물이었다.** 정상 커널에서는 S=1789 에서도 13.3 % 다.
> SBC 실사용 구간(≤2K)에서 prefill 은 **FFN 이 지배(63.8 %)** 한다.
> → **분할 축 선택이 논문의 메인이 된다.**

**교훈**: 측정 전에 구현부터 확인해야 한다. 08 이나 다른 문서에서 "attention 57 %"를 보면
**그건 폐기된 수치다.**

### H3 — 깊이 분해로 prefill FLOPs ~48 % 감소

상위 절반 레이어의 KV 를 x_k 에서 직접 산출. → [Part 4](#part-4-우리-알고리즘의-수학).
**반증되면**: depth 축 폐기.

**현재 상태**: fit 스크립트만 있고 정확도 미검증.

### H4 — anchor amortization 이 SBC 체제에서 무너진다

→ [anchor amortization](#anchor-amortization). **반증되면**: 논문의 핵심 관찰 소멸 → 재설계.

**현재 상태**: 미검증(EXP-2 Star Attention 포팅 필요).
**H2 와 H4 가 이 연구의 핵심이다** — 둘 다 "기존 설계의 전제가 이 체제에서 깨진다"는
형태라 방어하기 쉽다.

### H5 — overlap 이 zero-sum

CPU 코어가 4개뿐이라 통신 스레드와 GEMM 이 경합.
**반증되면**: GPU 식 overlap 설계를 그대로 사용.

**현재 상태**: 미검증. 단 발견 #5 의 `poll()` 선택이 이 제약을 이미 반영한다 —
스핀이 아니라 커널 대기라야 코어를 안 뺏는다.

---

## baseline 위생 (baseline hygiene)

**뜻.** 우리 구현의 버그를 고치는 것. **논문 기여가 아니다.**

**발견 #1~#5 가 전부 여기 해당한다.** 이 구분을 흐리면
**"구현 미숙을 이긴 것"** 이라는 치명적 공격을 받는다.

**"SBC 문제인가, distributed-llama 문제인가"** ([03 §4](../03-attention-batching.md)):

| | 성격 | 근거 |
|---|---|---|
| 발견 #1 `-mtune=native` | 순수 빌드 설정 버그 | llama.cpp 는 ARM 에서 `-mcpu=native` 사용 |
| 발견 #2 attention | decode 전용 커널을 prefill 이 물려받음 | llama.cpp/ggml 은 attention 을 GEMM 2개로 표현하고 flash-attention op 도 보유 |

**둘 다 distributed-llama 구현 문제다.** distributed-llama 는 애초에 "홈 클러스터에서
decode 를 텐서 병렬로 가속"이 목표였고 prefill 은 부산물이었다.
**"SBC 라서 안 맞는" 게 아니라 "이 엔진이 decode 최적화만 되어 있어서"** 다.

**논문에서의 취급** ([02 §5(d)](../02-baseline-dotprod-fix.md)):
- 평가 섹션의 baseline 은 **수정된 빌드**로 다시 측정
- 발견 자체는 "SBC 에서 커널 디스패치 검증의 중요성" 정도로 각주 처리하거나
  artifact/reproducibility 섹션에
- upstream 에 리포트할 가치가 있다

**baseline 전략 3종 — 전부 필요하다** ([03 §6](../03-attention-batching.md)):

1. distributed-llama 를 고쳐서 쓴다 (발견 #1~#4 ✔)
2. **llama.cpp 를 외부 기준선으로 대조** — 같은 노드/모델/양자화로 `llama-bench`.
   **이게 있어야 "네 baseline 이 원래 느린 거 아니냐"가 그래프 하나로 끝난다**
3. 분산 실험 baseline 은 수정본 + llama.cpp RPC 둘 다

**외부 기준선 현황** (llama.cpp build 6ea215d, 같은 노드/Q4_0/4스레드 — [06 §1](../06-baseline-status.md)):

| | tok/s |
|---|---|
| pp512 | **16.15 ± 0.14** |
| pp2048 | **15.27 ± 0.43** |

> 최초 측정 17.91 은 편차 ±1.78 로 불안정했다. **재측정값을 기준으로 쓴다.**

> **그리고 분산 논문의 baseline 은 분산 baseline이다** ([06 §5](../06-baseline-status.md)).
> 단일 노드 연산만 고쳐놓고 완료로 간주하면, 다중 노드에서 새 구현 문제가 나올 때
> 측정을 전부 다시 해야 한다. **다중 노드는 아직 시작도 안 했다.**

---

## measurement study

**뜻.** 새 기법이 아니라 **"기존 설계의 전제가 이 환경에서 깨진다"는 측정 자체**가 기여인 논문.

[00 §3](../00-RESEARCH-PLAN.md) 의 기여 1번이 이 형태다 —
N-node SBC/Ethernet 클러스터에서 기존 설계의 전제 3가지가 깨지는 것을 실측
(anchor amortization / DMA-offloaded communication / attention-dominated cost).

---

## 발견 #n 색인

| # | 무엇 | 분류 | 문서 | 효과 |
|---|---|---|---|---|
| **#1** | `-mtune=native` 가 dotprod 경로를 껐다 | baseline 정정 | [02](../02-baseline-dotprod-fix.md) | 2.8× |
| **#2** | prefill 이 decode 커널을 재사용해 KV 반복 스트리밍 | baseline 정정 | [03](../03-attention-batching.md) | attention 3.6~8.5× |
| **#3** | 토큰 임베딩이 F32 로 저장됨 | 설계 공간 | [03 §6.5](../03-attention-batching.md) | 6.32 → 4.34 GB 가능 |
| **#4** | Q4_0 repack 부재 | baseline 정정 | [06 §3](../06-baseline-status.md) | GEMM 2.2~3.2× |
| **#5** | 배리어 대기 루프의 1 ms sleep | baseline 정정 | **문서 없음** (`nn-network.cpp:95`) | 배리어 43.9 → 16.3 ms |

> 번호가 문서 순서와 어긋난다(#5 가 07 에서 언급됨). **발견된 순서다.**

---

## 진행 순서 / 리스크

**EXP 순서** ([00 §5](../00-RESEARCH-PLAN.md)) — **EXP-1 결과가 스토리를 결정한다:**

```
[ ] EXP-1 prefill 비용 분해        ← 현재 단계 (계측 완료, 스윕 미실행)
[ ] EXP-2 Star Attention 포팅 + anchor 오버헤드 실측 (H4)
[ ] EXP-3 정확도 스윕 (블록 2/4/8 × anchor 크기)
[ ] EXP-4 exact Ring CP → matched-accuracy 비교
[ ] 비용 모델 피팅 → phase diagram
[ ] 플래너 알고리즘
```

**주요 리스크와 대비** ([00 §5](../00-RESEARCH-PLAN.md)):

| 리스크 | 대비 |
|---|---|
| "체제 차이가 사소하다" | phase diagram 의 경계가 실사용 구간을 관통함을 보인다 |
| "Pi 에서만 성립" | SBC 2종 이상 + 대역폭 스윕(100M/1G/2.5G) |
| "근사인데 exact 와 비교 불가" | exact CP 직접 구현 → matched-accuracy |
| "기여가 엔지니어링" | 비용 모델과 최적화 문제 정식화를 본문 전면 배치 |
| Pulsar Attention 과 충돌 | anchor 를 **정확도**가 아니라 **통신 vs 재계산 비용** 각도로만 |
| "Star Attention 포팅 아니냐" | anchor 브로드캐스트가 고유 기여. 재계산 방식과 실측 비교(B-4) |

**타겟.** 주: INFOCOM / ICDCS / MobiSys / IPDPS. 상방: MLSys / EuroSys.
저널: TPDS / TMC / IoT-J. 예비 발표: KCC/KSC.

---

## 운영 규칙 (측정을 망치지 않으려면)

[06 §6](../06-baseline-status.md) 이 "이번에 크게 데임"이라고 적은 항목들.
**`prefill_bench/run_one.sh` 에 가드로 고정되어 있다.**

| 사고 | 원인 | 가드 |
|---|---|---|
| OOM #1 | 백그라운드가 죽은 줄 알고 두 번째 실행 → 8B 모델 2개 동시 로드 | 프로세스 중복 탐지(이름 변형까지) |
| OOM #2 | 추론 중 빌드 동시 실행 | 시작 전 가용 메모리 ≥ 8 GB 확인 |
| OOM #3 | KV 가 `--max-seq-len` 전체로 할당 | 프롬프트 길이에서 자동 산정 |
| 디스크 19.5 GB 소진 | 종료 시 트래픽 요약 무한 출력 | 로그 200 MB 상한 감시 + 출력 횟수 제한 |
| 좀비 프로세스 | 이전 세션의 `dllama inference` 생존 | 중복 실행 가드가 탐지 |

**진단할 때의 규칙 3가지:**

1. **한 번에 하나만 바꾼다.** 바이너리·프롬프트 길이·repack 유무를 동시에 바꿔 원인 추적이 세 바퀴 돌았다.
2. **바이너리 신원을 매번 확인한다.** stale 바이너리를 실행해 "고쳤는데 왜 그대로지"를 반복했다.
   root/worker `md5sum` 대조를 실행 전에.
3. **로그는 크기부터 본다.** `grep` 으로만 확인해 5.5 GB 로그를 놓쳤다.

**추가 규칙** ([01 §6](../01-EXP1-cost-breakdown.md)):
- **worker-first**: sub 노드 워커를 항상 root 보다 먼저 기동
- 로그는 `bench_prefill/<RUN_ID>/` 아래에만 (`bench_logs/` 는 PiPP 실험용)
- governor `performance` 고정, run 사이 **60초 쿨다운**
- 모델/토크나이저는 `layer-skip-bypass` 로의 심볼릭 링크 — **절대 복사 금지**

---

# Part 7. 선행연구

## 우리와의 관계별 정리

### A. 직계 조상 — 병렬 컨텍스트 인코딩

| 논문 | 한 줄 | 우리와의 간극 |
|---|---|---|
| **Star Attention** (ICML'25) [arXiv:2411.17116](https://arxiv.org/pdf/2411.17116) · [code](https://github.com/NVIDIA/Star-Attention) | 블록 로컬 인코딩(통신 0, anchor 접두) → 쿼리 단계 단일 merge. 최대 11×, 정확도 97~100 % | **S=128K 전제.** 호스트당 여러 블록 → anchor amortize. 우리는 amortize 불가 → 2× 연산 |
| **APE** (ICLR'25) [arXiv:2502.05431](https://arxiv.org/pdf/2502.05431) | 병렬 인코딩 KV 의 분포 정렬(shared prefix, attention temperature, scaling) | 동일하게 초장문·RAG 전제. **정확도 회복 기법으로 차용 가능** |
| **Pulsar Attention** [arXiv:2607.20457](https://arxiv.org/html/2607.20457) | anchor 를 통계적 요약으로 대체 | **정면 충돌 위험.** 정확도 각도로 다루므로 우리는 **비용 각도로만** |
| Block-Attention / CacheBlend / EPIC | RAG 프리픽스 재사용 | 같은 문제의 다른 각도 |

### B. 직계 조상 — KV 생성 연산 줄이기

| 논문 | 한 줄 | 우리와의 간극 |
|---|---|---|
| **SwiftKV** [arXiv:2410.03960](https://arxiv.org/pdf/2410.03960) | SingleInputKV — 상위 레이어 KV 를 x_k 에서 직접. prefill FLOPs 30~50 %↓ | **GPU distillation 필요**, dense 행렬. 우리는 **calibration-only + 공유 저랭크** |
| **KVSharer** [arXiv:2410.18517](https://arxiv.org/pdf/2410.18517) / **CommonKV** [arXiv:2508.16134](https://arxiv.org/abs/2508.16134) | training-free 레이어 간 KV 공유 | **공유 기저 가설의 방증.** 어느 레이어를 근사해도 안전한지 고르는 지표 |
| **LazyLLM** [OpenReview](https://openreview.net/forum?id=am5Z8dXoaV) / **Speculative Prefill** [arXiv:2502.02789](https://arxiv.org/pdf/2502.02789) | 토큰 중요도 기반 prefill 축소 | PiPP(decode skip)와의 서사 연결 지점 |
| **PrefillOnly** [arXiv:2505.07203](https://arxiv.org/html/2505.07203v1) | prefill-only 워크로드는 마지막 레이어 KV 만 필요 | **lm_head 낭비 관찰과 같은 뿌리** |

### C. exact 분산 — 우리 baseline

| 논문 | 한 줄 |
|---|---|
| **Context Parallelism** [arXiv:2411.01783](https://arxiv.org/pdf/2411.01783) | pass-KV / pass-Q ring. **TCP 등 저대역폭에서도 확장됨** — 우리 exact baseline 설계도이자 decode LSE 병합의 근거 |
| **Ring Attention** [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) | CP 의 원형 |

### D. 경쟁군 — 엣지/홈 클러스터 분산 추론

| 논문 | 한 줄 | 간극 |
|---|---|---|
| **Galaxy** (INFOCOM'24) [arXiv:2405.17245](https://arxiv.org/pdf/2405.17245) | TP+SP 하이브리드 + 타일 단위 통신-연산 오버랩, 최대 2.5× | **Jetson GPU 보드**, BERT/GPT2-L 급, TTFT 지표 아님. **타겟 학회의 모범 답안** |
| **Prima.cpp** [arXiv:2504.08791](https://arxiv.org/abs/2504.08791) | piped-ring + mmap prefetch, 70B 홈 클러스터 | 목표가 **decode** token latency |
| **TPI-LLM** [arXiv:2410.00531](https://arxiv.org/pdf/2410.00531) | sliding-window 메모리 관리 + 링크 최적화 | 메모리 제약 해소가 주목적 |
| EdgeShard / PipeEdge | 이종 디바이스 레이어 분할(DP) | 정적 분할, 청크 파이프라이닝 없음 |
| **Dynamic Micro-Batch & Token-Budget** (Sensors'26) [doi](https://doi.org/10.3390/s26041101) | 4노드 PP token budget 동적 조정, 버블 55 %↓ | **아이디어 골격 겹침 주의.** GPU 기준 |

### E. 방법론·기반

| 대상 | 한 줄 |
|---|---|
| **Roofline** (CACM'09) | 산술 강도/균형점의 원전. **우리 phase diagram 이 이것의 변형** |
| **Pope et al.** (MLSys'23) [arXiv:2211.05102](https://arxiv.org/abs/2211.05102) | 추론 병렬화 + 산술 강도의 체계화. **분산 설계 논의의 공통 언어** |
| **FlashAttention** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | "attention 은 연산이 아니라 **메모리 이동** 문제". 발견 #2 의 사고방식 원본 |
| **SARATHI** [arXiv:2308.16369](https://arxiv.org/pdf/2308.16369) → Sarathi-Serve | **chunked prefill 의 원조.** 우리 청크 크기 논의의 직계 조상 |
| **vLLM / PagedAttention** [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | KV 캐시를 페이지로. **KV 가 1급 자원이라는 관점**의 출발 |
| **Orca** (OSDI'22) | continuous batching |
| **DistServe** / **Splitwise** | P/D disaggregation — "두 단계는 다른 기계다"의 시스템판 |
| **Alpa** (OSDI'22) [arXiv:2201.12023](https://arxiv.org/abs/2201.12023) | **"플래너 자체가 기여"인 논문의 정본.** 우리 §3-B 의 서술 모델 |
| **T-MAC** [arXiv:2407.00088](https://arxiv.org/pdf/2407.00088) | LUT 기반 저비트 CPU 커널 |
| **Sandwich** [arXiv:2507.18454](https://arxiv.org/pdf/2507.18454) | CPU 서빙 prefill/decode 컴파일 분리. **우리 문제의식과 가장 가까운 CPU 논문** |
| **llama.cpp / ggml** | CPU 추론의 사실상 표준. **우리 baseline 의 외부 기준선** |

> 읽는 순서는 [04-reading-path](../04-reading-path.md) 의 Stage 0~5 를 따를 것.
> **Stage 0(roofline, Pope, FlashAttention)을 건너뛰지 말 것** — 우리 논거가 전부 그 언어로 쓰인다.

---

# 부록 A. 숫자 한 곳에

**출처를 붙였다. 회차가 다른 값이 섞여 있으니 표끼리 빼지 말 것.**

## 모델 (Llama-3-8B)

| | 값 |
|---|---|
| h (hidden) | 4096 |
| d_ff | 14336 |
| L (레이어) | 32 |
| nHeads / nKvHeads | 32 / 8 (GQA, kvMul=4) |
| headDim | 128 |
| kvDim | 1024 |
| vocab | 128256 |
| 가중치 (dllama .m) | **6.32 GB** (임베딩 F32) |
| 가중치 (GGUF Q4_0) | 4.34 GB |
| 레이어·토큰당 FLOPs | 436 M |

## 하드웨어 (Pi5, Cortex-A76 ×4)

| | 값 | 출처 |
|---|---|---|
| GEMM 포화 성능 | **~110 GFLOPS** (배치 32 이상) | 02 §3 |
| 메모리 대역폭 | ~8–10 GB/s | 03 §5 |
| machine balance | **11~14 FLOP/byte** | 05 B-3 |
| fp32 NEON 피크 | 76.8 GFLOPS | 03 §8 |
| 네트워크 | 1 GbE | — |

## 단일 노드 TTFT 누적 개선 (S=447)

| 단계 | prefill | tok/s | llama.cpp 대비 |
|---|---|---|---|
| 원본 | 496 ms/tok | 2.01 | 12 % |
| + dotprod | 78,150 ms | 5.72 | 35 % |
| + attention 배치화 | 72,938 ms | 6.13 | 38 % |
| + repack | 35,889 ms | 12.45 | 77 % |
| + lm_head 제한 | **32,417 ms** | **13.79** | **85 %** |

출처 06 §2. 누적 6.9×.

## 분산 축 실측 (S=447)

| 구성 | prefill | syncWait | syncXfer | gemm |
|---|---|---|---|---|
| 단일 | 37,939 ms | — | — | 33,079 |
| TP2 | 33,633 | 14,611 | 1,965 | 13,942 |
| PP4+wave | 39,825 | 23,102 | 829 | — |
| SP2 | 43,620 | 1,785 | 537 | 36,502 |
| SP4 | 47,374 | 8,178 | 1,687 | 32,766 |

출처 07 §1. **06 §2 표와는 다른 회차다.**

## 비용 구성비 (S=1789, 발견 #2 수정 후)

| 항목 | 시간 | 비중 |
|---|---|---|
| FFN | 201.3 s | 63.8 % |
| attention projection | 47.6 s | 15.1 % |
| attention core | 41.9 s | 13.3 % |
| lm_head | 18.0 s | 5.7 % |
| norm/기타 | ~7 s | 2.2 % |

출처 05 B-1.

## 통신·배리어

| | 값 | 출처 |
|---|---|---|
| 배리어 1회 (poll 수정 후) | **16.3 ms** | 07 §4 |
| 배리어 1회 (수정 전) | 43.9 ms (회선 6.9 + sleep 37) | nn-network.cpp:99 |
| TP2 배리어 횟수 (S=447) | 896 | 07 §4 |
| TP 배리어 빈도 | 레이어당 2회 → 청크당 64회 | llm.cpp:454, 605 |
| KV 수집 비용 | S=447 → 117 MB ≈ 1초 / S=2048 → 537 MB ≈ 4.8초 | 07 §2 |

## 깊이 분해 절감 (상위 16레이어, K+V)

| | 파라미터 | 스트리밍량 | 연산 |
|---|---|---|---|
| dense | 134 M | 75 MB | 268 MFLOP/token |
| 공유 기저 | 1.58 M | 0.9 MB | 18.9 MFLOP/token |
| 절감 | 85× | **83×** | 14× |

출처 08 §4-4.

---

# 부록 B. 자주 틀리는 것 모음

1. **"attention 이 57 % 다"** → **폐기된 수치.** 깨진 커널의 인공물. 정상은 13.3 %.
2. **`gemmMs` 를 `attnProjMs`+`ffnMs`+`lmHeadMs` 와 함께 더하기** → 중복. gemmMs 가 그 합이다.
3. **07 §1 표와 06 §2 표의 숫자를 섞기** → 다른 회차. lm_head 제한 적용 여부가 다르다.
4. **dllama `--sp-size` 를 Sequence Parallel 로 읽기** → 연산은 안 나뉜다.
5. **`A`(anchor 토큰 수)와 `A_l`(사영 행렬)을 같은 것으로 읽기** → 무관한 개념.
6. **블록 병렬과 깊이 분해를 하나로 읽기** → 별개의 두 아이디어. 곱해진다.
7. **KV 가 프롬프트 길이만큼 잡힌다고 가정** → `--max-seq-len` 전체로 잡힌다. OOM.
8. **wave 모드 로그에서 op 분해를 찾기** → 비동기라 집계에서 제외된다.
9. **적합 오차만 보고 깊이 분해 성공 판정** → holdout 오차를 봐야 한다.
10. **단일 노드 baseline 을 고쳤으니 baseline 완료로 간주** → 분산 논문의 baseline 은 분산 baseline이다.
