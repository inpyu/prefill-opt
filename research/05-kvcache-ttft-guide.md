# KV 캐시 생성과 TTFT — 학습 가이드

이 저장소의 실제 코드 경로와 실측값에 붙여 쓴 가이드.
Llama-3-8B 기준: `h=4096`, `nHeads=32`, `nKvHeads=8`(GQA), `headDim=128`,
`kvDim=1024`, `L=32`, `d_ff=14336`, `vocab=128256`.

---

# Part A — KV 캐시는 어떻게 만들어지는가

## A-1. 한 레이어에서 일어나는 일

`src/llm.cpp:306-425` 가 만드는 op 그래프가 그대로 답이다. 레이어 `l` 의 입력 `x_l` 에 대해:

```
x_l
 ├─ OP_INV_RMS  "block_norm_pre"      RMS 계산
 ├─ OP_RMS_NORM "block_norm"          정규화        → y
 ├─ OP_CAST     "block_cast_y"        F32 → Q80     → yq   (matmul 입력 양자화)
 │
 ├─ OP_MATMUL   "block_matmul_q"      yq × W_q → q
 ├─ OP_MATMUL   "block_matmul_k"      yq × W_k → kTemp     ← K 생성
 ├─ OP_MATMUL   "block_matmul_v"      yq × W_v → vTemp     ← V 생성
 │
 ├─ OP_ROPE     "block_rope_k"        위치 인코딩 (position 의존!)
 │
 ├─ OP_SHIFT    "block_shift_k"       kTemp → kBuffer[pos] ← 실제 캐시 쓰기
 ├─ OP_SHIFT    "block_shift_v"       vTemp → vBuffer[pos]
 │
 └─ (spSize>1)  addSpKvSync            SP 그룹 간 KV allgather
```

**핵심 3가지**

1. **KV 생성 자체는 단순 선형 사영이다.** `yq × W_k`, `yq × W_v` 두 개의 GEMM이 전부다.
2. **RoPE 때문에 위치에 의존한다.** 캐시에 들어가는 K는 RoPE가 적용된 값이다.
   → 위치를 바꿔 재사용하려면 RoPE를 되돌려야 한다(APE 계열 기법이 여기를 건드린다).
3. **캐시 쓰기는 `OP_SHIFT` 다.** 이름이 shift인 이유는 `positionPipe` 가 가리키는 위치에
   기록하는 방식이기 때문. `NnShiftOpCodeConfig{positionPipeIndex, localSeqStart, localSeqLen}` —
   **이미 시퀀스 차원 슬라이싱 인자를 갖고 있다**(Part D-3에서 중요).

## A-2. 왜 순차적인가 — 이것이 모든 문제의 뿌리

`x_{l+1} = x_l + Attn(x_l) + FFN(...)` 이므로 레이어 `l+1` 의 KV를 만들려면
레이어 `l` 의 **attention과 FFN을 전부 끝내야** 한다.

> **KV를 만드는 데 필요한 본질적 연산은 전체의 3.9% 인데,
> 그 입력 `x_l` 을 얻으려고 나머지 96.1% 를 순차로 돌린다.**

토큰 1개·레이어 1개당 FLOPs:

| 연산 | FLOPs | 비중 |
|---|---|---|
| K projection (2·h·kvDim) | 8.4 M | 1.9 % |
| V projection | 8.4 M | 1.9 % |
| Q projection (2·h·h) | 33.6 M | 7.7 % |
| O projection | 33.6 M | 7.7 % |
| FFN SwiGLU (3·2·h·d_ff) | 352.3 M | 80.7 % |
| **합계** | **436 M** | 100 % |

이 관찰이 SwiftKV 계열(상위 레이어 KV를 `x_k` 에서 직접 산출)의 출발점이다.
→ [00-RESEARCH-PLAN.md](00-RESEARCH-PLAN.md) §1

## A-3. 메모리 레이아웃과 크기

`nn-core.cpp:215` `sliceKvCache()`:

```c
s.kvDim0       = kvDim / nNodes;      // TP: 헤드 차원 분할
s.localSeqLen  = seqLen / spSize;     // SP: 시퀀스 차원 분할
s.localSeqStart= spRank * localSeqLen;
s.keySize      = size2D(F_32, seqLen, kvDim0);   // 버퍼는 항상 full seqLen
```

버퍼 인덱싱은 `keyCache[headIndex * headDim + t * kvDim0]` — **position-major, head-minor**.
즉 한 위치 `t` 의 모든 KV 헤드가 연속으로 놓인다.

**크기 계산 (Llama-3-8B, 1노드)**

| | 계산 | 값 |
|---|---|---|
| 토큰·레이어당 | 2 × kvDim × 4B (F32) | 8 KB |
| 토큰당 (32레이어) | × 32 | **256 KB** |
| S=2048 프롬프트 | × 2048 | **512 MB** |

> ⚠️ **이 코드베이스는 KV를 F32로 저장한다.** llama.cpp는 기본 F16이다.
> → 메모리 2배, **attention 메모리 트래픽도 2배**. F16 전환은 무손실에 가까운 개선 후보다.
> ([03-attention-batching.md](03-attention-batching.md) 후속 항목)

## A-4. 직접 확인해 볼 것

```bash
# 1) op 그래프에서 KV 경로만 뽑아보기
grep -n "block_matmul_k\|block_rope_k\|block_shift_k" src/llm.cpp

# 2) 캐시 슬라이싱 규칙
sed -n '215,228p' src/nn/nn-core.cpp

# 3) 실제 KV projection 시간 비중 (attnProjMs 안에 포함)
./dllama inference ... --stage-timing 1 | grep -E "attnProjMs|ffnMs"
```

---

# Part B — TTFT를 결정하는 요소

## B-1. 정의와 분해

```
TTFT = prefill + (첫 decode 스텝)
prefill = Σ_chunks ( 연산 + 통신 + 동기대기 )
```

단일 노드에서는 통신·대기가 0이므로 **연산만 남는다**. 우리 실측(S=1789, Pi5 1노드,
[03-attention-batching.md](03-attention-batching.md) 수정 적용 후):

| 항목 | 시간 | 비중 | S에 대한 스케일링 |
|---|---|---|---|
| **FFN** | 201.3 s | **63.8 %** | O(S · h · d_ff) — 선형 |
| attention projection (Q,K,V,O) | 47.6 s | 15.1 % | O(S · h²) — 선형 |
| attention core (score+AV) | 41.9 s | 13.3 % | **O(S² · h) — 이차** |
| **lm_head** | 18.0 s | **5.7 %** | O(S · h · V) — 선형, **그런데 필요한 건 1토큰뿐** |
| norm / 기타 | ~7 s | 2.2 % | 선형 |

**읽어낼 것**

- **FFN이 지배한다.** 어떤 최적화든 FFN을 건드리지 않으면 상한이 36%다.
- **attention core만 이차항이다.** S가 커질수록 비중이 오르지만, 커널이 정상이면
  S=1789에서도 13.3%다. (깨진 커널에서는 57%였다 — 측정 전에 구현부터 확인해야 하는 이유)
- **lm_head 5.7%는 순수 낭비다.** TTFT에 필요한 로짓은 마지막 토큰 1개뿐인데
  모든 위치에 대해 `2·h·V = 1.05 GFLOP/token` 을 계산하고 있다. 레이어 하나의 2.4배다.

## B-2. TTFT 산출물은 정확히 두 개뿐

> 1. 모든 토큰 × 모든 레이어의 **KV 캐시**
> 2. **마지막 토큰 1개**의 로짓

토큰 `0..S-2` 의 상위 레이어 residual stream은 **오직 그 레이어의 KV를 만들기 위해서만**
존재한다. 이 사실이 모든 prefill 최적화의 논리적 출발점이다.
(→ lm_head 제거, SwiftKV식 depth 분해, PrefillOnly 논문)

## B-3. 하드웨어 쪽 요소 — machine balance

Pi5 (Cortex-A76 ×4) 실측:

| | 값 |
|---|---|
| 연산 (int8 dotprod GEMM) | ~110 GFLOPS (배치 32 이상에서 포화) |
| 메모리 대역폭 | ~8–10 GB/s |
| **균형점** | **약 11–14 FLOP/byte** |

산술 강도가 이보다 낮으면 memory-bound다.

- **GEMM(FFN, projection)**: 배치 B일 때 강도 ≈ B (가중치 1회 읽기에 B토큰 연산)
  → B ≥ 12~14 면 compute-bound. **실측 포화점 32와 일치**
- **attention**: 타일 크기 T일 때 강도 ≈ T/2 (F32 KV)
  → T ≈ 22~28 필요

> **규칙: Pi5에서 prefill 청크는 32 이상이어야 GEMM과 attention이 동시에 memory-bound를
> 벗어난다.** machine balance의 함수이므로 다른 하드웨어로 전이 가능하다.

## B-4. 분산했을 때 추가되는 요소

| 요소 | 내용 |
|---|---|
| 통신량 | prefill의 all-reduce 1회 크기 = `2·B·S·h` (decode는 `2·B·h`) — **S배 크다** |
| 동기 대기 | 집합통신은 가장 느린 노드에 맞춰진다. 열 스로틀링·OS 스케줄링 jitter가 레이어마다 누적 |
| 코어 경합 | Pi5는 코어 4개뿐 — 통신 스레드가 GEMM 코어를 뺏는다 (GPU에는 없는 제약) |
| 중복 연산 | 블록 병렬 인코딩의 anchor 등, 통신을 피하려고 지불하는 연산 |

---

# Part C — 학습 순서

각 단계에 **"직접 해볼 것"** 을 붙였다. 읽기만 하면 남지 않는다.

### 1단계. 단일 토큰의 forward를 손으로 따라간다 (1~2일)
- `src/llm.cpp` 의 `buildLlmNet()` 에서 op 추가 순서를 종이에 적는다
- 각 op의 입출력 버퍼 인덱스를 연결해 그래프를 그린다
- **해볼 것**: `--stage-timing 1` 로 돌려 op 분류별 시간이 A-2의 FLOPs 표와 비례하는지 확인

### 2단계. KV 캐시의 생명주기를 추적한다 (1~2일)
- `block_shift_k` → `kBuffer` → `multiheadAtt` 에서 읽히는 경로
- `sliceKvCache()` 의 `kvDim0` / `localSeqLen` 두 축이 각각 무엇을 쪼개는지
- **해볼 것**: S를 2배로 늘려 KV 버퍼 크기와 attention 시간이 각각 어떻게 변하는지 측정
  (선형 vs 이차 확인)

### 3단계. roofline으로 각 op를 분류한다 (2~3일)
- 읽기: Roofline 원논문 → Pope et al. → [LLM Inference Unveiled](https://arxiv.org/html/2402.16363v4)
- **해볼 것**: `prefill_bench/bench_sgemm.cpp` 를 attention 커널용으로 변형해
  타일 크기 T에 대한 GFLOPS 곡선을 그린다. B-3의 T≈22~28 예측이 맞는지 검증

### 4단계. attention의 IO 관점을 익힌다 (2~3일)
- 읽기: FlashAttention → FlashAttention-2
- **해볼 것**: 우리 `multiheadAttBatch_F32` 가 아직 `att` 중간 버퍼를 메모리에
  실체화하고 있다(8.4 MB/청크). online softmax로 융합하면 얼마나 줄지 계산해 본 뒤 구현

### 5단계. 서빙 시스템이 prefill을 어떻게 다뤄왔는지 (1주)
- 읽기: Orca → vLLM/PagedAttention → SARATHI → DistServe
- **해볼 것**: `--prefill-chunk-size` 를 8/16/32/64/128로 스윕해 SARATHI의 주장이
  이 하드웨어에서도 성립하는지 확인 (B-3 규칙과 대조)

### 6단계. 분산 (Part D로)
- 읽기: Galaxy → Context Parallelism → Star Attention
- **해볼 것**: Part D의 표를 우리 클러스터 파라미터로 채워 넣는다

> 상세 논문 목록은 [04-reading-path.md](04-reading-path.md).

---

# Part D — 그래서 어떻게 분산할 것인가

## D-1. 결정의 본질

> **분산 축을 고르는 것 = KV 캐시가 "어디에, 어떤 모양으로" 남을지를 고르는 것이다.**

prefill이 끝나면 decode가 시작되는데, decode는 **마지막 토큰 1개가 전체 KV를 봐야** 한다.
따라서 축을 고를 때 반드시 세 가지를 함께 봐야 한다.

1. prefill 중 **통신량과 동기 횟수**
2. prefill 후 **KV가 어디에 남는가**
3. 그 배치에서 **decode를 돌릴 수 있는가 / 옮기는 비용은 얼마인가**

3번을 빼먹으면 "prefill은 빨라졌는데 KV를 옮기느라 TTFT가 그대로"가 된다.
이 저장소에 `--strict-kv-affinity`, `--allow-kv-migration` 플래그가 있는 것이
그 함정을 이미 겪었다는 증거다.

## D-2. 네 개의 축

| 축 | 무엇을 쪼개나 | 레이어당 통신 | KV가 남는 모양 | decode 인계 |
|---|---|---|---|---|
| **TP** (텐서) | 헤드/hidden 차원 | all-reduce ×2, `2·B·S·h` | 각 노드가 `kvDim/N` × **전체 위치** | 같은 TP 구성이면 그대로 OK |
| **PP** (레이어) | 레이어 | P2P ×1, `B·S·h` | 각 노드가 **자기 레이어** × 전체 위치 | 같은 PP면 그대로 OK |
| **CP/SP** (시퀀스) | 토큰 위치 | ring KV 교환, `B·S/N·kvDim` | 각 노드가 **자기 위치 구간** × 전체 레이어 | 노드별 partial attention + logsumexp reduce 필요 |
| **블록 병렬** (근사) | 토큰 위치, 블록 간 attention 생략 | **0** (인코딩 중) | CP와 동일 | 단일 merge |

**통신량 비교** (S=512, N=4, F32, 추정):

| 축 | prefill 총 통신량 |
|---|---|
| TP | ~768 MB |
| PP | ~25 MB |
| CP (ring, GQA) | ~100 MB |
| 블록 병렬 | ~1 MB |

> GQA가 CP에 유리하게 작용한다. 교환 대상이 hidden(4096)이 아니라 kvDim(1024)이라
> 1/4이다.

## D-3. 이 저장소가 이미 갖고 있는 것

새로 만들기 전에 확인할 것:

- **TP**: `sliceRowMatmul`, `kSlice`/`vSlice`, `SYNC_NODE_SLICES` — 완비
- **PP**: `--pp-size`, `stage_bridge_merge_add` — 완비
- **SP**: `sliceKvCache()` 의 `localSeqStart`/`localSeqLen`, `SYNC_SP_KV`,
  `--sp-size`, `--prefill-sp-only`, `--sp-prefill-threshold` — **부분적으로 있음**

  단 현재 `SYNC_SP_KV` 는 **allgather** 다 (`nn-core.cpp:223` 주석: "버퍼는 항상 full seqLen,
  각 rank가 자기 슬라이스만 쓰고 allgather로 채운다"). 즉 **KV를 전 노드에 복제**한다.
  → ring 방식(pass-KV)으로 바꾸면 통신량이 크게 준다. **CP 구현의 출발점이 여기다.**

## D-4. 결정 절차

```
[1] 모델이 노드 1대 RAM에 들어가는가?
      NO  → 가중치를 쪼개야 함 → TP 또는 PP 필수
      YES → CP / 블록 병렬 가능 (가중치 복제)

[2] 정확도 손실을 허용하는가?
      NO  → TP / PP / CP (exact)
      YES → 블록 병렬 (통신 3자릿수 감소, 대신 근사)

[3] 대역폭 대비 프롬프트 길이는?
      통신량은 TP > CP > PP > 블록 병렬
      S가 길수록 TP의 all-reduce(2·B·S·h)가 급격히 불리해진다

[4] decode를 같은 구성으로 돌릴 것인가?
      YES → prefill 축 = decode 축 으로 맞춘다 (KV 이동 0)
      NO  → KV 재배치 비용을 TTFT에 포함시켜 계산할 것

[5] 청크 크기는?
      B-3의 machine balance 규칙으로 하한(Pi5: 32)을 정하고,
      그 위에서는 통신/파이프라인 관점으로만 결정
```

## D-5. 우리 연구에서의 다음 단계

1. **EXP-1 (측정)** — 수정된 빌드로 다중 노드 TP를 돌려
   `wait_frac` / `xfer_frac` 을 확보. **H1(동기 구조가 병목) 판정**
2. **exact CP 구현** — `SYNC_SP_KV` 를 allgather → ring(pass-KV)으로.
   D-3에 기반이 이미 있으므로 신규 구현이 아니라 교체다
3. **블록 병렬(Star Attention) 포팅** — 근사 축. CP와 정확도-속도 파레토 비교
4. **플래너** — D-4 절차를 비용 모델로 정식화

> 순서가 중요하다. **1·2(exact)를 먼저 해 baseline을 정상화한 뒤 3(근사)을 얹어야**
> "구현 미숙을 이긴 것"이라는 공격을 피할 수 있다.
> → [03-attention-batching.md](03-attention-batching.md) §6
