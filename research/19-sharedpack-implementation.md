# SharedPack-SDOT 구현 — 현재 상태와 인수인계

**상태: 인프라 완료, net builder 배선 남음**  ·  작성 2026-08-22
관련: [18-compute-path.md](18-compute-path.md), [16-derivepp.md](16-derivepp.md)

> 이 문서는 세션이 끊겨도 이어받을 수 있도록 **왜 이 구조인가**와 **다음에 무엇을
> 해야 하는가**를 남긴다. 측정 근거는 §18, 논문 전체 맥락은 §16 에 있다.

---

## 0. 한 장 요약

### 무엇을 고치는가

`matmulForward_repack` 이 **스레드마다 activation 전체를 중복 변환**한다. 스레드가
출력 열로 나뉘어 모두 배치 전체를 필요로 하는데 op 내부 배리어가 없기 때문이다.

    K=14336, B=32, 4스레드
      dup(현재)    198.7 GOPS
      shared       333.8 GOPS      1.68×
      1~3스레드는 차이 없음 (1.00~1.03)

`shared` 는 K 와 무관하게 327~334 GOPS 로 평탄하다 — **panel 크기가 아니라 사본 수가
문제다.**

### 어떻게 고치는가

`OP_PACK_Q80X4` 를 **별도 op** 로 분리한다. executor 의 op 경계가 barrier 를 제공하므로
네 스레드가 서로 다른 batch-row group 을 **하나의 공유 버퍼**에 병렬로 쓸 수 있다.

    Q80 activation
           │
           ▼
    OP_PACK_Q80X4      thread t 는 group [nG*t/T, nG*(t+1)/T) 담당
           │
           ▼           ← executor op boundary = barrier
    shared block_q8_0x4
      ├─ thread 0: GEMM output columns 0
      ├─ thread 1: ...
      └─ ...

### 왜 이 순서인가

    1. Down(block_matmul_w2) 하나만    ← 지금
    2. Gate/Up (pack(yq) 공유)
    3. Q/K/V (pack(xq) 공유)
    4. F32→Q80 단계에서 SDOT layout 직접 산출  (novelty)

Down 이 최대 단일 op(29.44%)이고 격차도 가장 크다(1.68×). 하나만 먼저 통과시켜야
원인 추적이 쉽다.

---

## 1. 완료된 인프라 (커밋 전, 빌드 통과 `ebee12b3`)

| 파일 | 변경 |
|---|---|
| `src/nn/nn-core.hpp` | `OP_PACK_Q80X4` (enum **끝**에 추가), `NnPackQ80x4OpCodeConfig`, `NnMatmulOpConfig.prepackedBufferIndex`, `NN_NO_PREPACK` |
| `src/nn/nn-core.cpp` | `opCodeToString` 에 `"PACK_Q80X4"` |
| `src/nn/nn-cpu-ops.cpp` | `packQ80x4Forward` 구현, op 등록(`Q80_Q80_Q80`), `matmulForward_repack` 공유 버퍼 경로 |
| `src/llm.cpp` | 기존 `NnMatmulOpConfig{...}` 13곳에 `NN_NO_PREPACK` 채움 |

### ⚠️ enum 은 반드시 끝에 추가한다

op 코드는 **워커로 직렬화되는 값**이다. 중간에 넣으면 기존 번호가 밀려 root 와 worker
바이너리가 어긋난다(md5 가드가 잡지만 원인 파악이 어렵다).

### 기존 동작이 완전히 보존된다

`prepackedBufferIndex = NN_NO_PREPACK` 이면 `matmulForward_repack` 이 예전처럼
`thread_local` 스크래치로 중복 변환한다. 13곳을 전부 그렇게 채웠으므로 **현재 그래프는
이전과 바이트 단위로 동일하게 동작한다.** `DLLAMA_REPACK=0` 경로도 그대로다.

### `packQ80x4Forward` 분할 규칙

```cpp
nGroups = batchSize / 4;              // 4행 배수만. 나머지 1~3행은 GEMV fallback
g0 = nGroups * threadIndex / nThreads;
g1 = nGroups * (threadIndex + 1) / nThreads;
for (g = g0; g < g1; g++)
    nnPackQ80To4x4(&in[g*4*kBlocks], &out[g*kBlocks], kBlocks);
```

**한 스레드만 pack 하게 하지 않는다** — 그러면 다른 세 스레드가 노는 시간이 그대로
추가된다. 네 스레드가 나눠 쓰고 op 경계에서 만난다.

---

## 2. 남은 작업 — net builder 배선

`src/llm.cpp` 의 dense FFN 구간(`block_cast_d2` 와 `block_matmul_w2` 사이)에 넣는다.

    block_cast_d2      F32 hidden -> Q80          (기존)
           │
           ▼
    block_pack_dq      Q80 -> shared block_q8_0x4  ← 추가
           │
           ▼
    block_matmul_w2    prepackedBufferIndex = 그 버퍼

### 필요한 것

**(a) 공유 버퍼 선언**

    크기 = (maxBatch/4) * (K/32) * sizeof(block_q8_0x4)
    Down: K=14336 -> kBlocks 448.  B=32 이면 8 * 448 * 136 = 476 KiB

executor 버퍼는 `nBatches` 기준으로 잡히므로 **최대 배치로 할당**하고 실행 시
`batchSize` 만큼만 쓴다. `thread_local` 이나 전역 static 이 아니라 **실행 context 소유**.

**(b) pack op 추가** — `ff.addOp(OP_PACK_Q80X4, "block_pack_dq", layerIndex, ...)`

**(c) matmul_w2 의 config 에 버퍼 인덱스 지정**

### 주의

- **`rowBegin` 오프셋** — `matmulForward_repack` 은 `lmHeadRowRange` 로 행 범위를 정한다.
  공유 버퍼 인덱싱을 `rowBegin/4` 로 오프셋해 뒀다. Down 은 항상 전체 행이라 `rowBegin=0`
  이지만 명시적으로 처리돼 있다.
- **wave pipeline** — 같은 workspace 를 여러 마이크로배치가 덮어쓰지 않는지 확인해야
  한다. executor 버퍼는 op 단위 재사용이라 한 forward 안에서는 안전하지만 **검증 필요**.
- **4행 미만 microbatch** — pack 하지 않고 원본 Q80 으로 GEMV fallback (이미 구현됨).
- **`lm_head`** — 마지막 행 범위만 계산하므로 초기 적용 대상에서 제외한다.

---

## 3. 검증 순서 (성능보다 먼저)

1. `nnPackQ80To4x4` 기존 결과와 공유 버퍼 전체 `memcmp`
2. Down GEMM F32 output `memcmp`
3. 한 레이어 출력 비교
4. `N=1`, `B=16`/`B=32` logits — reference 이미 확보:

       N=1  8b8178a50a97   (B=16, B=32 동일)
       N=8  c52e50e37e65   (B=16)

   `DLLAMA_DUMP_LOGITS=<path>` 로 덤프한다.
5. `N=8` 동일 구성 logits
6. 마지막 4행 미만 microbatch 확인

**gate 는 동일 실행 구성끼리만 비교한다**(§17 §7.7.4). `N=1` vs `N=8` 은 원래 다르다
(q80 stage-boundary 경로) — WCEP 통과 조건이 아니었듯 여기서도 아니다.

---

## 4. 성능 판정

### ⚠️ `333.8 GOPS` 를 그대로 쓰면 안 된다

벤치에서 shared buffer 를 **타이밍 루프 밖에서** 만들었다. 따라서 그 값은

    T_shared = T_pack-once + T_barrier + T_GEMM

중 `T_GEMM` 만이다. **kernel upper bound** 다.

최종 판정은 반드시 합으로 한다. op profiler 에 `block_pack_dq` 와 `block_matmul_w2` 가
분리 출력되므로(`DLLAMA_OP_PROFILE=1`) 그대로 읽으면 된다.

### 사전등록 기준

| 항목 | 기준 |
|---|---|
| Down kernel (pack 포함) | ≥1.45×, 목표 ≥1.60× |
| Gate/Up | 비열화 금지 |
| 전체 projection 가중평균 | ≥1.15× |
| 단일 노드 E2E | ≥1.12× 1차, ≥1.15× 강한 성공 |
| 정확성 | packed bytes 및 GEMM output **bit-identical** |
| 최종 목표 | 후속 최적화 포함 E2E ≥1.20× |

### 예상 E2E

    Down          29.44% / 1.68
    Gate+Up       39.28% / 1.04
    Q/K/V/O       15.41% / 1.04
    non-GEMM      15.87%

    T_new ≈ 0.860   →   E2E ≈ 1.16×

20% 까지 남는 것은 약 2.7 %p. SharedPack 후 전체 GEMM 에 **추가 4%** 면 도달한다.

---

## 4b. 측정 결과 — Down 1.60×, 정확성 통과

### 정확성: 통과

    B=16 x2,  B=32 x2   전부 8b8178a50a97   (reference 와 일치)

네 가지가 동시에 확인됐다.

1. `packQ80x4Forward` 가 기존 중복 변환과 **바이트 단위 동일**
2. 스레드 group 분할 경계(`g0 = nG*t/T`)에 누락·중복 없음
3. `d_pack` 버퍼 선형 접근 가정이 맞음
4. **executor op 경계가 실제로 barrier 역할을 한다** — 가정이었는데 검증됨

### 성능: op profile (wave off, `--stage-timing 1`, S=447, B=32)

| op | baseline | SharedPack | 변화 |
|---|---|---|---|
| **block_matmul_w2 (Down)** | 10,111.6 | **6,326.4** | **1.60×** |
| block_pack_dq | — | **11.0** | 신규, 전체의 **0.04%** |
| block_matmul_w1 (Gate) | 6,728.7 | 6,731.7 | 1.000 (대조군) |
| block_matmul_w3 (Up) | 6,761.4 | 6,599.0 | 1.025 (대조군) |
| **total** | 34,346.5 | **30,241.4** | **1.136×** |

- **사전등록 기준 통과** — Down `≥1.45×` 통과, 목표 `≥1.60×` 도달.
  벤치의 1.68× 와 부합한다(production 은 pack op 오버헤드가 있어 약간 낮다).
- **pack 오버헤드 우려 해소** — 11.0 ms, 전체의 0.04%. 기존에는 4스레드가 각자
  전체를 변환했는데 나눠서 한 번만 하니 거의 공짜다.
- **회귀 없음** — 미적용 projection 이 1.000 / 1.025 로 불변. Down 개선이 우연이
  아님도 뒷받침한다.

### ⚠️ E2E 는 이 기기에서 판별되지 않는다

wave 를 켠 E2E 측정:

    B=32   SharedPack 30,069 / 29,933   baseline 31,208 / 30,107 / 29,637   → 1.004×
    B=16   SharedPack 26,722 / 30,365 / 40,370  (편차 **51%**)              → 중앙값 1.10×

**연산은 1.136× 빨라졌으나 wave 실행에서 그 이득이 묻힌다.** 이 기기의 E2E 편차가
최대 51% 이므로 단일·소수 측정으로는 판별할 수 없다.

> ⚠️ 첫 단일 측정(26,722 = 1.25×)만 보고 "사전등록 기준 초과" 라고 판단했다가
> 철회했다. **이 기기에서 E2E 성능은 anchor 설계 없이 판단하지 않는다.**
> op profile 은 op 시간의 직접 합이라 훨씬 안정적이다.

E2E 확정에는 §16 의 anchor 설계(정순/역순 교차, `flock` 배타 실행)가 필요하다.

---

## 4c. 전체 확장 결과 — **op 총합 1.205×**, 정확성 통과

Down 통과 후 같은 primitive 를 Q/K/V 와 Gate/Up 으로 확장했다.
`yq` 를 Q·K·V 가 공유하고(`block_pack_yq`), FFN 입력을 Gate·Up 이 공유한다
(`block_pack_yq2`). **pack 3회로 6 projection 을 커버**한다 — 기존에는
6 projection × 4 스레드 = 24회 중복 변환이었다.

### 정확성: 통과

    Down only         B=16 x2, B=32 x2    8b8178a50a97
    + Q/K/V, Gate/Up  B=16,    B=32       8b8178a50a97

`y_pack` 버퍼를 attention 과 FFN 이 **번갈아** 쓰는데도 안전하다.
(`pack_yq → q/k/v`, `norm_1` 이 yq 갱신, `pack_yq2 → w1/w3`)
wave 파이프라인에서 마이크로배치가 겹치지 않음도 확인된다.

### 성능 (op profile, wave off, `--stage-timing 1`, S=447, B=32)

| op | baseline | Down-only | 전체 | baseline 대비 |
|---|---|---|---|---|
| **block_matmul_w2 (Down)** | 10,111.6 | 6,326.4 | **6,163.6** | **1.64×** |
| block_matmul_w1 (Gate) | 6,728.7 | 6,731.7 | 6,212.6 | 1.08× |
| block_matmul_w3 (Up) | 6,761.4 | 6,599.0 | 6,141.1 | 1.10× |
| block_matmul_q | 2,117.3 | — | 1,811.8 | 1.17× |
| block_matmul_k | 572.8 | — | 459.0 | **1.25×** |
| block_matmul_v | 558.9 | — | 454.4 | **1.23×** |
| block_matmul_wo | 2,044.9 | — | 1,970.5 | 1.04× |
| **total** | 34,346.5 | 30,241.4 | **28,511.0** | **1.205×** |

**예상(1.16×)을 넘었다.** 벤치의 `K=4096` dup/shared 차이가 1.04× 였으므로
Gate/Up 은 그 수준을 예상했는데 1.08~1.10× 다. `K/V` 가 가장 크게 개선됐다
(1.23~1.25×) — 출력 차원이 작아(1024) 스레드당 열이 적은데도 각자 전체
activation 을 변환하던 비용이 상대적으로 컸기 때문으로 보인다.

### 사전등록 기준 대비

| 항목 | 기준 | 결과 |
|---|---|---|
| Down kernel | ≥1.45×, 목표 ≥1.60× | **1.64×** 통과 |
| Gate/Up 비열화 금지 | — | 1.08 / 1.10× (개선) |
| 전체 projection 가중평균 | ≥1.15× | **1.205×** 통과 |
| 정확성 | bit-identical | 통과 |

**연산 층에서 20% 목표를 달성했다.**

### pack op 비용 — 무시할 수준

| op | ms | % |
|---|---|---|
| block_pack_yq2 (Gate/Up) | 30.4 | 0.11% |
| block_pack_yq (Q/K/V) | 29.4 | 0.10% |
| block_pack_dq (Down) | 24.4 | 0.09% |
| **합계** | **84.2** | **0.30%** |

기존에는 6 projection × 4 스레드 = 24회 중복 변환이었다. 3회로 줄이고 그것도
스레드가 나눠 하니 총 비용이 전체의 0.3% 다.

### ⚠️ E2E 는 별도 확정이 필요하다

op profile 은 op 시간의 직접 합이다. wave 를 켠 E2E 는 이 기기의 편차(최대 51%)에
묻히므로 §16 의 anchor 설계(정순/역순 교차, `flock` 배타 실행)로 따로 재야 한다.

---

## 4d. 확정 — 동일 바이너리 paired anchor A/B

`artifacts/sharedpack_ab/`. **동일 바이너리 `9300a29c` 에서 `DLLAMA_SHARED_PACK` 만
전환** — 세션·빌드 교락이 없다. 커널·가중치·누적 순서는 그대로이고 그래프만 바뀐다.

    warm-up 폐기
    정순  BASE SP BASE SP BASE
    역순  SP BASE SP BASE
    = 보관 9회 (BASE 5, SP 4)
    S=447, B=32, N=1, wave off, --stage-timing 1

SD 카드 불량 섹터(§20 §1.3) 때문에 anchor 17회 대신 10회로 제한했다
(6.3 GB × 10 = 63 GB 읽기).

### 정확성: 9회 전부 동일

    logits hash 종류 1개:  8b8178a50a97

### 성능 (기하평균 paired ratio)

| 지표 | 값 | 회차별 |
|---|---|---|
| **prefill (operator time)** | **1.293×** | 1.203 / 1.423 / 1.382 / 1.181 |
| **Down (block_matmul_w2)** | **2.224×** | 1.982 / 2.446 / 2.437 / 2.070 |
| pack 비용 | 212.6 ms | — |

    BASE  n=5  중앙값 35.8s  CV 5.4%
    SP    n=4  중앙값 27.1s  CV 6.3%

### 부수 효과 — 안정성도 개선된다

`BASE` 는 31.4~36.5s 로 흔들리는 반면 `SP` 는 25.6~30.0s 다. 스레드별 사본이 만드는
cache 경합 자체가 회차마다 달라지는 변동원이었고, 공유하면 그 원인이 사라진다.

### ⚠️ 이전 수치와의 관계

| 측정 | 값 | 조건 |
|---|---|---|
| 최초 단일 측정 | 1.25× | 단일 관측 — **철회** |
| 서로 다른 세션/빌드 비교 | 1.205× | baseline 이 이전 artifact — 교락 있음 |
| **동일 바이너리 paired anchor** | **1.293×** | **확정값** |

Down 도 1.60× → 1.64× → **2.224×** 로 올라갔는데, 앞선 값들이 서로 다른 세션·빌드
비교였기 때문이다. paired 가 정확하다.

### ⚠️ 이것은 **단일 노드 prefill operator time** 이다

    확정:  N=1, wave off 에서 prefill operator time 1.293×
    미확정: 분산 wave 실행(N=8)에서의 이득

wave 를 켠 E2E 에서는 `B=32` 가 1.004× 였다(§4b). 연산이 빨라져도 파이프라인에서
묻히는지, 아니면 E2E 편차(최대 51%)에 가려진 것인지 **아직 판별하지 못했다.**

**따라서 시스템 기여(4.57~4.68×)와 곱하면 안 된다.**

    4.57 × 1.29 = 5.9x    ← 이렇게 쓰지 않는다

    층위 1  N=8 분산, wave on,  llama.cpp 대비    4.57~4.68×
    층위 2  N=1 단일, wave off, 자기 baseline 대비 1.293×

두 숫자를 **나란히 놓되 곱하지 않는다.** 곱셈을 주장하려면 `N=8, wave on` 에서
같은 flag paired A/B 가 필요하다.

---

## 5. 계측 도구 (이미 구현됨)

| 도구 | 용도 |
|---|---|
| `DLLAMA_OP_PROFILE=1` | op 이름별 시간 누적. `--stage-timing 1` 필요(그때만 step profiling 이 켜진다) |
| `DLLAMA_DUMP_LOGITS=<path>` | prefill 종료 후 logits f32 raw 1회. 정확성 reference |
| `prefill_bench/bench_shape.cpp` | dup vs shared 대조, K sweep |
| `prefill_bench/bench_roofline.cpp` | production 커널 roofline (`ggml_gemm_q4_0_4x4_q8_0`) |
| `prefill_bench/kp_sdot.cpp` | KP-SDOT 프로토타입(중단). accumulator 저장·복원 bit-identical 확인용으로 보존 |

### ⚠️ 벤치할 때 반드시 production 커널을 쓴다

    ggml_gemm_q4_0_4x4_q8_0   production (block_q4_0x4 x block_q8_0x4)
    llamafile_sgemm           production 이 아니다 — 2.6배 느리다

WCEP 에서 후자를 baseline 으로 삼을 뻔했다(§17 §7.5.1).

---

## 6. 이 방향에 오기까지 기각된 것들

| 방향 | 기각 근거 |
|---|---|
| WCEP weight-aware | matched null 대비 −0.37 %p. 실제 weight 배열에 구조 없음 (§17 §7.10) |
| WCEP range-aware hybrid | eligible panel 5.07%, `E_max = 1.053×` (§17 §7.11.7) |
| QCFuse (FFN 중간버퍼) | 제거 대상 2.45%, 상한 1.025× (§18 §2) |
| KP-SDOT (K panelization) | shared 는 238 KiB panel 에서도 평탄. 없는 문제를 푼다 (§18 §6c) |

**KP-SDOT 의 부산물은 보존한다** — accumulator 저장·복원으로 K-panel continuation 이
bit-identical 함을 확인했다(P=64/128/224 모두 YES). 나중에 필요하면 쓸 수 있다.

---

## 7. 반복된 실수 (같은 함정을 다시 밟지 않기 위해)

| 실수 | 횟수 | 대응 |
|---|---|---|
| `pgrep -f`/`pkill -f` 가 자기 명령줄 매칭 | 5 | `-x` 사용. 정리와 실행을 **다른 명령으로 분리** |
| `\| head` 가 SIGPIPE 로 실행 중단 | 2 | 긴 실행은 파일로 받고 나중에 grep |
| 소표본 관측을 일반 결론으로 확대 | 3 | anchor 설계 + 사전등록 기준 |
| 회계 단위 불일치 | 3 | baseline 과 최적화가 같은 단위인지 매번 확인 |
| 파일 포맷 오인 (embedding 타입, FFN 순서) | 2 | **검증기를 먼저 쓴다.** byte 크기만으로 부족 — 이름·shape 를 loader 와 대조 |
| 두 프로젝트 자원 충돌 (바이너리·포트) | 2 | 이름과 포트를 분리 |
| 교락 미분리 | 1 | K-sweep 이 중복 pack 조건에서만 측정됐다 |

**가장 값비쌌던 것**: 검증기를 나중으로 미룬 것. Phase 1a 를 세 번 무효화했다.
