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
