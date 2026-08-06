# 코드 맵 — 용어·개념이 소스 어디에 있는가

[08-glossary](../08-glossary.md) 와 [07-design](../07-design-block-parallel.md) 이 인용하는
개념을 `prefill-opt/src` 의 실제 파일·함수에 연결한다.

> **줄 번호는 참고값이다.** 함수 이름으로 먼저 찾을 것 (`grep -rn "함수명" src`).
> 확인 시점: 2026-08-06, `dllama` 빌드 기준.

---

## 0. 파일 한눈에

| 파일 | 줄수 | 역할 | 이 연구에서의 의미 |
|---|---|---|---|
| `src/llm.cpp` | 764 | **모델 그래프를 조립**한다 (레이어별 op 나열) | "어떤 연산이 어떤 순서로" 의 정본. KV 경로·배리어 위치가 전부 여기 |
| `src/nn/nn-cpu-ops.cpp` | 2074 | 각 op 의 **CPU 커널 구현** | 발견 #2·#4, lm_head 제한, 캘리브레이션 덤프가 전부 여기 |
| `src/nn/nn-network.cpp` | 2444 | 노드 간 소켓 통신, **배리어 실체** | syncWait/syncXfer 계측 |
| `src/app.cpp` | 2144 | root/worker 루프, controlPacket, 인자 파싱 | phase 브로드캐스트, 토폴로지 |
| `src/dllama.cpp` | 1002 | CLI 진입점, **prefill/decode 루프**, 지표 출력 | 문서의 모든 ms 값이 여기서 찍힌다 |
| `src/nn/nn-executor.cpp` | 463 | 스레드 풀, 세그먼트 실행, sync 호출 | 배리어가 "언제" 걸리는가 |
| `src/nn/nn-repack.cpp` | 730 | Q4_0 → SIMD 레이아웃 재배치 (llama.cpp 이식) | 발견 #4 |
| `src/nn/nn-core.cpp` | 393 | 텐서 크기·슬라이싱 계산 | `sliceKvCache` — 07 §1(c) 의 근거 |

---

## 1. 추론의 두 단계 — 08 Part 1

| 개념 | 위치 | 메모 |
|---|---|---|
| **prefill 루프** | `dllama.cpp:414~530` | 청크 단위로 `forward()` 반복. `useWave`, `prefillUsesSp` 분기가 여기 |
| **decode 루프** | `dllama.cpp:~586` 이후 | `setDecodePhase(true)` 이후 구간 |
| **prefill/decode 구분자** | `NnExecutor::setDecodePhase` → `nnCpuOpsSetDecodePhase` (`nn-executor.cpp:395`, `nn-cpu-ops.cpp:1280`) | 커널이 배치 경로와 단일 경로를 가르는 스위치 |
| **다중 노드 phase 전파** | `app.cpp:817` (`controlPacket.phase`) | root 가 정하고 워커가 받는다 → 모든 노드가 같은 판단 |
| **TTFT 출력** | `dllama.cpp:723` `ttftMs` | `hasFirstPredToken` 이면 실제 TTFT, 아니면 prefill wall |

### 문서의 지표 이름 → 코드 위치

07 §1 표의 컬럼이 정확히 여기서 나온다 (`dllama.cpp:707~722`):

| 문서 표기 | printf | 구성 |
|---|---|---|
| prefill | `prefillMs` | prefill wall clock |
| syncWait | `syncWaitMs` | **바이트가 안 움직인 대기** = straggler. H1 판정용 |
| syncXfer | `syncXferMs` | 실제 바이트 이동 |
| gemm | `gemmMs` | `attnProj + ffn + lmHead` 합 |
| attn | `attnMs` | O(S²) 항만 |

> `gemmMs` 는 합성값이다. **`attnProjMs`+`ffnMs`+`lmHeadMs` 와 중복 집계하지 말 것.**
> 계측은 `prefillOpProfilingOn` 일 때만 켜진다 (`dllama.cpp:451`).

---

## 2. KV 캐시 — 08 Part 1 / 05 Part A

| 개념 | 위치 |
|---|---|
| **KV 캐시 슬라이싱** | `nn-core.cpp:215` `sliceKvCache(kvDim, seqLen, nNodes, spSize, spRank)` |
| 호출부 | `llm.cpp:241` |
| K/V 버퍼 할당 | `llm.cpp:306~307` (`"k"`, `"v"`) |
| **KV 캐시에 쓰기** | `llm.cpp:410~418` `OP_SHIFT` (`block_shift_k`, `block_shift_v`) |
| 커널 | `nn-cpu-ops.cpp:1878` `shiftForward_F32_F32` |

> **07 §1(c) 의 핵심 근거가 `sliceKvCache` 다.** `localSeqStart/localSeqLen` 은
> KV **저장**만 시퀀스 방향으로 쪼갠다. 모든 노드가 같은 토큰을 계산한다 —
> 즉 `--sp-size` 는 메모리 최적화지 병렬화가 아니다.

### KV 크기 계산 (08 의 512 MB 표)

`run_one.sh:9~13` 주석이 실제 할당식을 적어 둔다:

```
size2D(F_32, seqLen, kvDim0) × 2(K,V) × nLayers
```

**`--max-seq-len` 전체로 잡힌다** — 실제 프롬프트 길이가 아니다. OOM #3 의 원인.

---

## 3. 레이어 하나의 연산 순서 — 05 A-1 / 08 §4-1

`llm.cpp` 의 `buildLlmNet` (179행~) 이 조립하는 순서. **이 목록이 곧 수식이다.**

```
[start 세그먼트]
  OP_EMBEDDING          llm.cpp:294

[레이어 l — att 세그먼트]
  OP_CAST     block_cast_x        315
  OP_MERGE_ADD block_merge_add    322     ← TP 조각 합치기
  OP_INV_RMS  block_norm_pre_0    330  ┐
  OP_RMS_NORM block_norm_0        336  ┘  = 08 의 φ(·)
  OP_MATMUL   block_matmul_q      350
  OP_MATMUL   block_matmul_k      356     ← K_l = W_k·φ(x_l).  08 §4-2 의 그 곱셈
  OP_MATMUL   block_matmul_v      362     ← V_l
  OP_ROPE     block_rope_q        395  ┐
  OP_ROPE     block_rope_k        403  ┘  ← 위치 의존. 08 Part1 "RoPE 적용 전" 의 경계
  OP_SHIFT    block_shift_k       411  ┐
  OP_SHIFT    block_shift_v       417  ┘  ← KV 캐시에 기록
  OP_MULTIHEAD_ATT block_multihead_att  427
  OP_MATMUL   block_matmul_wo     443
  ── addSync(SYNC_NODE_SLICES)    454     ← ★ 배리어 1/2
[레이어 l — ff 세그먼트]
  OP_MERGE_ADD block_merge_add2   458
  OP_INV_RMS / OP_RMS_NORM        464/470
  OP_MATMUL   block_matmul_w1     502  ┐
  OP_MATMUL   block_matmul_w3     508  ├  SwiGLU
  OP_SILU     block_act           514  ┘
  (w2)
  ── addSync(SYNC_NODE_SLICES)    605     ← ★ 배리어 2/2

[end 세그먼트]
  ── addSync(logits, SYNC_NODE_SLICES)  678
```

> **08 "TP 는 레이어마다 2번, 32레이어면 청크당 64번" 의 근거가 `llm.cpp:454` 와 `605` 다.**
> 두 줄이 그 주장 전부다.
>
> **08 "캘리브레이션 덤프를 사영 직후에 뜨는 이유"** 도 여기서 보인다 —
> `block_matmul_k`(356) 와 `block_rope_k`(403) 사이에서 떠야 위치 무관 값이 나온다.

---

## 4. 발견 #1~#4 — 02, 03, 06

| 발견 | 위치 | 무엇을 보면 되나 |
|---|---|---|
| **#1 dotprod** | `Makefile` 의 `-mtune=native`, `src/nn/llamafile/sgemm.cpp` | 컴파일 플래그 문제. 코드 자체는 정상 |
| **#2 attention 배치화** | `nn-cpu-ops.cpp:1695` `multiHeadAttForward_F32_F32` | 함수 안 `if (batchSize > 1u)` 분기(1710행 부근)가 수정의 전부. 주석이 이유를 적어 둠 |
| **#3 임베딩 F32** | `nn-cpu-ops.cpp:1117/1131` `embeddingForward_*` | `_F32_F32_Q80` 변형 유무 |
| **#4 Q4_0 repack** | `nn-repack.{hpp,cpp}` | 아래 상세 |

### 발견 #4 상세

| 항목 | 위치 |
|---|---|
| in-place 가능성 강제 | `nn-repack.hpp:39` `static_assert(sizeof(block_q4_0x4) == 4*sizeof(NnBlockQ40))` |
| 재배치 본체 | `nn-repack.hpp:64` `nnRepackQ40InPlace(weight, d, kBlocks)` |
| 지원 형상 판정 (폴백 경계) | `nn-repack.hpp:67` `nnRepackSupported(d, kBlocks)` |
| 활성화 변환 | `nn-repack.hpp:70` `nnPackQ80To4x4` |
| repack 커널 | `nn-repack.hpp:54/57` `ggml_gemv_q4_0_4x4_q8_0`, `ggml_gemm_q4_0_4x4_q8_0` |
| **로드 완료 훅** | `nn-cpu.cpp:224` `NnCpuDeviceSegment::loadWeight` → `loadedBytes` 누적(239) → 다 차면 repack(250~268) |
| 폴백 경로 | `nn-cpu-ops.cpp:1489` `matmulForward_Q80_Q40_F32` |

> 06 §3 의 "설계 판단 3가지" 중 (2) 로드 완료 감지가 `nn-cpu.cpp:239~268` 이다.
> root/worker 가 모두 이 경로를 지나므로 훅이 한 곳이면 된다.

---

## 5. lm_head prefill 제한 — 06 §4

| 항목 | 위치 |
|---|---|
| 행 범위 계산 | `nn-cpu-ops.cpp:1293` `lmHeadRowRange(context, batchSize, &rowBegin, &rowCount)` |
| 사용처 | `nn-cpu-ops.cpp:1311`, `1382` |
| phase 스위치 | `nn-cpu-ops.cpp:1280` `nnCpuOpsSetDecodePhase` |
| 전달 경로 | `nn-executor.cpp:395` → root `app.cpp:817` → 워커 |
| wave 예외 | `app.cpp:1035` `drainPrefillLogits` (로짓을 실제로 읽는 유일한 경로) |

> 06 §4 "남은 기회: 세그먼트 통째 스킵" 을 구현하려면 `llm.cpp:678` 의
> `end` 세그먼트 sync 를 건드려야 한다. `drainPrefillLogits` 와 충돌한다.

---

## 6. 분산 — 08 Part 3 / 07

| 개념 | 위치 |
|---|---|
| **배리어 실체** | `nn-network.cpp:1867` `NnNetworkNodeSynchronizer::sync` |
| 배리어 호출 지점 | `nn-executor.cpp:277` `context->synchronizer->sync(...)` |
| 단일 노드 무동작판 | `nn-executor.cpp:55` `NnFakeNodeSynchronizer::sync` |
| 집합 통신 | `nn-network.cpp:863` `writeMany` / `960` `readMany` |
| **wait vs xfer 분류** | `writeMany` 내부 `nn-network.cpp:877` 주석 | 바이트가 움직인 패스는 xfer, 아니면 wait |
| 지표 수집 | `nn-network.cpp:1101` `getSyncTimeBreakdown(waitUs, xferUs)` |
| 트래픽 분해 | `nn-network.cpp:1075` `getTrafficBreakdown` |
| 무한 출력 버그(06 §7) | `nn-network.cpp:1111` `printSocketTrafficSummary` |

> **08 "통신 시간의 대부분은 전송이 아니라 대기였다 (14.6초 vs 2.0초)"** 의
> 정의가 `nn-network.cpp:877` 주석에 있다. 이 분류 규칙을 모르면 숫자 해석이 틀린다.

### 토폴로지 인자

| 문서 표기 | CLI | 파싱 |
|---|---|---|
| TP | (기본, 노드 수) | `app.cpp` |
| PP | `--pp-size` | `app.cpp` |
| SP (dllama 의) | `--sp-size` | `dllama.cpp:932`, `app.cpp:438` |
| wave | `--wave-pipeline 0\|1` | `dllama.cpp:956`, `app.cpp:445` |
| 청크 | `--prefill-chunk-size`, `--prefill-chunk-threshold` | `dllama.cpp:954~955` |
| **블록 병렬** | `--block-parallel N` | **미구현** (07 A-1) |

---

## 7. 깊이 분해 — 08 Part 4 / 07 §4

08 Part 4 의 수학이 실제로 돌아가는 곳.

| 개념 | 위치 |
|---|---|
| **캘리브레이션 덤프** | `nn-cpu-ops.cpp:1425~1469`. 환경변수 `DLLAMA_CALIB_DIR` |
| 덤프 대상 판정 | `nn-cpu-ops.cpp:1452` `calibKindOf` — `block_matmul_k` / `block_matmul_v` 직후 |
| 덤프 실행 | `nn-cpu-ops.cpp:1462` `calibDumpAfterMatmul` |
| **최소제곱 적합** | `prefill_bench/fit_kv_projection.py` `ls_fit(X, Y, ridge)` |
| ridge(정칙화) | 같은 함수, `XtX[diag] += ridge * trace/h` |
| 재구성 오차 | 같은 파일 `rel_err(Y, Yhat)` |
| dense / 공유 저랭크 비교 | 같은 파일, `--k`, `--rank` 인자 |

실행:
```bash
DLLAMA_CALIB_DIR=/path/to/dump ./dllama inference ...
python3 prefill_bench/fit_kv_projection.py <calib_dir> --k 16 --rank 256
```

> 08 §4 를 읽다 막히면 이 스크립트가 가장 짧은 설명이다.
> `ls_fit` 6줄이 08 §4-3 의 `A_l* = argmin_A ‖A·φ(x_k) − K_l‖²` 전부다.

---

## 8. 벤치·운영 스크립트

| 파일 | 용도 | 문서 |
|---|---|---|
| `prefill_bench/run_one.sh` | **단일 노드 측정 1회 + OOM 가드 3종** | 06 §6 — 다중 노드도 반드시 이걸로 |
| `prefill_bench/run_breakdown.sh` | EXP-1 스윕 | 01 §3 |
| `prefill_bench/parse_breakdown.py` | 로그 → TSV | 01 §3 |
| `prefill_bench/make_prompts.py` | 목표 토큰 길이별 프롬프트 생성 | 01 §3 |
| `prefill_bench/start_workers.sh` | **워커 선기동** (root 보다 먼저) | 01 §3 |
| `prefill_bench/redeploy_workers.sh` | 워커 바이너리 재배포 | 06 §5 — stale 바이너리 사고 대응 |
| `prefill_bench/bench_sgemm.cpp` | GEMM 마이크로벤치 (110 GFLOPS 의 출처) | 00 §1, 02 §3 |
| `prefill_bench/bench_repack.cpp` | repack 마이크로벤치 (3.0~3.2×) | 06 §3 |
| `prefill_bench/test_repack.cpp` | repack 정확성 (rel.err ~1e-6) | 06 §3 |
| `prefill_bench/monitor_node.sh` | 온도·주파수 샘플러 | 01 §3 |

---

## 9. 코드를 처음 읽을 때의 순서

1. `llm.cpp:179` `buildLlmNet` — **여기부터.** 그래프가 곧 모델이다. op 이름만 훑어도 05 Part A 가 잡힌다.
2. `nn-cpu-ops.cpp` 에서 그 op 이름의 `*Forward_*` 함수 하나를 골라 읽는다 (`shiftForward` 가 짧다).
3. `dllama.cpp:414` prefill 루프 — 청크가 어떻게 흐르는가.
4. `nn-executor.cpp:277` — 배리어가 언제 걸리는가.
5. `nn-network.cpp:1867` / `863` / `960` — 배리어가 실제로 무엇을 하는가.

> 4→5 가 07 전체의 근거다. **배리어를 코드로 보기 전에는 07 §4 의 "12배 차이" 주장이 안 잡힌다.**
