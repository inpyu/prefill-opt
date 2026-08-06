# Baseline 현황 — 어디까지 왔고 무엇이 남았나

이 문서는 "우리 구현이 성숙한 엔진(llama.cpp)에 준하는가"를 추적한다.
**여기 있는 항목은 전부 논문 기여가 아니라 baseline 위생이다.**
왜 그런지는 [03-attention-batching.md](03-attention-batching.md) §4 참조.

---

## 1. 외부 기준선

llama.cpp (build 6ea215d), 같은 노드 / 같은 모델 계열 / Q4_0 / 4스레드:

| | tok/s |
|---|---|
| pp512 | **16.15 ± 0.14** |
| pp2048 | **15.27 ± 0.43** |

> 최초 측정 17.91 은 편차 ±1.78 로 불안정했다. 재측정값을 기준으로 쓴다.

## 2. 단일 노드 진행 (llama3-8b_q40, Cortex-A76 4스레드, S=447)

| 단계 | prefill | tok/s | llama.cpp 대비 | 문서 |
|---|---|---|---|---|
| 원본 | 496 ms/tok | 2.01 | 12 % | — |
| + dotprod 활성화 | 78,150 ms | 5.72 | 35 % | [02](02-baseline-dotprod-fix.md) |
| + attention 배치화 | 72,938 ms | 6.13 | 38 % | [03](03-attention-batching.md) |
| + Q4_0 repack | 35,889 ms | 12.45 | 77 % | 아래 §3 |
| **+ lm_head prefill 제한** | **32,417 ms** | **13.79** | **85 %** | 아래 §4 |

누적 **6.9×**. 네 항목 모두 **출력 byte-for-byte 동일**을 확인했다.

## 3. 발견 #4 — Q4_0 가중치 repack 부재

llama.cpp 는 로드 시점에 Q4_0 을 SIMD 친화 레이아웃으로 재배치한다
(`ggml/src/ggml-cpu/repack.cpp:4583`, Cortex-A76 은 `q4_0_4x4_q8_0` 경로).
distributed-llama 가 벤더링한 llamafile sgemm 에는 이 개념이 없다.

기존 커널은 (i,j,l) 조합마다:

| 용도 | 명령어 |
|---|---|
| 4비트 언팩 (load/and/shr/sub) | ~6 |
| 실제 연산 (vdotq_s32) | 2 |
| 누적 (cvt, mla) | 2 |
| 스케일 (fp16 스칼라 곱) | ~1 |

유효 64 FLOP / 명령어 11개 ≈ 5.8 FLOP/instr → 이론 111 GFLOPS.
**실측 97~110 GFLOPS 와 정확히 일치한다.**

repack 이 바꾸는 것:
1. **zero-point(-8) 를 pack 시점에 XOR(`0x88888888`) 로 흡수** → 런타임 `vsubq_s8` 제거
2. **출력 행 4개를 인터리브** → 로드 1회가 4행을 커버 (`vdotq_laneq_s32`)
3. 스케일 `d[4]` 가 연속 → `vmulq_laneq_f32` 로 벡터화

### 이식 (src/nn/nn-repack.{hpp,cpp}, 출처 llama.cpp MIT)

마이크로벤치 (`prefill_bench/bench_repack.cpp`), 4스레드:

| shape | batch 32 기존 | batch 32 repack | 배율 |
|---|---|---|---|
| qkv/o (4096×4096) | 105.2 | 317.6 | 3.02× |
| ffn_w1 (14336×4096) | 106.2 | 307.0 | 2.89× |
| ffn_w2 (4096×14336) | 103.2 | 331.3 | 3.21× |

정확성 (`prefill_bench/test_repack.cpp`): gemm/gemv 모두 **L2 상대오차 ~1e-6**
(양자화 차이가 아니라 fp32 누적 순서 차이 수준).

### 통합 설계 판단 3가지

1. **in-place repack** — `NnBlockQ40 4개(72B) == block_q4_0x4(72B)` 라 추가 메모리 0.
   `static_assert` 로 강제한다.
   또한 행 그룹 g 의 원본 바이트 범위와 목적지 범위가 정확히 겹치므로
   그룹 하나 분량(kBlocks×72 B)만 임시로 잡으면 된다.
   (초기 구현은 op 마다 가중치 전체 사본을 잡았다 — lm_head 는 295 MB 로 Pi5 8GB 에서 위험)
2. **로드 완료 감지** — `NnCpuDeviceSegment::loadWeight` 에서 `loadedBytes` 누적.
   root/worker 가 모두 이 경로를 지나므로 훅을 두 곳에 만들 필요가 없다.
3. **배리어 없는 스레드 분할** — 실행기는 op 경계에서만 동기화한다.
   출력 **열**로 나누고 각 스레드가 활성화 변환을 자기 `thread_local` 스크래치에 중복 수행.
   - 중복 비용: batch 32 / k 4096 기준 스레드당 ~139 kB 셔플
   - **행**으로 나누면 배리어는 불필요하지만 모든 스레드가 가중치 전체(FFN w1 28 MB)를
     읽어 트래픽이 nThreads 배. 연산 11.4 ms vs 트래픽 14 ms 로 역전되어 기각.

폴백은 유지된다 — repack 미지원 형상(d 가 4의 배수가 아님)이나 비-aarch64 는
기존 `matmulForward_llamafile` 로 자동 폴백. decode(batch=1)는 gemv 가
평범한 `NnBlockQ80` 을 그대로 받으므로 활성화 변환 없이 동작한다.

### 실측 대비 마이크로벤치 갭

마이크로벤치 3.1× 인데 실측 GEMM 은 2.2× 다. 후보:
- 활성화 변환의 스레드별 중복
- GEMM 이 3배 빨라지며 드러난 메모리 대역폭 한계

## 4. lm_head prefill 제한

**근거**: prefill 루프는 위치 0..n-2 만 처리하고 `forward()` 후 로짓을 **전혀 읽지 않는다.**
decode 첫 스텝이 마지막 입력 토큰을 처리해 첫 출력 로짓을 만든다.
즉 prefill 의 lm_head 계산은 전부 낭비다.

**구현**: 그래프 형태와 sync 는 그대로 두고 행렬곱 행 범위만 제한
(`lmHeadRowRange`, `nn-cpu-ops.cpp`). phase 는
`NnExecutor::setDecodePhase` → `nnCpuOpsSetDecodePhase` 로 전달된다.

**다중 노드 정합성 확인됨**: `controlPacket.phase` 가 root→worker 브로드캐스트되고
(`app.cpp:1242`), 워커도 매 forward 마다 `executor.setDecodePhase()` 를 호출한다
(`app.cpp:2070`). 따라서 모든 노드가 같은 판단을 한다.

### 남은 기회: 세그먼트 통째 스킵

현재는 청크마다 마지막 행 1개를 계산하는데, 그 한 행 때문에
**lm_head 가중치 295 MB 를 청크마다 스트리밍**한다 (메모리 바운드, 청크당 ~32 ms).

`end` 세그먼트를 prefill 동안 통째로 건너뛰면:

| | 연산 절감 | logits sync 절감 |
|---|---|---|
| 행 제한 (현재) | 97 % | ✘ |
| 세그먼트 스킵 | 100 % | ✔ **청크당 ~16 MB** (batch 32 × vocab 128256 × 4 B) |

1GbE 에서 청크당 0.14 초다. **다중 노드 단계에서 승격할 것.**
(단 wave 모드는 `drainPrefillLogits` 가 로짓을 읽으므로 예외 처리 필요)

## 5. 남은 항목

### 단일 노드 (수확 체감)

| 항목 | 예상 |
|---|---|
| lm_head 완전 스킵 | 1.4 % |
| KV 캐시 F32 → F16 | attention 트래픽 2× |
| online softmax 융합 (`att` 중간버퍼 8.4 MB/청크 실체화 제거) | attention 2~3× |
| 활성화 변환 중복 제거 | GEMM 2.2× → 3.1× 근접 |
| 토큰 임베딩 F32 → 양자화 | RAM 1.8 GB. **성능이 아니라 CP 노선을 여는 것** |

### 다중 노드 — **아직 시작도 안 함. 이것이 진짜 남은 baseline**

| 항목 | 왜 필수 |
|---|---|
| **워커 재배포** | 워커에 구버전 바이너리가 남아 있다. 지금 다중 노드를 돌리면 워커가 6.9× 느려 전부 straggler 가 되고 `wait_frac` 이 오염된다 |
| **다중 노드 정확성** | repack·lm_head 변경이 TP/PP 구성에서도 무손실인지 미확인 |
| **EXP-1 스윕** | `wait_wait`/`xfer_frac` 계측을 만들어놓고 한 번도 실측하지 못했다. H1 판정 미완 |

> **분산 논문의 baseline 은 분산 baseline이다.** 단일 노드 연산만 고쳐놓고
> 완료로 간주하면, 다중 노드에서 새 구현 문제가 나올 때 측정을 전부 다시 해야 한다.

## 6. 운영 교훈

`prefill_bench/run_one.sh` 에 가드로 고정했다. **다중 노드 실험은 반드시 이 스크립트로 돌릴 것.**

| 사고 | 원인 | 가드 |
|---|---|---|
| OOM #1 | 백그라운드 실행이 죽은 줄 알고 두 번째를 띄움 → 8B 모델 2개 동시 로드 | `pgrep -f "dllama[^ ]* +(inference\|worker)"` — 이름 변형까지 탐지 |
| OOM #2 | 추론 중에 빌드를 동시 실행 | 시작 전 가용 메모리 ≥ 8 GB 확인 |
| OOM #3 | KV 캐시가 실제 프롬프트가 아니라 `--max-seq-len` 전체로 할당됨 (8B/8192 이면 KV 만 2.1 GB) | 프롬프트 길이에서 `--max-seq-len` 자동 산정 |
| **디스크 19.5 GB 소진** | 종료 시 트래픽 요약이 무한 출력 (아래 §7) | 로그 200 MB 상한 감시 + 코드 측 출력 횟수 제한 |
| 좀비 프로세스 | 이전 세션의 `dllama inference` 가 살아남아 자원 점유 | 위 중복 실행 가드가 탐지 |

### 진단할 때의 규칙 (이번에 크게 데임)

1. **한 번에 하나만 바꾼다.** 바이너리·프롬프트 길이·repack 유무를 동시에 바꿔
   원인 추적이 세 바퀴 돌았다.
2. **바이너리 신원을 매번 확인한다.** 빌드보다 먼저 복사한 stale 바이너리를 실행해
   "고쳤는데 왜 그대로지"를 반복했다. root/worker `md5sum` 대조를 실행 전에 찍을 것.
3. **로그는 크기부터 본다.** `grep` 으로만 확인해 5.5 GB 로그를 놓쳤다.

## 7. 알려진 기존 버그 (distributed-llama 원본)

### 종료 시 트래픽 요약 무한 출력

정상 추론이 끝난 뒤 종료 경로에서:

```
📦 [NET_TRAFFIC] root<->worker[1] socket=0 tx_total=648B ...   ← 정상
📦 [NET_TRAFFIC] invalid socket=1 for root<->worker[2] (nSockets=1)
📦 [NET_TRAFFIC] invalid socket=2 for root<->worker[3] (nSockets=1)
...                                                            무한 증가
```

호출부(`app.cpp:1963`, `app.cpp:2126`)는 둘 다 `i < network->nSockets` 로 경계가
잡혀 있는데 인덱스가 무한 증가했다. **원인 미확정**(종료 시점의 객체 수명 문제로 추정).

한 번의 실행에서 로그가 14 GB, 다른 실행에서 5.5 GB 까지 자랐다.
**추론 결과 자체는 정상**이므로 측정값은 유효하다.

대응: `printSocketTrafficSummary` 의 invalid 분기 출력을 4회로 제한하고,
`run_one.sh` 에서 로그 크기를 감시한다. 근본 원인은 다중 노드 작업이 안정된 뒤 추적할 것.
