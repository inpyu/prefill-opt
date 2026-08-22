# CPU Prefill 연산 경로 최적화 — QCFuse 기각과 SA-EDOT 제안

**상태: 측정 진행 중**  ·  작성 2026-08-22
관련: [16-derivepp.md](16-derivepp.md), [17-weight-compiled-prefill.md](17-weight-compiled-prefill.md)

> 우선순위 전환: 스케줄링 알고리즘이 아니라 **스케줄링 위에 올릴 CPU 연산**이 중심이다.
> DerivePP 는 분산 실행 기반으로 유지하되 `N`·`B`·partition 재선택 연구는 동결하고,
> 고정 실행 조건에서 순수 연산 이득을 측정한다.

---

## 1. op 별 시간 분해 (계측 기반)

`DLLAMA_OP_PROFILE=1` — executor 가 이미 재는 `stepTimeUs` 를 op 이름별로 누적한다
(추가 측정 비용 없음). `--stage-timing 1` 필요.

`S=447`, `B=32`, 단일 노드, wave off. total 34,346.5 ms.

| op | ms | % |
|---|---|---|
| **block_matmul_w2 (Down)** | **10,111.6** | **29.44%** |
| block_matmul_w3 (Up) | 6,761.4 | 19.69% |
| block_matmul_w1 (Gate) | 6,728.7 | 19.59% |
| block_multihead_att | 3,143.5 | 9.15% |
| block_matmul_q | 2,117.3 | 6.16% |
| block_matmul_wo | 2,044.9 | 5.95% |
| block_matmul_k / v | 1,131.7 | 3.30% |
| final_matmul_logits | 538.0 | 1.57% |
| block_mul | 386.3 | 1.12% |
| block_act (SiLU) | 224.1 | 0.65% |
| block_cast_d2 | 193.0 | 0.56% |
| block_cast_y3 | 37.8 | 0.11% |
| 나머지 (norm/rope/shift/merge/cast) | ~930 | ~2.7% |

**Q4×Q8 GEMM 합계 ≈ 84%.**

---

## 2. QCFuse graph-level materialization fusion — **기각**

제거 대상(`cast_y3 + act + mul + cast_d2`) 합계:

    386.3 + 224.1 + 193.0 + 37.8 = 841.2 ms = **2.45%**
    이상적 제거 상한 = 1/(1−0.0245) = **1.025×**

사전등록 기준(`<10% → 단독 20% 불가능`)에 크게 미달한다.

**왜 예상이 빗나갔나.** `B=32, hidden=14336` 이면 Gate/Up F32 버퍼가 각 1.75 MiB 이지만,
`mul`·`act`·`cast` 는 **원소당 1회 읽기/쓰기**뿐이다. 반면 GEMM 은 원소당 `K`번
(4096 또는 14336) 연산한다. 연산 강도가 3~4 자릿수 차이나서 materialization 이 GEMM 에
묻힌다.

### ⚠️ 기각의 범위를 한정한다

`block_cast_y3` 는 **F32→Q80 변환**이지 GEMM 내부의 **Q80→Q8×4 repack** 이 아니다.
현재 커널은 각 projection·각 스레드마다 activation 을 별도 repack 하며, 그 비용은
`block_matmul_w*` 시간 **안에** 들어 있다(`nn-cpu-ops.cpp:2012` 에 "각 스레드가 자기
스크래치에 전체를 중복 변환한다" 고 명시).

> **QCFuse 의 graph-level materialization 제거 가설은 기각됐다.
> GEMM 내부 shared packing 가능성은 아직 측정되지 않았다.**

---

## 3. Down 단독 최적화로는 20% 가 안 된다

Down 비중 29.44%. Gate/Up 수준(1.5×)으로 만들면

    E = 1 / (0.7056 + 0.2944/1.5) = **1.109×**

E2E 1.20× 를 Down 단독으로 내려면

    1/(0.7056 + 0.2944/r_D) = 1.20   →   r_D ≈ **2.31×**

현재 강한 SDOT baseline 에서 Down 단독 2.31× 는 가능성이 낮다.
**연구 방향을 "Down 전용" 으로 좁히면 안 된다.**

---

## 4. 새 중심 방향 — Shape-Adaptive Exact SDOT (SA-EDOT)

Down 의 1.5배 격차를 **출발점**으로 삼되, 목표는 모든 Q4×Q8 projection 에 적용되는
microkernel family 다.

```text
입력:  batch 크기, K, N, cache/register 제약
결정:  batch-row tile × output-column tile
       K panel 크기, activation packing lifetime
       weight prefetch 거리, SDOT/scale 명령 배치
출력:  기존과 bit-identical 한 GEMM
```

요청 단위 scheduling 이 아니라 **GEMM 내부의 연산·데이터 이동 알고리즘**이다.

GEMM 비중 84% 이므로 모든 projection 을 평균 1.25× 가속하면

    E = 1 / (0.16 + 0.84/1.25) ≈ **1.20×**

    Down 단독 목표          원인 규명용
    전체 Q4×Q8 GEMM 목표    가중평균 ≥ 1.25×
    최종 E2E 목표           ≥ 1.20×

---

## 5. Down 이 왜 느린가 — 대조 실험 설계

Gate/Up 과 Down 은 MAC 수가 거의 같고 형상만 반대다.

    Gate/Up : K=4096,  N=14336
    Down    : K=14336, N=4096

현재 고정 `16 batch row × 4 output col` 커널이 두 형상에 모두 쓰인다. 후보 원인:
긴 K 로 packed activation working set 증가 / activation scale·Q8 이 cache 에서 축출 /
긴 K loop dependency chain / prefetch 거리 불일치 / projection 마다 반복되는 repack /
output-column 분할이 Down 형상에서 불리.

| 축 | 값 |
|---|---|
| Batch | 4, 8, 16, 32 |
| Shape | 4096×14336, **중간 형상**, 14336×4096 |
| Thread | 1, 4 |
| 구간 | activation pack / GEMM core / epilogue **분리 계측** |
| 카운터 | cycle, instruction, L1/L2 refill, stall |
| 순서 | Gate→Down, Down→Gate **교차 실행** |

**중간 형상은 총 weight 원소 수를 같게 둔다** — 연산량을 고정하고 K 만 길어질 때
성능이 떨어지는지 분리한다.

---

## 6. 원인별 구현 분기 (측정 후 결정)

**A. activation repack 이 크다면** — `Q8×4` packed activation 을 projection family 가
공유한다. `pack(X)` 한 번으로 Gate·Up 을, Q·K·V 에도 동일 적용.

**B. 긴 K 가 cache 를 깨뜨린다면** — K 길이에 따라 batch tile 을 바꾸는 microkernel.
단순히 batch tile 을 줄이면 weight 를 반복 읽으므로 **cache miss 와 weight traffic 을
함께** 평가한다.

**C. SDOT dependency 가 병목이면** — 독립 accumulator chain interleave, weight
load·nibble unpack 선행, scale 변환과 다음 SDOT overlap, K loop software pipelining.

**D. 고정 tile geometry 가 문제면** — 같은 accumulator 예산 안에서 복수 variant
(`16×4` 기본, `32×2` weight reuse 우선, `8×8` 짧은 batch). **각 출력 원소의 K 누적
순서는 유지**해 bit-identical 을 보존한다.

---

## 6b. K-sweep 검증 — 예측 적중, 임계점 확인

`artifacts/saedot/k_sweep.tsv`. 총 `K×N` 을 최대한 고정하고 `K` 만 6단계로 늘렸다.
활성 재사용 단위는 **16-row panel** 이다(커널이 batch 를 16씩 처리).

    W_X(R=16, K) = (16/4) × (K/32) × 136 B
    K=4096 → 68 KiB,  K=8192 → 136 KiB,  K=14336 → 238 KiB

**`B=16` 과 `B=32` 의 활성 panel 이 같다** — 관측된 "둘 다 Down 확장 나쁨" 과 일치한다.
(앞서 `B=32` 를 476 KiB 로 계산한 것은 batch 16 단위 처리를 놓친 오류였다.)

### B=32 결과 (GOPS)

| shape | panel KiB | 1t | 2t | 3t | 4t | 4t/1t |
|---|---|---|---|---|---|---|
| K= 4096 N=14336 | 68 | 85.3 | 165.8 | 243.9 | 304.0 | 3.56 |
| K= 6144 N= 9556 | 102 | 84.4 | 167.2 | 244.6 | 293.1 | 3.47 |
| K= 8192 N= 7168 | 136 | 85.1 | 167.9 | 246.6 | 291.6 | 3.43 |
| K=10240 N= 5732 | 170 | 85.6 | 170.2 | 248.9 | 289.2 | 3.38 |
| K=12288 N= 4776 | 204 | 84.5 | 170.3 | 248.7 | **248.8** | 2.94 |
| K=14336 N= 4096 | 238 | 85.6 | 169.7 | 245.5 | **189.7** | 2.22 |

### 세 가지 확인

**1. 1·2·3 스레드는 `K` 와 완전히 무관하다.**
편차 1t 1.4%, 2t 2.7%, 3t 2.0%. `K` 가 3.5배 늘어도 성능이 그대로다 —
**K 길이 자체는 원인이 아니다.**

**2. 4스레드에서만, 특정 지점에서 꺾인다.**

    panel ≤ 170 KiB   289~304 GOPS   정상
    panel   204 KiB   248.8          −15%
    panel   238 KiB   189.7          −35%

**3. 임계점이 3↔4 스레드 사이에 있다.** `K=14336` 에서 3스레드는 245.5 로 정상인데
4스레드가 189.7 로 무너진다.

    3스레드 × 238 KiB = 714 KiB    L3 2 MB 에 여유
    4스레드 × 238 KiB = 952 KiB    + weight stream → 경합

`panel 204 KiB` 에서 4스레드 816 KiB 부터 꺾이는 것도 일관된다.

> **원인이 cache-residency 로 좁혀졌다.** cache miss counter 없이 "규명 완료" 라고
> 쓰지 않는다 — 관측은 "4스레드에서 panel 크기에 따라 단조 감소" 이고, 그것이 cache
> 경합과 **일관된다** 는 것까지다.

### KP-SDOT 목표 정량화

    현재  K=14336, 4t          189.7 GOPS
    목표  panel 128 블록(68 KiB) 제한  →  ~300 GOPS
    비율  1.58×    (사전등록 "강한 성공 ≥1.45×" 초과)
    E2E   1/(0.7056 + 0.2944/1.58) = **1.121×**

---

## 6c. 교락 분리 — 원인은 K 길이가 아니라 **스레드별 중복 pack**

`artifacts/saedot/dup_vs_shared.tsv`. 같은 조건에서 activation packing 방식만 바꿨다.
`dup` = production(스레드마다 전체를 자기 스크래치에 변환), `shared` = 하나를 공유.

### K=14336, B=32 스레드별

| threads | dup | shared | 비율 |
|---|---|---|---|
| 1 | 84.5 | 84.1 | 1.00 |
| 2 | 169.8 | 170.9 | 1.01 |
| 3 | 245.5 | 252.5 | 1.03 |
| **4** | **198.7** | **333.8** | **1.68** |

### 4스레드, B=32, panel 크기별

| panel KiB | dup | shared | 비율 |
|---|---|---|---|
| 68 | 318.4 | 329.6 | 1.04 |
| 102 | 312.3 | 330.9 | 1.06 |
| 136 | 305.9 | 327.1 | 1.07 |
| 170 | 297.5 | 327.4 | 1.10 |
| 204 | 252.9 | 330.9 | 1.31 |
| **238** | **198.7** | **333.8** | **1.68** |

### 결론

**`shared` 는 K 와 완전히 무관하다** — 68 KiB 부터 238 KiB 까지 327~334 GOPS 로 평탄하다.
`dup` 만 panel 크기에 따라 318 → 199 로 단조 감소한다. 1~3스레드는 차이가 없다(1.00~1.03).

> 기존 구현은 네 스레드가 동일한 activation 을 **서로 다른 주소에 각각 pack** 하여 네 개의
> 독립된 cache footprint 를 만든다. 큰 K 에서는 이 footprint 가 private L2 와 shared
> cache/memory path 를 압박해 4스레드 확장성을 무너뜨린다. 하나의 packed activation 을
> 공유하면 `K=14336` 에서도 333.8 GOPS 로 회복된다.

("트래픽이 정확히 4배" 는 부정확하다 — 논리적 read 횟수는 shared 에서도 네 스레드에
남는다. 차이는 **동일 cache line 을 공유하느냐, 서로 다른 네 사본을 읽느냐** 다.)

### ⚠️ 앞선 K-sweep 해석 철회

    철회:  "긴 K 의 16-row activation panel 이 L2 에 안 들어간다"
    정정:  panel 크기 자체는 무관하다. 공유하면 238 KiB 에서도 평탄하다.
           K-sweep 은 **중복 pack 조건에서만** 측정됐으므로 K 효과와 교락돼 있었다.

### ⚠️ KP-SDOT 중단

`333.8` 이 238 KiB panel 에서 나오므로 **K-panelization 은 존재하지 않는 문제를 푼다.**

다만 KP-SDOT 프로토타입의 부산물은 보존한다 — accumulator 저장·복원으로
**K-panel continuation 이 bit-identical 함을 확인**했다(P=64/128/224 모두 YES).
production 우선순위에서는 내려놓되, 정확한 panel 분할이 가능하다는 보조 결과다.

### ⚠️ `shared=333.8` 은 kernel upper bound 다

측정에서 shared buffer 를 타이밍 루프 **밖에서** 생성했다. 따라서

    T_shared = T_pack-once + T_barrier + T_GEMM

중 `T_GEMM` 만 잰 값이다. production 예상치로 바로 쓸 수 없으며, 별도 pack op 를
넣은 최종 측정에서 pack 과 executor boundary 까지 포함해야 한다.

---

## 6d. SharedPack-SDOT — 구현 구조

### 두 단계 공유

**1. Inter-thread sharing** — 하나의 GEMM 안에서 네 스레드가 동일 packed activation 공유.
Down 의 1.68× 는 주로 이 효과다.

**2. Inter-projection sharing** — 입력이 같은 projection 들이 재사용.

    Attention   pack(xq) 1회 → Q, K, V
    FFN         pack(yq) 1회 → Gate, Up
    Down        pack(dq) 1회 → Down
    O           별도 입력 → 별도 pack

    기존:  7 projection × 4 private copies
    변경:  QKV 1 + Gate/Up 1 + Down 1 + O 1

Inter-thread 부터 구현하고, buffer lifetime 을 늘려 projection-family sharing 을 추가한다.

### 구조 — 별도 `OP_PACK_Q80X4`

    Q80 activation
           │
           ▼
    OP_PACK_Q80X4        네 스레드가 shared buffer 의 서로 다른
                         batch-row group 범위를 병렬로 작성
           │
           ▼             executor op boundary = barrier
    shared Q8x4 activation
      ├─ thread 0: GEMM output columns 0
      ├─ ...

**한 스레드만 pack 하게 하지 않는다.** 기존 executor 의 op boundary 가 필요한
synchronization 을 제공한다. shared buffer 는 `thread_local` 이나 전역 static 이 아니라
**실행 context 가 소유**해야 한다.

### 처리해야 할 예외

- 마지막 microbatch 가 4행 미만이면 기존 Q80 GEMV fallback 유지
- `lm_head` 의 마지막 행 범위는 초기 적용 대상에서 제외
- shared packed buffer 의 stride 와 `rowBegin` 을 명시적으로 전달
- wave pipeline 에서 서로 다른 실행이 같은 workspace 를 덮어쓰지 않도록 context 별
  generation 관리
- `DLLAMA_REPACK=0` 경로 유지
- Q80 원본은 fallback 이 필요하므로 첫 구현에서 보존

### 예상 E2E

    Down              29.44% / 1.68
    Gate+Up           39.28% / 1.04
    Q/K/V/O GEMM      15.41% / 1.04
    non-GEMM          15.87%

    T_new ≈ 0.2944/1.68 + 0.3928/1.04 + 0.1541/1.04 + 0.1587 ≈ 0.860
    E2E   ≈ **1.16×**

초기 목표는 **1.15~1.17×**. 20% 까지 남는 것은 약 2.7 %p 이므로, SharedPack 성공 후
전체 GEMM 에 **추가 4% 정도**면 도달한다.

### 사전등록 기준

| 항목 | 기준 |
|---|---|
| Down kernel | ≥1.45×, 목표 ≥1.60× |
| Gate/Up | 비열화 금지, 가능하면 ≥1.03× |
| 전체 projection 가중평균 | ≥1.15× |
| 단일 노드 E2E | ≥1.12× 1차, ≥1.15× 강한 성공 |
| 정확성 | packed bytes 및 GEMM output **bit-identical** |
| 최종 목표 | 후속 최적화 포함 E2E ≥1.20× |

### novelty 는 정직하게

GEMM 의 shared input packing 자체는 전통적 BLAS 기법이다. **이것만으로 새 알고리즘이라
주장할 수 없다.** 강한 형태는 다음 단계까지 포함해야 한다.

> 여러 CPU 스레드와 연속 projection 이 하나의 quantized activation panel 을 공유하고,
> F32→Q80 생성 단계에서 **downstream SDOT layout 을 직접 산출**하는 exact
> packed-activation dataflow.

---

## 7. 진행 순서

    1. QCFuse intermediate fusion 기각 기록          ← 완료
    2. B=16 materialization tax 확인용 1회
    3. GEMM 내부 nnPackQ80To4x4 시간 별도 계측
    4. Down/Gate 형상 교차 microbenchmark (중간 형상 포함)
    5. PMU 기준으로 Down 1.5배 격차 원인 확정
    6. SA-EDOT microkernel variant 하나 구현
    7. 전체 projection 가중평균 1.25× gate
    8. bit-identical 및 E2E 1.20× 검증
