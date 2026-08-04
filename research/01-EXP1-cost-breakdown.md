# EXP-1: Prefill Cost Breakdown

**목적**: 가설 H1(통신/대기 비중)과 H2(attention 비지배)를 검증하고, 논문 Figure 1을 만든다.

---

## 1. 측정 대상

prefill 벽시계 시간을 4개로 분해한다.

| 구분 | 정의 | 현재 계측 여부 |
|---|---|---|
| **T_gemm** | projection + MLP GEMM 실행 시간 | `STEP_EXECUTE_OP` 에 포함 (세분화 필요) |
| **T_attn** | attention score/softmax/AV — O(S²) 항 | ✗ **추가 계측 필요** |
| **T_xfer** | 실제 소켓 송수신 시간 | `STEP_SYNC_NODES` 에 포함 (분리 필요) |
| **T_wait** | barrier에서 다른 노드를 기다린 시간 (straggler) | ✗ **추가 계측 필요 — 가장 중요** |

현재 `src/dllama.cpp` 는 `Eval / Sync` 2분할과 traffic breakdown(KV tx/rx, Act tx/rx)까지만 출력한다.
**T_wait 를 T_xfer 에서 분리하는 것이 이 실험의 핵심이다.** 이 값이 크면 H1 성립 → 통신량이 아니라
동기 구조가 문제라는 뜻이고, 논문의 방향이 확정된다.

### 코드 변경 — **적용 완료 (2026-08-04)**

1. **`src/nn/nn-network.cpp/.hpp` — sync를 wait/xfer로 분리** ✔
   `readMany`/`writeMany`의 busy-poll 루프에서, 바이트가 실제로 움직인 패스는 `syncXferUs`,
   진행이 없던 패스와 재시도 sleep은 `syncWaitUs`로 적산.
   API: `getSyncTimeBreakdown()` / `resetSyncTimeBreakdown()`.
   > 실행기 단계(`STEP_SYNC_NODES`)를 쪼개는 대신 소켓 계층에서 나눴다. 실제 대기가 발생하는
   > 지점이 여기이고, 스텝 타입을 늘리면 `N_STEP_TYPES` 의존 코드가 전부 흔들린다.

2. **`src/nn/nn-executor.cpp/.hpp` — attention score를 projection에서 분리** ✔
   기존 `getLastForwardOpBreakdown()`이 Q/K/V/O projection과 attention score를 `attnUs`로
   함께 묶고 있었다. `multihead_att`·`softmax`만 걸러 `attnCoreUs`(= O(S²) 항) 필드를 신설.

3. **`src/dllama.cpp` — prefill 구간 적산 및 출력** ✔
   `--stage-timing 1` 일 때 root에서도 step profiling을 켜고, 청크 forward마다 누적.
   Timing 블록에 다음을 출력한다.
   ```
    syncWaitMs / syncXferMs          ← H1
        attnMs (= attnCore, O(S²))   ← H2
        gemmMs (= attnProj+ffn+lmHead)
      attnProjMs / ffnMs / normMs / lmHeadMs / otherMs
   ```
   > wave 모드(`--wave-pipeline`)는 forward가 비동기라 op breakdown 집계에서 제외된다.

`parse_breakdown.py`가 위 필드를 전부 읽어 `wait_frac`, `xfer_frac`, `attn_frac`,
`attn_over_gemm`을 자동 산출한다.

---

## 2. 스윕 설계

| 축 | 값 | 이유 |
|---|---|---|
| 프롬프트 길이 S | 128 / 512 / 2048 / 8192 | H2의 O(S²) 전환점 탐색 |
| 노드 수 N | 1 / 2 / 4 / 8 | 통신 스케일링, straggler 누적 |
| 대역폭 W | 1GbE / 100Mbps / 2.5GbE* | 체제 경계 탐색 (`net_shape.sh` 로 tc 제한) |
| 모델 | llama3-8b_q40 (주), qwen3-8b_q40 (교차검증) | 모델 형상 의존성 |
| 반복 | 3회 | 열 스로틀링 분산 확인 |

\* 2.5GbE는 하드웨어 없으면 생략, 100Mbps 제한으로 반대편 체제만 확보해도 충분.

**총 실행 수**: 4(S) × 4(N) × 2(W) × 3(rep) = 96 run (모델 1종 기준). 8B q40 로딩 시간 포함 약 3~4시간.

---

## 3. 실행 절차

```bash
cd /home/ubuntu/prefill-opt/prefill_bench

# 0) 설정 확인 (노드 IP, 모델 경로, 스윕 범위)
vim env.sh

# 1) 프롬프트 생성 (목표 토큰 길이별)
python3 make_prompts.py --lengths 128,512,2048,8192 --out ../prompts_gen

# 2) 워커 선기동 (worker-first 규칙 — 반드시 root보다 먼저)
bash ../scripts/rpi_worker_run.sh

# 3) 스윕 실행 (노드별 온도/주파수 샘플러 자동 동반)
bash run_breakdown.sh

# 4) 로그 → TSV
python3 parse_breakdown.py ../bench_prefill/<RUN_ID> -o ../bench_prefill/<RUN_ID>/breakdown.tsv
```

---

## 4. 산출물

### Figure 1 (논문 첫 그림) — Stacked bar: S × N 에 따른 prefill 시간 4분할

읽어내야 하는 것:
- **T_wait / (T_wait + T_xfer)** 비율 → H1. 이게 0.5 이상이면 "대역폭이 아니라 동기 구조가 문제"
- **T_attn / T_gemm** 이 S에 따라 어떻게 변하는가 → H2. S=2048 에서도 T_attn < T_gemm 이면 H2 성립
- N=1 → 8 스케일링 효율 곡선 → 이상적 N배 대비 실제 몇 배인가

### Table — 노드별 straggler 분산

`monitor_node.sh` 가 수집한 온도/주파수와 노드별 T_gemm 을 대조.
스로틀링이 straggler의 원인인지, OS 스케줄링인지 구분한다.

---

## 5. 판정 기준

| 결과 | 다음 행동 |
|---|---|
| T_wait 비중 > 20 % | H1 성립. 동기 구조 제거(블록 독립 인코딩) 노선 확정 |
| T_wait 비중 < 10 % | H1 반증. 통신 최적화 접고 연산 축(H3/depth 분해)으로 |
| S=2048 에서 T_attn < 0.3 × T_gemm | H2 성립. **"짧은 프롬프트에서는 attention이 병목이 아니다"** → 분할 축 재선택이 논문 메인 |
| S=2048 에서 T_attn 지배 | H2 반증. 기존 블록 독립 인코딩 노선이 그대로 유효 |
| 노드별 T_gemm 편차 > 15 % | 이종/열 인지 스케줄링이 별도 기여점으로 성립 |

---

## 5.5. 예비 측정 (2026-08-04, 단일 노드 스모크 테스트)

계측 검증용 1회 실행. **클러스터가 아니라 개발 머신(4코어) 단일 노드**이므로
절대값은 의미가 없고, **op 구성비만** 참고한다.

- 모델 llama3-8b_q40, 프롬프트 110 토큰(실측), 4 chunk, `--stage-timing 1`

| 항목 | 시간 | prefill 대비 |
|---|---|---|
| prefill 전체 | 56,946 ms | 100 % |
| **attn (O(S²))** | **277 ms** | **0.49 %** |
| gemm 합계 | 56,312 ms | 98.9 % |
| ├ ffn | 43,352 ms | 76.1 % |
| ├ attn projection | 9,576 ms | 16.8 % |
| └ **lm_head** | **3,384 ms** | **5.9 %** |
| norm | 36 ms | 0.06 % |

**관찰 1 — H2를 강하게 지지한다.** `attn/gemm = 0.005`. S=110에서 O(S²) 항은 사실상 없고,
비용은 전적으로 **FFN(76%)** 이 지배한다. 판정 기준(< 0.30)을 두 자릿수 차이로 통과했다.
다만 이건 S가 가장 짧은 지점이므로, **S=2048/8192에서 이 비율이 어디까지 올라가는지가
실제 판정 대상**이다. 전환점이 실사용 구간 밖에 있으면 H2가 논문의 축이 된다.

**관찰 2 — lm_head 5.9%는 그대로 낭비다.** prefill에서 로짓이 필요한 토큰은 마지막 1개뿐인데,
현재는 모든 위치에 대해 lm_head를 계산하고 있다. 정확도 손실 없는 순수 이득이라
**즉시 적용 가능한 baseline 개선**이고, 동시에 "TTFT 산출물은 KV 캐시 + 마지막 토큰 로짓뿐"
이라는 연구 계획 §1의 전제를 코드로 확인해 준 셈이다.
→ 별도 항목으로 분리: `research/02-lmhead-prefill-prune.md` (예정)

**아직 모르는 것**: `wait_frac`/`xfer_frac`은 단일 노드라 0이다. **H1은 다중 노드 실행이
있어야 판정 가능**하며, 그것이 EXP-1 본 스윕의 핵심이다.

---

## 6. 주의사항 (기존 운영 규칙 준수)

- **worker-first**: sub 노드 워커를 항상 root보다 먼저 기동
- 로그는 `bench_prefill/<RUN_ID>/` 아래에만 저장 (`bench_logs/` 는 PiPP 실험용이므로 건드리지 않음)
- governor는 `performance` 고정, 각 run 사이 **60초 쿨다운**으로 열 조건 정렬
- 모델/토크나이저는 `layer-skip-bypass` 로의 심볼릭 링크 — 절대 복사하지 말 것
