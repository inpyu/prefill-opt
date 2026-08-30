# 10. 현재 상태

*기준일 2026-08-31. 일반성 스윕은 바이너리 `a6c22236`(SharedPack 확정본, 커밋 `c2fbe07`)로 돌렸다.*
*스윕 종료 후 attention 계측을 넣어 `6852f155` 를 빌드했고, 그 판으로 계측 pilot A 를 시작했다.*
*두 바이너리 모두 SSD 이전 후 새 환경에서 비트 단위 재현을 확인했다 —*
*[`ops/baseline/REPRODUCIBILITY.md`](/mnt/ssd/ops/baseline/REPRODUCIBILITY.md).*

> 진행률·조합 수 같은 숫자는 본문에 적지 않는다. `artifacts/generality/run.tsv` 가
> 1차 자료다. 본문 숫자는 곧 낡는다.

---

## 10.1 논문 구조 재편 (2026-08-25)

**논문 중심을 `Wave Pipeline + SharedPack` 의 계층적 co-design 으로 재구성했다.**
계획 탐색 절차와 효과 없던 병렬화 축들은 본문에서 내렸다.

| 구분 | 내용 |
|---|---|
| **본문 핵심** | Wave Pipeline (노드 간), SharedPack (노드 내), 둘의 co-design |
| 본문에 남김 | `B`(wave 실행 파라미터), `N`(스테이지 수·확장 곡선), 균등 layer partition, TP/CP 를 안 쓴 이유(짧게) |
| **본문에서 뺌** | 계획 탐색 절차 전반과, 측정에서 비활성으로 나온 병렬화 축들 (근거는 `research/16` §7·§12) |
| 옮긴 곳 | 이 폴더에서 삭제. 원 기록은 `research/16~21` 에 남아 있다 |

**시도했다가 도입하지 않은 것은 이 폴더에 두지 않는다.** 현재 설계를 지금 형태로
만든 트러블슈팅만 남긴다 — 그것은 [04](04-sharedpack.md) §5.2,
[07](07-design-space.md), [09](09-measurement.md) §7.5 에 있다.

---

## 10.2 서 있는 것

| 항목 | 수치 | 상태 |
|---|---|---|
| 공개 baseline — llama.cpp RPC (N=2) | 0.41× (느려진다) | 확정, 원인 소스 확인 |
| 공개 baseline — 공식 dllama (N=4) | 0.44× (느려진다) | 확정, 원인 소스 확인 |
| Wave Pipeline (`S_real` 447 / 1789) | **4.68× / 4.57×** | 확정 |
| Wave Pipeline — 자기 단일 노드 대비 | **6.64×**, 효율 83% | 확정 |
| `B` 선택 (`S_real` 447 / 1789) | 1.132× / 1.140× (앵커 기준) | 확정 |
| SharedPack (N=8 TTFT, `S_real=447`, B=32) | **1.436×**, 같은 wave 구성 BASE 와 비트 동일 | 확정 |
| SharedPack 메커니즘 — 1스레드에서 이득 없음 | **R = 0.9977** (−0.23%, 잡음 수준) | 확정 |
| 13B capacity | 8 GB 단독 노드에서 불가능한 모델을 클러스터에서 prefill | 확정 |

### 메커니즘 확인이 특히 중요하다

cache gate 는 *"중복 packing 이 없으면(스레드 1개) 이득은 0"* 을 예측한다.

| 스레드 | BASE | SP | R |
|---|---|---|---|
| 4 | 6,523.5 ms | 4,400.9 ms | **1.4823** |
| 1 | 13,588.3 ms | 13,619.5 ms | **0.9977** (−0.23%, 잡음 수준) |

**반증 가능한 예측이 맞았다.** 단일 커널에서 관측한 전이의 **두 끝점**(1스레드,
4스레드)이 N=8 분산 E2E 에서도 재현됐다. 2·3 스레드는 N=8 에서 아직 재지 않았다.

---

## 10.3 완료 — 일반성 스윕

`S_real {447, 1789, 7212} × B {16, 32} × threads {1, 4}`, N=8 wave on.
조합마다 `A,B,B,A` 를 연달아 돌려 같은 세션 안에서만 비율을 만든다.

결과 표는 [08-evaluation.md](08-evaluation.md) §8.4 에 있다. 요약하면:

- 4스레드 6조합 전부에서 이득 (1.1696~1.4823, 기하평균 1.3366)
- 1스레드 6조합 전부에서 이득 없음 (0.9890~1.0109, 기하평균 1.0016 — 잡음 수준)
- 프롬프트가 길수록 이득 감소 (447 → 1789 → 7212 에서 1.48 → 1.38 → 1.23). **미분해**
- **`B=16` vs `B=32` 는 차이 0.96% 로 log-SD ±1.4% 안. 구분되지 않는다**

> 이전 판에 *"B=16 이 B=32 보다 이득이 크다"* 고 적었으나 **철회한다.**
> 데이터가 그렇게 말하지 않는다.

harness `prefill_bench/generality.sh`, artifact `artifacts/generality/`.

## 10.4 다음 작업 — attention 계층 착수

병목이 이동했다. `S_real=7212` 에서 attention core 가 prefill 의 **약 53%** 이고,
projection 을 무한 가속해도 상한이 **1.78×** 다 ([05](05-attention-layer.md) §5.1).

### Phase 0~1 (지금)

| # | 작업 | 상태 |
|---|---|---|
| 0 | SharedPack 일반성 스윕 완료 + artifact 고정 | **완료** (48/48, `artifacts/generality/run.tsv`) |
| 0 | 기준선 두 개 고정: `BASE-A`(Wave+per-thread pack), `BASE-B`(Wave+SharedPack) | 대기 |
| 1 | **attention 내부 분해** — `DLLAMA_ATT_PHASE=1` 계측 | **빌드 완료**, pilot A 2/4 회차에서 중단 |

> 새 attention 의 비교 기준은 **`BASE-B`** 다. 그래야 SharedPack 과 attention
> 이득이 중복 계산되지 않는다.

### Phase 2~8

```
2  production stride 를 그대로 쓰는 AV microbenchmark (4경로 A/B/C/D)
3  append-friendly blocked V cache — 매번 transpose 하지 않는다
4  register-level GQA 공유 microkernel — (R,D,T) 를 register 예산에서 유도
5  QK·softmax 경로 동시 정리
6  계층적 정확성 검증 5단
7  2x2x2 ablation
8  일반성 (모델축 kvMul=1/4/더 큰 GQA, head dim 64/128/256)
```

**예상 3~4주.** 분해 1~2일, microkernel 3~5일, KV cache 통합 4~7일,
정확성·성능 디버깅 3~5일, 평가 4~7일.

### 확정된 우선순위 원칙

> **CTGHP, AxisCert, FamilyShard 같은 새 축을 동시에 벌리지 않는다.**
> 본문은 병목 이동 서사 하나로 유지한다 → [README](README.md)

FamilyShard 는 상한을 재봤더니 SharedPack 이후 family share 가 47.5% 라
`family r=1.5` 를 얻어야 E2E +18.8% 다. attention 쪽 상한이 훨씬 크므로
**지금은 열지 않는다.**

### 아직 남은 SharedPack 작업

| # | 작업 |
|---|---|
| a | 2×2 ablation (앵커) — harness `prefill_bench/ablation2x2.sh` 준비됨 |
| b | recurrence 로 1.436× 원인 분해 → [06](06-codesign.md) §6.3 |
| c | 최종 시스템 대 llama.cpp 동일 조건 재측정 (곱셈 금지를 푸는 유일한 길) |

## 10.4b 최종 ablation 은 2×2 가 아니라 2×2×2 다

세 기법이 있으므로 8개 조합이 원칙이다.

| Wave | SharedPack | RoleSplit | 의미 |
|---|---|---|---|
| 0 | 0 | 0 | 원본 기준 |
| 1 | 0 | 0 | Wave 단독 |
| 0 | 1 | 0 | SharedPack 단독 |
| 0 | 0 | 1 | Attention 단독 |
| 1 | 1 | 0 | **현재 DerivePP** |
| 1 | 0 | 1 | Wave–attention 상호작용 |
| 0 | 1 | 1 | 두 CPU 연산 최적화의 합 |
| 1 | 1 | 1 | 최종 시스템 |

모든 길이에서 8조합을 다 돌리면 비용이 크므로 나눈다.

```
S_real=447    8조합 전체
S_real=1789   핵심 6조합
S_real=7212   Wave+SP  대  Wave+SP+Attention 중심
13B           BASE 와 최종만
```

측정 항목: TTFT / QK·softmax·AV 시간 / projection 시간 / stage service time /
syncWait / L1·L2·LLC miss / 메모리 대역폭 / KV cache 용량 / decode latency /
정확성 계층 결과.

## 10.5 리뷰어가 찌를 지점

| 위험 | 현재 방어 | 필요한 것 |
|---|---|---|
| "1.436× 의 메커니즘이 설명 안 됐다" | 1스레드 R=0.9977 (원인이 중복 packing 임은 확인) | §10.4 의 2번 (critical path 까지) |
| "두 기법을 나열했을 뿐이다" | 예비 관측의 방향성 | §10.4 의 1번 (2×2 앵커) |
| "wave PP 는 GPipe 와 유사하다" | 저속·비대칭 링크라는 조건, 공개 baseline 0.41~0.44× | — |
| "shared packing 은 알려진 원리다" | 분산 파이프라인 E2E 측정, 계층 co-design | (선택) CTGHP |
| "`N` 축이 노드 구성과 교락됐다" | 문서에 명시 | `(N, π)` 계획 비교로 표기 |
| "단일 플랫폼·단일 모델" | 13B capacity | held-out 모델 검증 |
| "attention 최적화가 이미 시도돼 실패했다" | 실패 원인이 기록돼 있고 RoleSplit 이 그 원인을 겨냥한다 | [05](05-attention-layer.md) §5.3 |

---

## 10.6 절대 하지 말 것

```
4.57×  ×  1.436×  =  6.56×                       ← 곱하지 않는다
"pipeline amplification"                          ← 관측이지 설명이 아니다
"스테이지가 빨라져 bubble 이 줄었다"                ← 슬롯 수·비율이 준 게 아니다
"연산 1.20× + syncWait 1.77× 이득"                 ← 이중 계산
"다른 엣지 가속은 정확도를 희생하지만 우리는 아니다"  ← 근거 없음
"새로운 pipeline 알고리즘과 packing 알고리즘"       ← 개별 기법은 선행이 있다
```

---

## 10.7 하드웨어

| 노드 | 상태 |
|---|---|
| `.113 .191` (16 GB) | 정상 |
| `.54 .94 .96 .103 .104` (8 GB) | 정상 |
| **`.166`** | 네트워크·sshd 복구. **공개키 인증 미해결** — 재설치로 `authorized_keys` 소실 |
| `.116` | 존재하지 않음 (`.166` 의 오타) |

root 노드 SD 카드에 **선재하는 배드 섹터**. 실험이 원인은 아니고 모델 무결성은
확인됐다. **교체 권장.**
