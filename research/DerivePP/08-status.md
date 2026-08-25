# 08. 현재 상태

*기준일 2026-08-25. 커밋 `ae2b69e`+, 바이너리 `a6c22236`.*

---

## 8.1 논문 구조 재편 (2026-08-25)

**AxisCert 와 여러 축을 논문 중심에서 내리고, `Wave Pipeline + SharedPack` 의
계층적 co-design 으로 재구성했다.**

| 구분 | 내용 |
|---|---|
| **본문 핵심** | Wave Pipeline (노드 간), SharedPack (노드 내), 둘의 co-design |
| 본문에 남김 | `B`(wave 실행 파라미터), `N`(스테이지 수·확장 곡선), 균등 layer partition, TP/CP 를 안 쓴 이유(짧게) |
| **본문에서 뺌** | `p` sublayer partition DP, `r` CoRePP core split, `g` CP2 island, `π` placement, potential, 축별 `δ_x`, plateau, completion-vector Pareto DP 상세 |
| 옮긴 곳 | Appendix / design-space 절 / negative-result 문서 |

이유는 [06-not-adopted-axiscert.md](06-not-adopted-axiscert.md) 에 있다.

---

## 8.2 서 있는 것

| 항목 | 수치 | 상태 |
|---|---|---|
| 공개 baseline — llama.cpp RPC (N=2) | 0.41× (느려진다) | 확정, 원인 소스 확인 |
| 공개 baseline — 공식 dllama (N=4) | 0.44× (느려진다) | 확정, 원인 소스 확인 |
| Wave Pipeline (S=447 / S=1789) | **4.68× / 4.57×** | 확정 |
| Wave Pipeline — 자기 단일 노드 대비 | **6.64×**, 효율 83% | 확정 |
| `B` 선택 (S=447 / S=1789) | 1.132× / 1.140× | 확정 |
| SharedPack (N=8 TTFT, S=447, B=32) | **1.436×**, 로짓 비트 동일 | 확정 |
| SharedPack 메커니즘 — 1스레드에서 이득 0 | **R = 0.9977** | 확정 |
| 13B capacity | 8 GB 단독 노드에서 불가능한 모델을 클러스터에서 prefill | 확정 |

### 메커니즘 확인이 특히 중요하다

cache gate 는 *"중복 packing 이 없으면(스레드 1개) 이득은 0"* 을 예측한다.

| 스레드 | BASE | SP | R |
|---|---|---|---|
| 4 | 6,523.5 ms | 4,400.9 ms | **1.4823** |
| 1 | 13,588.3 ms | 13,619.5 ms | **0.9977** |

**반증 가능한 예측이 맞았다.** 단일 커널 마이크로벤치의 전이
(1~3스레드 1.00~1.03×, 4스레드 1.68×)가 **N=8 분산 E2E 에서 그대로 재현된다.**

---

## 8.3 진행 중 — 일반성 스윕

`prompt {447, 1789, 7212} × B {16, 32} × threads {1, 4}`, N=8 wave on, 48회.
조합마다 `A,B,B,A` 를 연달아 돌려 같은 세션 안에서만 비율을 만든다.

| 조합 | BASE | SP | R |
|---|---|---|---|
| s447 B16 t4 | 6,523.5 | 4,400.9 | **1.4823** |
| s447 B16 t1 | 13,588.3 | 13,619.5 | **0.9977** |
| s447 B32 t4 | (진행 중) | | |

`B=16` 이 `B=32`(1.426×)보다 이득이 크다. 배치가 작을수록 pack 중복 비중이
커진다는 예측과 방향이 맞는다.

harness `prefill_bench/generality.sh`, artifact `artifacts/generality/`.

---

## 8.4 다음 작업 — 우선순위

| # | 작업 | 왜 |
|---|---|---|
| 1 | **2×2 ablation (앵커)** | 논문의 핵심 표. harness 준비 완료 (`prefill_bench/ablation2x2.sh`) |
| 2 | **recurrence 로 1.436× 원인 분해** | "pipeline amplification" 을 관찰이 아니라 메커니즘으로 |
| 3 | 일반성 스윕 완료 | 진행 중 |
| 4 | 최종 시스템 대 llama.cpp **동일 조건 재측정** | 곱셈 금지를 푸는 유일한 길 |
| 5 | (선택) CTGHP 자동화 | SharedPack 을 독립 기여로 키울 때만 |

### 1번 — 2×2 ablation

| 구성 | Wave | SharedPack |
|---|---|---|
| 기존 시스템 | ✗ | ✗ |
| Wave only | ✓ | ✗ |
| SharedPack only | ✗ | ✓ |
| 최종 | ✓ | ✓ |

예비 관측에서 **wave off 의 SP 이득 ≈ 1.276×, wave on 의 SP 이득 ≈ 1.51×** 로
방향성이 보인다. 앵커로 확정해야 co-design 주장이 선다.

### 2번 — counterfactual 네 번

```
F_{k,j} = max( F_{k−1,j} + D_{k−1,k,j},  F_{k,j−1} ) + C_{k,j}
```

| # | `C` | `D` | 묻는 것 |
|---|---|---|---|
| 1 | BASE | BASE | 모델이 실측을 맞히는가 |
| 2 | 영향받은 matmul 만 SP | BASE | 그 projection 개선율이 1.205× 보다 컸는가 |
| 3 | 모든 stage compute SP | BASE | 링크 고정해도 1.426× 가 나오는가 |
| 4 | SP | SP | 모델 타당성 |

**3번이 재현하지 못하면 TTFT 개선의 일부는 SharedPack 으로 귀속할 수 없다.**

---

## 8.5 리뷰어가 찌를 지점

| 위험 | 현재 방어 | 필요한 것 |
|---|---|---|
| "1.436× 의 메커니즘이 설명 안 됐다" | 1스레드 R=0.9977 (원인이 중복 packing 임은 확인) | §8.4 의 2번 (critical path 까지) |
| "두 기법을 나열했을 뿐이다" | 예비 관측의 방향성 | §8.4 의 1번 (2×2 앵커) |
| "wave PP 는 GPipe 와 유사하다" | 저속·비대칭 링크라는 조건, 공개 baseline 0.41~0.44× | — |
| "shared packing 은 알려진 원리다" | 분산 파이프라인 E2E 측정, 계층 co-design | (선택) CTGHP |
| "`N` 축이 노드 구성과 교락됐다" | 문서에 명시 | `(N, π)` 계획 비교로 표기 |
| "단일 플랫폼·단일 모델" | 13B capacity | held-out 모델 검증 |

---

## 8.6 절대 하지 말 것

```
4.57×  ×  1.436×  =  6.56×                       ← 곱하지 않는다
"pipeline amplification"                          ← 관측이지 설명이 아니다
"스테이지가 빨라져 bubble 이 줄었다"                ← 슬롯 수·비율이 준 게 아니다
"연산 1.20× + syncWait 1.77× 이득"                 ← 이중 계산
"다른 엣지 가속은 정확도를 희생하지만 우리는 아니다"  ← 근거 없음
"새로운 pipeline 알고리즘과 packing 알고리즘"       ← 개별 기법은 선행이 있다
```

---

## 8.7 하드웨어

| 노드 | 상태 |
|---|---|
| `.113 .191` (16 GB) | 정상 |
| `.54 .94 .96 .103 .104` (8 GB) | 정상 |
| **`.166`** | 네트워크·sshd 복구. **공개키 인증 미해결** — 재설치로 `authorized_keys` 소실 |
| `.116` | 존재하지 않음 (`.166` 의 오타) |

root 노드 SD 카드에 **선재하는 배드 섹터**. 실험이 원인은 아니고 모델 무결성은
확인됐다. **교체 권장.**
