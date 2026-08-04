# Prefill/TTFT Optimization on SBC Commodity Clusters — Research Plan

작업 베이스: `layer-skip-bypass` 소스에서 분리 (모델/토크나이저는 심볼릭 링크, 실측 로그는 미복사)
선행 연구: PiPP (decode 구간 skip) — 본 연구는 **prefill 구간**을 대상으로 함

---

## 0. 한 줄 요약

> 기존 병렬 컨텍스트 인코딩 기법은 **초장문 · 고대역폭 · 연산 풍부** 체제에서 설계되었다.
> SBC 커머디티 클러스터(CPU only, 1GbE, 프롬프트 0.5~4K)는 그 체제에 속하지 않으며,
> 이 체제에서는 최적 구성이 **정량이 아니라 정성적으로** 달라진다.
> 그 새로운 최적 분할을 도출하는 **비용 모델 + 플래너 알고리즘**을 제안한다.

**우리가 제안하는 것은 attention 메커니즘이 아니라 "무엇을 어떻게 나눌지 결정하는 규칙"이다.**

---

## 1. 문제 정의

TTFT 시점에 필요한 산출물은 두 개뿐이다.
1. 모든 토큰 × 모든 레이어의 **KV 캐시**
2. **마지막 토큰 1개**의 로짓

토큰 0..S-2 의 상위 레이어 residual stream은 오직 그 레이어의 KV를 만들기 위해서만 계산된다.

### Llama-3-8B 레이어 1개 · 토큰 1개당 FLOPs (h=4096, d_ff=14336, GQA 8 KV head)

| 연산 | FLOPs | 비중 |
|---|---|---|
| K, V projection | 1.7e7 | 3.9 % |
| Q, O projection | 6.7e7 | 15.4 % |
| MLP (SwiGLU) | 3.5e8 | 80.7 % |
| **합계** | **4.36e8** | 100 % |

→ KV 생성의 "본질적" 연산은 4 % 인데, 입력 x_l을 만들려고 나머지 96 %를 순차 실행한다.
→ 그 96 %가 정확히 all-reduce를 유발하는 부분(O-proj 뒤, MLP 뒤)이다.

### 통신량 비교 (S=512, N=4, FP32, 추정치 — EXP-1에서 실측 검증)

| 방식 | prefill 총 통신량 | 동기 형태 |
|---|---|---|
| TP (distributed-llama 현행) | ~768 MB | all-reduce × 64 (barrier) |
| Ring CP (pass-KV, exact) | ~100 MB | P2P × 32 |
| 블록 독립 인코딩 + 단일 merge | ~1 MB | reduce × 1 |

---

## 2. 가설 (EXP-1이 검증/반증할 대상)

| ID | 가설 | 반증되면 |
|---|---|---|
| **H1** | prefill 시간의 25~45 %가 통신+barrier 대기이며, 그중 **대기(straggler)** 가 순수 전송보다 크다 | 통신 최적화 노선 폐기, 연산 축(H3)만 추진 |
| **H2** | S ≤ 4K 구간에서는 attention의 O(S²) 항이 지배적이지 않고 **MLP/projection이 병목** | 분할 축 재선택 논거 소멸 → 블록 독립 인코딩이 메인 |
| **H3** | 상위 절반 레이어의 KV를 x_k에서 직접 산출하면 prefill FLOPs가 ~48 % 감소 | depth 축 폐기 |
| **H4** | 기존 병렬 인코딩의 **anchor amortization 전제가 SBC 체제에서 무너져** anchor 오버헤드가 ~2× 연산으로 그대로 남는다 | 논문의 핵심 관찰 소멸 → 재설계 필요 |
| **H5** | CPU 코어가 4개뿐이라 통신 스레드와 GEMM이 경합, **overlap이 zero-sum**이 되는 구간이 존재한다 | GPU식 overlap 설계를 그대로 사용 |

**H2와 H4가 이 연구의 핵심이다.** 둘 다 "기존 설계의 전제가 이 체제에서 깨진다"는 형태이고, 방어하기 쉽다.

---

## 3. 기여 구조 (논문 클레임 초안)

> **We do not propose a new attention mechanism.** We show that existing parallel
> context-encoding designs are tuned for a regime — ultra-long prompts, high-bandwidth
> interconnects, abundant compute — that commodity SBC clusters do not occupy, and that in
> the short-prompt, low-bandwidth, compute-scarce regime their optimal configuration changes
> **qualitatively**, not merely quantitatively.

1. **Measurement study** — N-node SBC/Ethernet 클러스터에서 기존 설계의 전제 3가지가 깨지는 것을 실측
   (anchor amortization / DMA-offloaded communication / attention-dominated cost)
2. **Analytical cost model** — 체제를 가르는 무차원 파라미터 ρ, 실측으로 검증,
   (bandwidth × prompt length × node count) 축의 **phase diagram**
3. **Regime-aware prefill partitioning algorithm** — 분할 축, 블록/anchor 정책, 코어 예산을 모델에서 결정
4. **Evaluation** — 4/8노드 동종·이종 구성, matched-accuracy 기준 TTFT 비교

### Baseline (반드시 전부 구현)

| Baseline | 목적 |
|---|---|
| TP (distributed-llama 현행) | 현재 시스템 |
| Ring CP (pass-KV, exact) | exact 상한 |
| **Star Attention 직접 포팅** | "포팅 아니냐" 질문을 그래프 하나로 종결 |
| 제안 기법 | — |

---

## 4. 타겟

- 주 타겟: **INFOCOM / ICDCS / MobiSys / IPDPS**
- 상방: MLSys / EuroSys (phase diagram + 이종 실험이 탄탄할 경우)
- 저널: IEEE TPDS / TMC / IoT-J
- 예비 발표: KCC/KSC

### 리스크와 대비

| 리스크 | 대비 |
|---|---|
| "체제 차이가 사소하다" | phase diagram에서 **선택이 뒤바뀌는 경계**가 실사용 구간(1GbE, 4~8노드, 0.5~4K)을 관통함을 보일 것 |
| "Pi에서만 성립" | SBC 2종 이상 + 대역폭 스윕(100M/1G/2.5G)으로 모델 검증 |
| "근사인데 exact와 비교 불가" | exact CP 직접 구현 → **동일 정확도 지점 속도** 비교 |
| "기여가 엔지니어링" | 비용 모델과 최적화 문제 정식화를 본문 전면 배치 |
| Pulsar Attention과 충돌 | anchor를 **정확도** 각도가 아니라 **통신 vs 재계산 비용** 각도로만 다룰 것 |

---

## 5. 진행 순서

- [ ] **EXP-1 prefill 비용 분해** ← 현재 단계. `research/01-EXP1-cost-breakdown.md`
- [ ] EXP-2 Star Attention 포팅 + anchor 오버헤드 실측 (H4)
- [ ] EXP-3 정확도 스윕 (블록 수 2/4/8 × anchor 크기, RULER/LongBench)
- [ ] EXP-4 exact Ring CP 구현 → matched-accuracy 비교
- [ ] 비용 모델 피팅 → phase diagram
- [ ] 플래너 알고리즘 설계 및 구현

**EXP-1 결과가 스토리를 결정한다.**
- H2 성립(attention 비지배) → 분할 축 재선택이 메인
- H2 반증 → 블록 독립 인코딩이 메인
