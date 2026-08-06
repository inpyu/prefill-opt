# 설계: 토큰 축 블록 병렬 prefill (SBC 클러스터)

**목표: TTFT 를 단일 노드 대비 3배 이상 단축한다.**

이 문서는 [06-baseline-status.md](06-baseline-status.md) 의 baseline 위에서
**논문 기여가 될 알고리즘**을 설계한다. 여기부터는 baseline 위생이 아니다.

---

## 1. 측정이 정한 출발점

전부 이 클러스터에서 직접 잰 값이다 (llama3-8b_q40, Pi5 4코어, 1GbE, S=447).

| 구성 | prefill | syncWait | syncXfer | gemm | 단일 대비 |
|---|---|---|---|---|---|
| 단일 노드 | 37,939 ms | — | — | **33,079** | 1.00× |
| TP 2노드 | 33,633 | 14,611 | 1,965 | **13,942** | 1.13× |
| PP 4노드+wave | 39,825 | 23,102 | 829 | — | 0.95× |
| SP 2노드 | 43,620 | 1,785 | 537 | **36,502** | 0.87× |
| SP 4노드 | 47,374 | 8,178 | 1,687 | **32,766** | 0.80× |

### 여기서 읽어야 할 세 가지

**(a) 연산이 전체의 87 % 이고, 나누면 정확히 나뉜다.**
gemm 33.1초는 prefill 37.9초의 87 %. TP2 에서 13.9초로 떨어졌다(2.37×, 이상값 2× 초과는
캐시 효과). **즉 연산 분할은 잘 되고, 남는 문제는 동기뿐이다.**

**(b) 깊이 축(PP)의 대기는 구조적이라 튜닝으로 안 줄어든다.**
[발견 #5](06-baseline-status.md#7-알려진-기존-버그-distributed-llama-원본)의 poll 수정으로
TP2 는 42 % 빨라졌는데 PP4 는 3 % 밖에 안 줄었다. PP4 는 전송량이 TP2 의 절반 이하인데
대기는 더 크다(23.1 vs 14.6초). 청크가 스테이지를 **순서대로** 통과해야 하기 때문이다.

**(c) dllama 의 `--sp-size` 는 토큰 축 병렬이 아니다.**
`sliceKvCache()` 의 `localSeqStart/localSeqLen` 은 **KV 저장을 시퀀스 방향으로 쪼갤 뿐**,
모든 노드가 같은 토큰을 계산한다. 측정이 이를 확인한다 — SP2/SP4 의 gemm 이
단일 노드와 동일(36.5 / 32.8초 vs 33.1초)하다. 메모리 최적화이지 병렬화가 아니다.

> **prefill 에서 본질적으로 병렬인 축은 토큰 축인데(레이어 안의 모든 토큰은 서로 독립),
> 그 축으로 연산을 나누는 기능이 없다.** 이것이 우리가 만들 것이다.

### 이론 상한

```
TTFT_ideal ≈ gemm/N + attention + merge
N=4 →  33.1/4 + α ≈ 9.5초   (3.99×)
N=8 →  33.1/8 + α ≈ 5.1초   (7.4×)
```

---

## 2. "KV 를 통합해야 하나" — 아니다, 조건부로

각 노드가 자기 토큰 구간의 KV 만 만들면, 그걸 하나로 합쳐야 하는 것 아닌가?

**decode 를 같은 축으로 돌리면 합칠 필요가 없다.** Context Parallel 의 표준 결과다.

decode 시 각 노드가 **자기 KV 구간에 대해 partial attention 과 log-sum-exp** 를 계산하고,
LSE 를 작은 all-reduce 로 병합하면 전체 attention 과 **수학적으로 동일**하다.
교환량은 레이어당 헤드당 `(head_dim + 1)` float 로 무시할 수준이다.
근거: [vLLM CP RFC](https://github.com/vllm-project/vllm/issues/26133),
[Meta Context Parallelism](https://arxiv.org/pdf/2411.01783) 의 pass-Q,
[Tree Attention](https://arxiv.org/html/2408.04093v2).

| decode 구성 | KV 통합 비용 |
|---|---|
| **토큰 축 (prefill 과 동일)** | **0** |
| 한 노드로 수집 | S=447 → 117 MB ≈ 1초 / S=2048 → 537 MB ≈ 4.8초 |
| TP (현행 dllama) | 헤드 축으로 재분배 → all-to-all, 가장 비쌈 |

> **설계 제약: prefill 축 = decode 축.**
> 현행 dllama decode 는 TP 이므로, 최종적으로는 decode 도 토큰 축으로 바꿔야 한다.
> 다만 v1 에서는 수집(gather)으로 우회한다 — §5 참조.

---

## 3. 알고리즘: Anchor-shared block-parallel prefill

```
입력 S 토큰을 N 블록으로 분할, 노드당 1블록

  노드 0: 블록 0 (= anchor)  ┐
  노드 1: 블록 1             ├─ 레이어별로 동시 계산
  노드 2: 블록 2             │
  노드 3: 블록 3             ┘

각 레이어 l 에서:
  노드 0   : 자기 블록 계산 → layer-l anchor KV 를 브로드캐스트
  노드 i>0 : [anchor KV ⊕ 자기 블록] 에만 attention
             그 외 노드 간 통신 없음

종료 후:
  KV 는 토큰 구간별로 샤딩된 채 유지
  쿼리 토큰만 분산 softmax(LSE merge)로 단일 병합
```

블록 간 attention 을 생략하므로 **근사**다. anchor 가 그 손실을 완화한다
([Star Attention](https://arxiv.org/pdf/2411.17116) 의 핵심 아이디어).

### 연산량

| 항목 | 단일 노드 | 블록 병렬 (노드당) |
|---|---|---|
| gemm | O(S·h²) | **O(S/N·h²)** |
| attention | O(S²·h) | **O((S/N + A)²·h)** |
| 통신 | 0 | anchor KV 브로드캐스트만 |

attention 은 N² 로 줄어든다(총합 기준 S²/N). **연산량 자체가 TP/CP 보다 적다.**

---

## 4. 핵심 설계 결정: anchor 를 재계산하지 말고 브로드캐스트한다

Star Attention 은 anchor 블록을 **모든 노드가 중복 계산**한다. 노드당 처리 토큰이
`S/N + A` 가 되고, `A = S/N` 이면 연산이 2배 → speedup 이 N → N/2 로 반토막 난다.

우리 클러스터 실측값으로 두 선택지를 비교하면 (N=4, S=447, L=32):

| 방식 | 비용 |
|---|---|
| anchor 재계산 | 노드당 +9.5초 (연산 2배) |
| **anchor KV 브로드캐스트** | 전송 32 MB ≈ 0.3초 + 배리어 32회 × **16.3 ms** ≈ **0.8초** |

**약 12배 차이.** 배리어 비용 16.3 ms 는 [발견 #5](06-baseline-status.md) 수정 후
TP2 에서 직접 측정한 값이다(syncWait 14,611 ms ÷ 896 배리어).

> 이전 초안에서 이를 "GPU 대비 반전"으로 서술했다가 근거 부족으로 철회했다.
> 지금은 **우리 하드웨어의 측정값만으로** 성립한다. GPU 와 비교할 필요가 없다.

### 브로드캐스트의 의존성 처리

노드 0 이 레이어 l 의 anchor KV 를 만들어야 노드 i>0 이 레이어 l 을 계산할 수 있다.
→ **노드 0 을 한 레이어 앞서 달리게 한다.** 스큐는 1 레이어(전체의 1/L = 3 %)뿐이고,
전송은 단방향이라 연산 뒤로 숨길 수 있다.

anchor 는 어차피 블록 0 이므로 노드 0 이 자기 일을 하면서 부산물로 만든다.

---

## 5. 구현 계획 (dllama 기준)

### Phase A — 블록 병렬 prefill + KV 수집 (v1 프로토타입)

decode 를 건드리지 않고 prefill 만 바꾼다. **TTFT 이득을 먼저 확인하는 것이 목적.**

| # | 작업 | 비고 |
|---|---|---|
| A-1 | 새 토폴로지 모드 `--block-parallel N` | 가중치를 **복제**한다(슬라이스 아님). 노드당 6.32 GB → 16 GB 노드 4대 필요 |
| A-2 | root 가 토큰 구간을 노드에 배정 | `controlPacket` 에 blockStart/blockLen 추가 |
| A-3 | 각 노드가 자기 구간을 position offset 과 함께 prefill | 기존 `setPosition`/`setBatchSize` 재사용 |
| A-4 | anchor 없이 먼저 동작시킨다 | 정확도는 나쁘지만 **속도 상한을 먼저 잰다** |
| A-5 | prefill 후 각 노드가 자기 KV 를 root 로 전송 | S=447 → 117 MB ≈ 1초 |
| A-6 | root 가 기존 방식으로 decode | 변경 없음 |

**예상 TTFT**: `33.1/4 + attention + 1.0 ≈ 10초` → **3.8×**

A-4 가 중요하다. anchor 를 붙이기 전에 "토큰 축이 실제로 gemm 을 1/N 로 줄이는가"를
먼저 확인해야 한다. 여기서 안 나오면 나머지 설계가 무의미하다.

### Phase B — anchor 도입 및 정확도 회복

| # | 작업 |
|---|---|
| B-1 | anchor KV 브로드캐스트 (`SYNC_ANCHOR_KV` 신설, 노드 0 → 나머지) |
| B-2 | 노드 0 을 한 레이어 앞서 실행하는 스큐 스케줄 |
| B-3 | anchor 크기 A 스윕 → 정확도-속도 파레토 |
| B-4 | 비교군: anchor 재계산 방식(Star 원본) 도 구현해 §4 의 12배 주장을 실측 검증 |

### Phase C — 토큰 축 decode (수집 제거)

| # | 작업 |
|---|---|
| C-1 | decode 시 각 노드가 partial attention + LSE 계산 |
| C-2 | LSE all-reduce 로 병합 |
| C-3 | A-5 의 KV 수집 제거 → TTFT 에서 1초 회수, 긴 프롬프트일수록 이득 큼 |

### 선행 조건

- **토큰 임베딩 F32 → 양자화** ([발견 #3](03-attention-batching.md#65-발견-3-부수적-토큰-임베딩이-f32-로-저장된다))
  가중치를 복제하므로 노드당 6.32 GB 다. 4.5 GB 로 줄이면 KV·활성화 여유가 생긴다.
  8 GB 노드까지 쓰려면 필수.
- 16 GB 노드 4대 확보됨: root(.105) + .113 + .166 + .191

---

## 6. 검증 계획

| 항목 | 방법 | 기준 |
|---|---|---|
| 속도 | S=447/1789, N=1/2/4 스윕 | Phase A 에서 3× 이상 |
| 연산 분할 확인 | `gemmMs` 가 1/N 로 떨어지는가 | SP 처럼 안 줄면 설계 실패 |
| 정확도 | RULER / LongBench / needle-in-haystack | Star Attention 은 4~8블록에서 97~100 % 보고 |
| exact 대비 | Ring CP 를 별도 구현해 동일 정확도 지점 비교 | Phase B 이후 |

**정확도에 유리한 조건**: 우리는 블록 수가 4~8 로 적다. 초장문을 겨냥한 기존 연구들은
수십 블록을 쓰므로 블록 간 attention 손실이 훨씬 크다.

---

## 7. 위험

| 위험 | 대응 |
|---|---|
| 근사 정확도가 실사용에 부족 | Phase B 의 anchor 크기 스윕. 그래도 부족하면 exact Ring CP 로 선회 |
| 가중치 복제로 메모리 부족 | 임베딩 양자화 선행. 8 GB 노드는 Phase C 이후로 |
| decode 축 불일치로 TTFT 이득 상쇄 | Phase A 에서 수집 비용을 명시적으로 측정해 보고. Phase C 로 제거 |
| "Star Attention 을 SBC 로 포팅한 것" 이라는 평가 | §4 의 anchor 브로드캐스트가 우리 고유 기여. 재계산 방식과 실측 비교(B-4)로 방어 |

---

## 8. 선행연구 상의 위치

| 연구 | 관계 |
|---|---|
| [Star Attention](https://arxiv.org/pdf/2411.17116) | 블록 독립 인코딩 + 단일 merge 의 원형. **anchor 를 재계산**한다 |
| [APE](https://arxiv.org/pdf/2502.05431) | 병렬 인코딩 KV 의 분포 정렬. 정확도 회복 기법으로 차용 가능 |
| [Meta CP](https://arxiv.org/pdf/2411.01783) | exact 비교군(Ring, pass-KV/pass-Q). decode 병합 방식의 근거 |
| [Pulsar Attention](https://arxiv.org/html/2607.20457) | anchor 를 통계 요약으로 대체. **정확도 각도로 충돌하므로 우리는 비용 각도로만 다룬다** |
| [Galaxy](https://arxiv.org/pdf/2405.17245) | 엣지 클러스터 분산 추론. GPU 보드 대상이라 배리어 비용 체계가 다르다 |

**우리 차별점**: 기존 연구는 전부 초장문(128 K)·고대역폭·GPU 전제다.
우리는 **짧은 프롬프트(0.5~4 K)·1GbE·CPU only** 체제이고, 그 체제에서는
anchor amortization 이 성립하지 않아 설계가 달라진다. 그 차이를
**측정된 배리어 비용(16.3 ms)** 으로 정량화한 것이 기여의 핵심이다.
