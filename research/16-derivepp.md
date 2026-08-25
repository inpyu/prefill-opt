# DerivePP — 저속 Ethernet CPU 클러스터에서 확장되는 Prefill Wave Pipeline

**목표 저널: IEEE TPDS**  ·  최종 검토 2026-08-19
관련: [14-scheduling-model.md](14-scheduling-model.md), [12-findings.md](12-findings.md),
[15-labmeeting.md](15-labmeeting.md)

> DerivePP 는 저속 Ethernet 으로 연결된 commodity CPU 클러스터에서 microbatch wave
> pipeline 을 통해 통신을 계산과 중첩하며, **기존 공개 분산 구현들이 로컬 CPU 성능의
> 0.41~0.44× 에 머문 환경에서 4.57~4.68× 를 달성한다.**

---

## 0. 먼저 읽는 한 장 요약

### 0.1 문제 — 기존 CPU 분산 구현은 노드를 늘려도 로컬을 넘지 못한다

동일 하드웨어(4×16 GB + 4×8 GB Pi 5), 동일 모델·양자화·스레드 수·프롬프트에서
측정했다(§7.16, §7.17).

| 시스템 | 최선 구성 | 로컬 llama.cpp 대비 |
|---|---|---|
| llama.cpp 로컬 | N=1 | 1.00× |
| llama.cpp RPC | N=2 | 최대 **0.41×** |
| 공식 distributed-llama | N=4 | 최대 **0.44×** |

**실패 원인은 서로 다르다.**

- **llama.cpp RPC** — 비동기 pipeline 경로가 소스에 없어(`set/get/cpy_tensor_async
  = NULL`) 서브그래프가 순차 실행된다. N=2→8 이 완전히 평탄하다.
- **공식 distributed-llama** — 같은 switch 안에서는 확장하나(N=4 에서 4.24×)
  **100 Mb 업링크를 건너면 붕괴**한다(N=8 에서 0.12×).

공통점은 하나다. **동기 실행이 저속 링크의 통신시간을 계산과 충분히 중첩하지 못한다.**

### 0.2 해결 — microbatch wave pipeline

프롬프트를 microbatch 로 나누고 여러 layer stage 에 wave 형태로 흘려보낸다.

```text
Stage 0: MB0 → MB1 → MB2 → MB3
Stage 1:       MB0 → MB1 → MB2 → MB3
Stage 2:             MB0 → MB1 → MB2 → MB3
```

느린 링크를 건너는 activation 전송이 **다른 stage 의 계산과 중첩**된다.

### 0.3 결과

| Prompt | Wave-PP 엔진 | Microbatch 선택 | 최종 |
|---|---|---|---|
| S=447 | 4.14× | 1.132× | **4.68×** |
| S=1789 | 4.51× | 1.013× | **4.57×** |

곱셈 분해:  `T_llama/T_B16 = (T_llama/T_B32) × (T_B32/T_B16)`

- 자기 단일 노드 대비 **6.64×**, 8노드 효율 약 83% (S=1789)
- 공식 distributed-llama 가 붕괴한 **동일한 100 Mb 업링크 환경에서도 확장**한다

> 평가한 세 CPU 실행 시스템 가운데 본 연구만 로컬 llama.cpp 의 prefill 성능을
> 초과했다. (모든 기존 CPU 분산 시스템을 평가한 것은 아니다.)

### 0.4 기여의 우선순위

1. **저속·비대칭 Ethernet 에서 확장되는 CPU prefill wave pipeline**
2. **공개 CPU 분산 구현 두 종류와의 공정한 비교** — 동일 모델·양자화·스레드·프롬프트,
   모델 로드 제외, 생성 단계 미포함
3. **계산·통신 중첩을 통한 8노드 확장성** — 링크가 10배 비대칭인 조건에서
4. Calibration 기반 microbatch 선택과 **AxisCert**
5. 효과가 없었던 **sublayer DP · CoRePP · CP 의 구조적 분석**

**AxisCert 의 위치.** 4.6× 를 만든 핵심 알고리즘이 **아니다.** 정확한 역할은:

> 효과 없는 최적화 축을 배제하고, 플랫폼에서 실제로 유효한 실행 설정만 선택하는
> 보조 계획 절차.

모든 요청에서 같은 최적화 축이 활성이라고 가정하지 않고, 길이와 플랫폼 조건별로
실질적 leverage 를 판정한다.

### 0.5 무엇이 결정되는가

| 결정값 | 뜻 | 상태 |
|---|---|---|
| `B` | microbatch 폭 | **활성** — 단 길이 의존(§7.18) |
| `N` | 사용할 stage 수 | 미측정 (audit 필요) |
| `p` | att/ff 서브블록 분할 | **비활성** — S=1789·7212 양쪽 (§7.9, §7.12c) |
| `π` | 노드 순서 | **미확정** — 근거였던 순위 역전이 철회됨 (§7.15) |
| `r` | stage 내 코어 분할 | **기각** — G=0.967 (§12.3) |
| `g` | CP island | **기각** — 상한 1/(M+N−2) (§12.4) |

### 0.6 `B` 축의 현재 표현 (주의)

**anchor 재측정으로 확정됐다**(§7.18 v2). 두 길이 모두에서 `B=16` 이 `B=32` 대비
**1.13~1.14×** 다.

| S | B16/B32 | 근거 |
|---|---|---|
| 447 | 1.132× | 동일 세션 측정 |
| 1789 | **1.140×** | anchor 9회 (J2 의 1.136× 와 독립 재현) |

한때 `S=1789` 에서 1.013× 로 보였던 것은 **세션 간 비교의 인공물**이었다 —
`B=32` 곡선과 `B=16` 곡선을 다른 세션에서 재 `B=32` 의 11% 드리프트가 비율에
들어갔다. **두 설정의 비율은 반드시 같은 세션 안에서 교차 배치해 재야 한다.**

### 0.7 sweep-free 의 정확한 의미

- **한다**: 짧은 production-path kernel/executor/link 캘리브레이션으로 플랫폼 비용
  함수를 한 번 구성한다.
- **하지 않는다**: 실제 프롬프트를 `(N,B,π,p)` 조합마다 end-to-end 로 반복 실행해
  가장 빠른 설정을 찾지 않는다.

### 0.8 실측이 바꾼 연구 방향 (세 번)

1. **`B` 의 닫힌형 법칙** → 반증(§12.2). 캘리브레이션 격자 선택 축으로 축소.
2. **`π`·`p` 공동 최적화가 중심** → J1b/J4-R 이 뒤집음. 분할축은 두 길이 모두 비활성.
3. **해석적 스케줄러가 헤드라인** → 세 시스템 비교가 뒤집음. **성능은 wave-PP 엔진에서
   나오며, AxisCert 는 보조 계획 절차다.**

---

## 1. 배경 — 처음 보는 독자를 위한 실행 예

### 1.1 Prefill과 pipeline parallelism

Prefill은 사용자가 보낸 `S`개 프롬프트 토큰을 Transformer의 모든 레이어에 통과시켜
KV cache를 만드는 단계다. 예를 들어 32개 레이어를 4개 노드에 다음처럼 배치할 수 있다.

```text
node0: layer  0.. 7
node1: layer  8..15
node2: layer 16..23
node3: layer 24..31 + lm_head
```

프롬프트 전체를 한 덩어리로 보내면 node0가 끝날 때까지 나머지 노드는 논다.
그래서 `S`개 토큰을 폭 `B`의 마이크로배치로 나눈다.

```text
S = 128, B = 32  →  M = ceil(128/32) = 4

시간 →       t0       t1       t2       t3       t4       t5       t6
stage0       mb0      mb1      mb2      mb3
stage1                mb0      mb1      mb2      mb3
stage2                         mb0      mb1      mb2      mb3
stage3                                  mb0      mb1      mb2      mb3
```

`mb0`가 stage1로 넘어가면 stage0는 `mb1`을 계산할 수 있다. 서로 다른 노드가
서로 다른 마이크로배치를 동시에 처리하는 것이 PP의 겹침이다.

### 1.2 `B`가 작을 때와 클 때

`B`에는 서로 반대 방향의 압력이 걸린다.

| `B` | 장점 | 단점 |
|---|---|---|
| 작음 | `M=ceil(S/B)`가 커져 파이프라인 버블이 작아짐 | 작은 GEMM이 CPU 타일을 채우지 못해 토큰당 계산시간 증가 |
| 큼 | GEMM 효율과 호출 고정비 상각이 좋아짐 | `M`이 작아져 스테이지가 놀고, 첫 출력까지 fill 시간이 커짐 |

`B_min`은 실행 가능한 최소값이 아니라 **그보다 작아질 때 커널 효율 손해가 뚜렷해지는
성능 knee**다. Pi 5의 현재 측정에서는 `B_min=16`이다. 그러므로
`N ≤ S/B_min` 역시 프로그램의 정확성 조건이 아니다. `M≥N`을 만족시키며 효율적인
커널 폭을 유지할 수 있는지를 빠르게 판단하는 성능상 기준이다.

### 1.3 고전식은 어디까지 맞는가

모든 스테이지와 마이크로배치 시간이 같은 이상적인 경우, 한 마이크로배치 시간이
`c`라면 다음과 같다.

```text
직렬 실행시간      = M·N·c
파이프라인 실행시간 = (M+N−1)·c
속도향상 상한       = M·N/(M+N−1)
버블 비율           = (N−1)/(M+N−1)
```

예를 들어 `N=8`, `M=64`면 버블 비율은 `7/71=9.9%`다. 그러나 실제 causal
attention에서는 뒤쪽 마이크로배치가 더 비싸고, 노드도 이종이며, 링크 전송도 있다.
따라서 이 식은 직관을 주는 특수 경우이지 DerivePP의 최종 비용식이 아니다.

---

## 2. 문제를 정확히 정의한다

### 2.1 기호

| 기호 | 의미 |
|---|---|
| `S` | 패딩 전 실제 프롬프트 토큰 수 |
| `S_exec` | 실제 실행 행 수. 타일 패딩 사용 시 `4·ceil(S/4)` |
| `B` | 마이크로배치의 최대 토큰 수 |
| `M=ceil(S_exec/B)` | 마이크로배치 개수 |
| `N_max` | 사용 가능한 최대 노드 수 |
| `N` | 이번 요청에서 활성화할 스테이지 수, `1≤N≤min(N_max,L)` |
| `π=(v_0,…,v_{N-1})` | 선택한 노드와 파이프라인 순서. 현재 구현에서는 root가 `v_0`으로 고정 |
| `L` | 연속 배치 가능한 서브블록 수. 32-layer 모델을 att/ff로 나누면 64 |
| `p=(p_0,…,p_N)` | `0=p_0 < … < p_N=L`인 경계. stage `k`는 `[p_k,p_{k+1})` 담당 |
| `d` | hidden dimension |
| `d_ff` | FFN intermediate dimension |
| `d_kv` | 모든 KV head를 합한 dimension |

초기 Python 구현은 노드 순서를 고정하고 `(N,B,p)`만 최적화한다. 그러나 새 실측에서
노드별 연산자 순위가 역전됐고(⚠️ §7.15 에서 철회) 링크 대역폭은 10배 차이 났다. 따라서 최종 문제는
노드 subset과 순서 `π`까지 포함한다. 고정 순서 결과는 전체 DerivePP가 아니라
**fixed-placement 하위 문제**의 결과로 구분한다.

### 2.2 마지막 부분 청크와 causal 위치

마이크로배치 `j`의 시작 위치, 실제 실행 폭, 마지막 KV 위치를 다음처럼 정의한다.
패딩을 사용하지 않으면 `S_exec=S`다.

```text
u_j = j·B
b_j = min(B, S_exec−u_j)  # 마지막 청크는 B보다 작을 수 있음
ℓ_j = u_j+b_j             # 이 청크가 처리한 뒤 존재하는 KV prefix 길이
```

attention의 정확한 내적 개수는 청크의 모든 query 위치가 서로 다르므로

```text
A_j = Σ_{r=1..b_j}(u_j+r)
    = b_j·u_j + b_j(b_j+1)/2
```

이다. QK와 AV를 합친 attention-core 연산량은 대략 `4·d·A_j`다. 이 정의가
중요한 이유는 기존의 `jB` 표현이 `j=0`일 때 첫 청크의 attention 비용을 0으로
만드는 오류가 있기 때문이다. 실제 첫 청크도 자기 앞 토큰을 보므로 비용이 0이 아니다.

현재 `--tile-aligned` 구현은 최대 3개의 dummy 행을 추가하고 마지막 prefill token을
반복해 계산한 뒤, decode 전에 position을 실제 `S`로 되돌린다. Prefill 로짓은 사용하지
않고 dummy 위치의 KV는 첫 decode에서 덮어쓰므로 모델 출력은 유지된다. 다만 planner는
실제로 수행한 dummy 연산까지 예측해야 하므로 비용 계산에는 `S`가 아니라 `S_exec`을
쓴다. 정확성 gate는 패딩 on/off가 동일 출력을 내는지 계속 검사한다.

### 2.3 서브블록 비용

노드 `k`가 청크 `j`에서 attention 서브블록 하나를 처리하는 비용을 다음처럼 구성한다.

```text
projection FLOPs = 4·b_j·d(d+d_kv)       # Q, K, V, O
attention FLOPs  = 4·d·A_j               # QK^T + AV, causal 합
FFN FLOPs        = 6·b_j·d·d_ff           # SwiGLU의 세 GEMM
```

실행시간은 FLOPs 자체가 아니라 캘리브레이션한 두 비용 함수로 바꾼다.

```text
Q_k(j) = T_q,k(4·b_j·d(d+d_kv), B=b_j) + T_a,k(4·d·A_j, B=b_j, prefix=ℓ_j)
F_k(j) = T_q,k(6·b_j·d·d_ff,       B=b_j)
```

- `T_q,k`: 노드 `k`의 Q4×Q8 GEMM 계열 시간. 호출 고정비와 작은 `B`의 효율 저하 포함.
- `T_a,k`: F32 QK/softmax/AV 계열 시간. prefix 길이와 working-set 효과 포함.
- RMSNorm, RoPE, cast, residual add처럼 낮은 차수의 연산은 production path에서
  직접 측정한다. held-out 잔차에 `γ_k·b_j`를 회귀해 맞추지 않는다.

stage `k`가 `[a,b)` 서브블록을 맡을 때의 계산시간은 그 구간에 들어 있는
attention/FFN 비용의 합이다.

```text
C_k,j(a,b) = Σ_{r∈[a,b), r=att} Q_k(j)
           + Σ_{r∈[a,b), r=ff } F_k(j)
           + X_k(j;a,b)
           + H_k(j)
```

`X_k(j;a,b)`는 production executor가 만드는 step 경계 동기화, thread별 종료시각
불균형, KV write, CAST/MERGE_ADD 같은 graph-local 이동 비용이다. 이는 end-to-end
잔차에 맞추는 자유 보정계수가 아니다. §7.6의 분해 실험을 통과한 성분만 직접 측정해
더한다. 특히 `X=n_step·h_barrier` 형태는 `h_barrier`가 `B`, 연산자, 레이어 수와
독립적이라는 held-out 검증을 통과할 때만 사용한다.

`H_k(j)`는 embedding, final norm, lm_head 같은 stage-local 비용이다. 현재 구현은
**각 prefill 청크의 마지막 행 하나에 lm_head를 계산**하고, 네트워크에는 마지막 청크의
로짓만 돌려보낸다. 따라서 현 코드의 비용 모델에는 한 행 lm_head를 모든 `j`에 넣되,
최종 로짓 전송은 `j=M−1`에만 넣어야 한다. 중간 청크의 lm_head 계산까지 생략하는
최적화를 적용하면 그때 `H_k(j)` 정의도 함께 바꾼다.

### 2.4 경계 전송 비용

서브블록 경계에서 다음 stage로 보내는 것은 `b_j×d` activation이다. 원소당 바이트 수를
`w`라 하면 전송 크기는 `x_j=w·b_j·d`이고, 링크 비용은 다음 affine 모델로 시작한다.

```text
D_k(j) = α_k + x_j/β_k
```

- `α_k`: stage `k→k+1` 링크의 고정 latency
- `β_k`: 해당 링크의 유효 bandwidth

여기서 `β_k`는 bandwidth에만 사용한다. lm_head 상수와 같은 이름을 재사용하지 않는다.

### 2.5 목적함수와 탐색 공간

최종 목표는 마지막 stage가 마지막 마이크로배치를 끝내고, 현재 프로토콜이 요구하는
최종 응답을 root가 받은 시각, 즉 TTFT의 prefill 부분을 최소화하는 것이다.

```text
minimize   T(N,B,π,p) = F_{N−1,M−1} + D_return
over       N ∈ {1,…,min(N_max,L)}
           B ∈ 정렬된 후보 집합 ℬ(S_exec,N)
           π ∈ root가 첫 stage인 활성 노드 순열
           p ∈ N개의 비어 있지 않은 연속 서브블록 분할
```

`L=64`, `N=8`이면 연속 분할만 `C(63,7)≈5.5×10^8`개다. 각 설정을 실제로 실행하는
end-to-end sweep은 불가능하고, 단순 전수 분할도 불필요하다.

`D_return`은 마지막 stage에서 root로 돌아오는 마지막 logit/control 응답 비용이다.
단일 노드에서는 0이고, 반환이 다른 activation 전송과 NIC를 공유한다면 §3.1의 별도
link event로 모델링한다. 이를 빼면 planner의 목적값과 실제 측정한 prefill 종료시점이
서로 달라진다.

---

## 3. 실제 파이프라인 시간을 계산한다

### 3.1 계산과 링크를 분리한 recurrence

다음 두 완료시각을 둔다.

- `F_{k,j}`: stage `k`가 microbatch `j`의 계산을 끝낸 시각
- `E_{k,j}`: 그 결과가 `k→k+1` 링크 전송까지 끝난 시각

```text
F_{0,j} = F_{0,j−1} + C_{0,j}

E_{k,j} = max(F_{k,j}, E_{k,j−1}) + D_k(j)

F_{k,j} = max(E_{k−1,j}, F_{k,j−1}) + C_{k,j}      (k > 0)

F_{k,−1}=0,  E_{k,−1}=0
```

각 `max`의 의미는 단순하다.

- 다음 stage는 **데이터가 도착해야** 현재 청크를 시작할 수 있다.
- 동시에 자기 stage의 **직전 청크 계산이 끝나야** 한다.
- 같은 링크의 전송은 FIFO로 직렬화한다.

이 모델은 계산과 비동기 전송이 겹칠 수 있고, 인접 링크가 독립이라는 가정이다.
현재 코드의 `forwardPrefillNoWait`와 맞는지 타임라인 trace로 검증한다. 100 Mb/s처럼
통신이 지배할 때 blocking send나 한 노드의 송·수신 NIC 경합이 드러나면 link state를
공유 NIC resource까지 확장한다. 검증하지 않고 “통신 overlap을 정확히 모델링한다”고
주장하지 않는다.

통신이 충분히 짧아 링크 직렬화를 무시할 수 있을 때는 `E_{k,j}=F_{k,j}+D_k(j)`가 되고,
문서의 간단한 recurrence로 환원된다.

### 3.2 고전적인 fill + steady-state 식과의 관계

모든 `C_{k,j}=C_k`이고 통신이 0이면 위 식은 다음 특수해를 갖는다.

```text
F_{N−1,M−1} = Σ_k C_k + (M−1)·max_k C_k
```

첫 항은 첫 마이크로배치가 모든 stage를 통과하는 **fill 시간**, 두 번째 항은 이후
마이크로배치가 가장 느린 stage 속도로 나오는 **steady-state 시간**이다. 그러나 causal
attention에서는 `C_{k,j}`가 `j`에 따라 증가하므로 `Σ+max`만 보존하면 정보가 사라진다.

### 3.3 문서와 현재 planner 구현의 차이

현재 `prefill_bench/derivepp.py`는 다음의 단순식을 사용한다.

```text
F_{k,j} = max(F_{k−1,j}+D_{k−1}(j), F_{k,j−1}) + C_{k,j}
```

즉 같은 링크에서 `j−1` 전송이 끝나는 시각 `E_{k,j−1}`을 상태로 들고 있지 않다.
통신이 계산보다 훨씬 짧은 현재 기본 순서에서는 근사오차가 작을 수 있지만, 100 Mb/s
업링크를 여러 번 지나는 순서를 평가할 때는 전송이 겹쳐 계산되는 오류가 생긴다.

또한 현재 brute-force enumerator도 같은 단순식을 사용한다. 따라서 “DP와 brute
force가 6/6 일치했다”는 결과는 **DP 구현의 내부 일관성**을 확인하지만, §3.1의
직렬화된 링크 recurrence가 구현됐음을 뜻하지 않는다.

개발 순서는 다음과 같다.

1. `E_{k,j}`를 simulator, DP, brute force 모두에 동일하게 구현한다.
2. 마지막 응답의 `D_return`도 실제 planner 목적값에 포함한다.
3. synthetic test에서 `D≫C`일 때 링크 처리량이 `1/D`를 넘지 않는지 검사한다.
4. 실제 `sendActivation` trace와 예측 event 순서를 비교한다.

이 네 항목이 끝날 때까지 네트워크-aware exact planner라는 표현은 사용하지 않는다.

---

## 4. Completion-vector DP

### 4.1 왜 스칼라 하나가 아니라 벡터인가

두 분할이 마지막 완료시간은 같더라도 중간 청크 완료시각은 다를 수 있다.

```text
plan A의 완료벡터: [5,  9, 13, 17]
plan B의 완료벡터: [6,  9, 14, 18]
```

이 예에서는 A가 모든 청크를 B보다 늦지 않게 끝낸다. 이후에 동일한 stage를 붙여도
A가 불리해질 수 없으므로 B는 버려도 된다. 반면 `[5,10,12,18]`과 `[6,9,14,17]`은
서로 앞서는 위치가 달라 둘 다 남겨야 한다.

### 4.2 DP 상태

```text
DP[k][i] = 앞 i개 서브블록을 앞 k개 stage에 배정했을 때 가능한
           stage k−1의 완료벡터 F_{k−1,*}의 nondominated frontier
```

`DP[k−1][h]`의 각 상태에서 새 stage에 `[h,i)`를 배정하고 §3.1 recurrence를 한 번
적용해 `DP[k][i]` 후보를 만든다. 각 상태는 완료벡터뿐 아니라 `(h, 이전 상태)`
backpointer를 저장해야 최종 분할을 복원할 수 있다.

### 4.3 지배 관계와 가지치기

완료벡터 `A`, `B`에 대해

```text
A ⪯ B  ⇔  A_j ≤ B_j  for every j
```

이면 A가 B를 지배한다고 한다.

**보조정리 1 — 단조성.** `A⪯B`이고 두 상태 뒤에 동일한 링크와 stage를 붙이면,
새 완료벡터도 `A'⪯B'`다.

**이유.** 링크 완료식과 stage 완료식은 모두 음수가 아닌 비용에 대한 `max`와 `+`로만
구성된다. 입력의 각 성분을 늦춰도 출력 완료시각이 빨라질 수 없다. 청크 순서에 대한
귀납으로 모든 성분에서 단조성이 유지된다.

**따름정리.** 같은 `DP[k][i]` 안에서 지배당한 상태를 제거해도 최적 분할을 잃지 않는다.
따라서 frontier를 전부 유지하면 **고정 `(N,B)`와 고정 노드 순서에 대한 최적 연속
분할**을 정확히 반환한다.

### 4.4 정확성 검증부터 한다

DP를 바로 큰 실험에 넣지 않는다.

1. `L≤12`, `N≤4`, `M≤8`의 작은 합성 문제를 만든다.
2. 모든 연속 분할을 brute force로 열거한다.
3. 각 분할을 같은 recurrence로 평가한다.
4. DP의 makespan과 경계가 brute-force 최적해와 일치하는지 검사한다.
5. 지배 pruning을 끈 결과와 켠 결과도 일치시킨다.

이 테스트가 통과해야만 실제 Llama 비용을 연결한다.

### 4.5 복잡도와 fallback

frontier의 최대 실측 크기를 `Φ`라 하면 후보 벡터 생성 비용은

```text
O(N·L²·Φ·M)
```

이다. 다만 frontier에서 서로의 지배 여부를 순진하게 검사하면 추가로 `Φ²M` 비용이
생긴다. 따라서 논문에는 생성 비용과 pruning 비용을 분리해 보고한다. `Φ`에는 작은
다항 상한이 보장되지 않으므로 exact DP를 무조건 “수 초”라고 단정하지 않는다.

구현 판단은 다음과 같다.

```text
대표 S/N/B에서 Φ와 메모리 실측
  ├─ Φ가 충분히 작음 → exact completion-vector DP
  └─ Φ가 큼          → 싼 scalar surrogate로 top-K 분할 생성
                        → 모든 후보를 exact recurrence로 재평가
```

fallback은 heuristic이므로 “정확한 최적 분할” 주장을 유지할 수 없다. 이 경우
brute-force가 가능한 작은 문제에서 최적성 gap, 큰 문제에서 best-of-grid attainment와
top-K 민감도를 반드시 보고한다.

새 실측에서는 관측한 실규모 한 점의 `Φ`가 879였으므로 exact 경로를 우선 구현한다.
다만 이것은 frontier가 다른 `S`, 모델, placement에서도 폭발하지 않는다는 보장이
아니다. 시간·메모리 guard와 fallback 코드는 제거하지 않는다.

### 4.6 노드 순서와 분할을 함께 고르는 subset DP

고정 순서 DP만으로는 새로 발견한 두 이종성을 이용할 수 없다.

- node `166`은 GEMM은 느리지만 attention은 빠르다.
- 서로 다른 스위치 군을 잇는 링크는 군 내부 링크보다 10배 느리다.

노드 순서를 먼저 정하고 분할을 나중에 정하면 두 결정이 서로 충돌할 수 있다. 이를
피하기 위해 DP 상태에 사용한 노드 집합과 마지막 노드를 추가한다.

```text
DP[U][v][i]
  U : 지금까지 사용한 노드 집합(root 포함)
  v : 현재 마지막 stage의 노드
  i : 앞에서부터 배정이 끝난 서브블록 수
  값: 가능한 완료시각 벡터의 nondominated frontier
```

전이는 아직 쓰지 않은 노드 `w`와 다음 경계 `e`를 동시에 고른다.

```text
(U,v,i)  -- node w가 subblock [i,e)를 담당 -->  (U∪{w},w,e)
```

이때 링크 비용은 `D_{v,w}`이고 계산 비용은 노드 `w`의 `T_q,w`, `T_a,w`에서 온다.
지배 비교는 미래 링크가 같은 상태끼리만 유효하므로 **동일한 `(U,v,i)` 안에서만** 한다.

노드가 `K`개일 때 출력 민감 복잡도는 대략

```text
O(2^K · K² · L² · Φ · M)
```

이다. `K=8`이면 순열 7!을 end-to-end로 실행하는 것보다 훨씬 작고, subset state의
dominance를 이용할 수 있다. 그래도 fixed-order DP보다 비싸므로 다음 순서로 구현한다.

1. `K≤6`, `L≤12`에서 node permutation × partition 전수열거와 일치시킨다.
2. 실제 `K=8`에서 frontier, 실행시간, 메모리를 잰다.
3. 너무 크면 먼저 링크 군을 연속 배치하도록 제한한 exact variant와 unrestricted
   heuristic을 모두 평가해 제한 손실을 측정한다.

이 확장은 단순한 기능 추가가 아니다. 새 실측이 직접 요구한 **operator- and
topology-aware placement**이며, TPDS 관점에서 가장 알고리즘적인 기여 후보다.

다만 subset DP 자체는 Held–Karp 계열의 알려진 설계 패턴이다. novelty는 단순히
`2^K` 상태를 썼다는 데 두지 않는다. 다음 결합이 선행연구에 없는지를 별도 문헌표로
검증해야 한다.

- causal position에 따라 변하는 microbatch service vector
- ~~노드별 operator-family 비용의 순위 역전~~ (§7.15 에서 철회)
- pairwise link topology
- 연속 att/ff 경계와 노드 순서의 동시 선택
- component-wise completion-vector dominance

이 중 결합 차별성이 약하면 subset placement는 구현 기여로 낮추고, 논문의 알고리즘
기여는 position-dependent completion-vector partitioning에 둔다.

---

## 5. `(N,B)`를 어떻게 고르는가

### 5.1 `N`은 가능한 값을 모두 본다

PP는 3, 5, 6, 7개 stage에서도 동작하므로 이유 없이 `{1,2,4,8}`만 보면 공동 최적
주장이 성립하지 않는다. 초기 구현은 `N=1..N_max`를 모두 평가한다.

현재 `derivepp.py`의 `N<8`은 고정 `STAGE_NODES`의 앞 `N`개만 사용한다. 이는 가장
좋은 `N`개 노드를 고른 결과가 아니므로 `N*` 검증으로 사용할 수 없다. §4.6의 subset
DP가 활성 node set까지 선택한 뒤에야 공정한 `N` 비교가 된다.

`N≤floor(S/B_min)`은 빠른 screening 힌트로는 유용하지만 하드 제약은 아니다.
`M<N`인 파이프라인도 정확히 실행되고, 특정 이종 구성에서는 여전히 최선일 수 있다.
비용 모델이 fill/drain을 직접 계산하므로 가능하면 pruning하지 않고 모델이 선택하게 한다.

### 5.2 새 결론 — `B`는 닫힌식이 아니라 캘리브레이션 격자에서 고른다

실측 `c(B)`는 매끄럽지 않다. 특히 `B=24`의 처리율 저하가 8개 노드에서 반복됐다.
따라서 `c(B)=τ+κB`를 적합해 `B₀∝√S`를 계산하면 실제 kernel notch를 지운다.

최종 알고리즘은 더 단순하다.

```text
Pi 5에서 측정한 ℬ_cal = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128}
유효 후보 ℬ = ℬ_cal ∩ [1, graph_batch_cap]

for B in ℬ:
    end-to-end 실행이 아니라 캘리브레이션 테이블로 C와 D를 합성
    DP로 예측 makespan 계산
가장 작은 예측 makespan의 B 선택
```

이것은 실제 요청을 반복하는 end-to-end sweep이 아니다. 짧은 kernel calibration에서
플랫폼의 불연속 비용곡선을 한 번 얻고, 모델 구조와 파이프라인 recurrence로 결과를
계산하는 **analytical table evaluation**이다. 제목의 `sweep-free`는 오해를 막기 위해
본문에서 항상 **end-to-end-sweep-free**로 풀어 쓴다.

이 숫자 목록 자체는 알고리즘 상수가 아니다. 새 플랫폼의 calibrator는 4행 정렬을
따르는 작은 값과 로그 간격의 큰 값을 측정하고, 메모리/그래프 상한까지 table을 만든다.
현재 세 길이에서는 `B=16`이 모두 이겼다. 그러나 이를 모든 `S`, `N`, 모델, CPU에
성립하는 법칙으로 고정하지 않는다. planner는 계속 전체 격자를 평가한다. 다른 CPU에서
커널 포화점이나 notch가 달라져도 알고리즘을 바꾸지 않고 calibration table만 바뀐다.

### 5.3 `B_ε`와 `B₀`의 남은 역할

두 식은 최종 선택 규칙에서는 제외하지만 직관과 sanity check로 남긴다.

```text
M_ε = ceil((1−ε)(N−1)/ε)                  # 버블 예산이 요구하는 M
B₀  = sqrt(S_exec·τ/((N−1)·κ))            # 매끄러운 이상 비용의 stationary point
```

- `B_ε`는 선택한 `B`의 이상적 버블이 얼마나 큰지 설명한다.
- `B₀`는 측정 테이블의 선택이 이상식과 크게 어긋나는지 진단한다.
- 둘 다 실제 `B`를 결정하거나 후보를 제거하지 않는다.

따라서 `B∝S`, `B∝√S`, `B=B_min` 중 어느 것도 보편 법칙으로 주장하지 않는다.
현재 관측은 “테스트한 범위에서 table evaluator가 `B_min=16`을 골랐다”이다.

### 5.4 전체 planner 의사코드

```text
input : model structure θ=(L,d,d_ff,d_kv,...)
        prompt length S
        calibrated compute functions {T_q,k,T_a,k}
        calibrated pairwise links {D_v,w}
        available nodes V, calibrated B grid ℬ

best = ∞
S_exec = tile_padding(S)
for N in 1..min(N_max,L):
    for B in calibrated_grid(S_exec):
        make chunks from S_exec: {(u_j,b_j,ℓ_j,A_j)}
        synthesize C_v,j(att), C_v,j(ff), D_v,w(j)
        π, p, T_core = placement_partition_dp(N,B,C,D)
        T_pred = T_core + final_return_cost(N,B)
        if T_pred < best:
            best = (T_pred,N,B,π,p,padding)

return best
```

출력 계획은 모델 로드 전에 stage별 가중치 배치를 정할 때 사용한다. 이미 모델을
로드한 뒤 매 요청마다 `N`이나 경계를 바꾸려면 가중치 이동이 필요하므로, 서버 모드에서는
§9의 Schedule Atlas로 길이 구간별 계획을 미리 컴파일한다.

---

## 6. 비용 모델과 해석적 예측

### 6.1 플랫폼 입력은 두 계산 함수와 링크 함수다

```text
T_q,k(F,B)        = τ_q,k(B) + F/P_q,k(B)          Q4×Q8 GEMM 계열
T_a,k(F,B,prefix) = τ_a,k(B,prefix) + F/P_a,k(B,prefix)
D_k(x)            = α_k + x/β_k
```

`B_min,k`, `ρ_k=P_q,k/P_a,k`, `τ/κ`, 노드 속도 scalar는 독립 입력이 아니라 이
함수들에서 얻는 요약값이다. 실제 구현에서는 측정점을 보간한 작은 lookup table 또는
구간별 회귀식을 사용한다.

**측정 계약**은 다음과 같다.

- 모든 throughput은 multiply-add를 2 operations로 세는 동일 convention 사용
- 워밍업 제외, 최소 반복 수와 중앙값 고정
- CPU affinity, thread 수, 주파수 governor 기록
- 비용식 회귀에 사용한 길이와 held-out 길이 분리
- node별 함수와 링크별 `α,β`를 저장하고 재현 artifact에 포함

#### 현재 calibrator에서 먼저 고쳐야 할 것

10~20% 과소예측을 곧바로 RMSNorm 등의 합으로 단정할 수 없다. 코드 대조 결과
calibration path와 production path 사이에 다음 차이가 있다.

| 차이 | 현재 상태 | 필요한 수정 |
|---|---|---|
| attention core | `calibrate.cpp`가 production online-softmax를 완전히 재사용하지 않고 별도 축약 loop를 구현 | 실제 `multiheadAttBatch_F32`를 공용 wrapper로 호출하거나 production op를 직접 계측 |
| projection shape | `4096×4096` 한 형상을 Q/K/V/O에 공통 사용 | Q/O와 K/V 형상을 분리 |
| lm_head | planner가 `proj` 처리율로 `vocab×d` GEMV를 추정 | lm_head 형상을 별도 측정 |
| 주변 연산 | norm, RoPE, cast, residual이 빠짐 | production subblock macrobench 또는 독립 `γ_op·b` 측정 |
| 다른 모델 | calibrator 형상이 Llama-3 8B에 hard-code | model config로 형상을 생성하거나 모델별 짧은 calibration 수행 |
| 링크 | TCP echo의 RTT/2 | 실제 `sendActivation` protocol의 header/copy/queue를 trace로 검증 |

따라서 다음 모델 갱신은 end-to-end 잔차에 `γ`를 맞추는 회귀가 아니다. 먼저 production
code path를 재사용하는 **calibration equivalence gate**를 통과시킨다. raw kernel이
같아도 barrier-free macrobench와 step-by-step executor는 다른 시간을 낼 수 있으므로
kernel equivalence와 executor equivalence를 별도 gate로 둔다. 그 뒤에도 남는 잔차만
구조적으로 분리한다. 이 순서가 zero-shot 주장을 지키면서 원인을 정확히 가른다.

### 6.2 Attention–FFN 비용비

한 레이어 전체에서 projection을 포함한 attention 서브블록과 FFN의 시간비를
긴 `S`에 대한 leading term으로 정리하면

```text
R_k(S) = T_att,k/T_ff,k
       ≈ 2(d+d_kv)/(3·d_ff) + ρ_k·S/(3·d_ff)

ρ_k = P_q,k/P_a,k
```

첫 항은 Q/K/V/O projection, 두 번째 항은 causal QK/AV다. `R<1`이면 FFN이 더
비싸고, `R>1`이면 attention이 더 비싸다. 이 비율이 길이에 따라 변하므로 고정
레이어 배분이 긴 입력에서 틀어질 수 있다.

초기 독립 마이크로벤치 값 `P_q=209 GOPS`, `P_a=21.4 GFLOPS`, 즉 `ρ=9.77`을 넣은
사전 예측은 다음과 같았다.

| `S` | 예측 `R(S)` | 실측 | 상대오차 |
|---|---:|---:|---:|
| 447 | 0.340 | 0.330 | **3.1%** |
| 1,789 | 0.644 | 0.597 | **7.8%** |

두 점은 식을 맞추는 데 사용하지 않았으므로 fitting이 아니라 독립 검증이었다. 그러나
새 raw-kernel calibration의 평균 `ρ_k`는 13.26이고 노드 범위도 10.13~15.57이다.
이는 단순히 더 좋은 새 값으로 교체할 문제가 아니다. 두 측정이 서로 다른 code path와
포함 연산을 재고 있기 때문이다. §6.1의 calibration equivalence gate 전에는 9.77과
13.26을 같은 정의의 상수처럼 섞지 않는다.

최종 planner는 공통 `R(S)` 하나가 아니라 노드별 비용 함수 전체를 사용한다. `R_k(S)`는
왜 길이에 따라 노드의 상대 순위가 바뀌는지를 설명하는 요약량이다.

### 6.3 역전점

노드 `k`에서 `R_k(S_k*)=1`인 길이는

```text
S_k* = (3·d_ff−2(d+d_kv))/ρ_k
```

이다.

| 모델 | `d` | `d_ff` | `d_kv` | 초기 예측 `S*` (`ρ=9.77`) |
|---|---:|---:|---:|---:|
| Llama-3.2 3B | 3,072 | 8,192 | 1,024 | **1,677** |
| Llama-2 13B (MHA) | 5,120 | 13,824 | 5,120 | **2,149** |
| Llama-3 8B | 4,096 | 14,336 | 1,024 | **3,354** |
| Llama-3.1 70B | 8,192 | 28,672 | 1,024 | **6,918** |

이 표는 새 calibration 이전에 등록한 **초기 기준 예측**으로 보존한다. 최종 예측에서는
노드마다 `S_k*` 범위가 생긴다. `S*`는 파라미터 수에 단조가 아니다. 예를 들어
Llama-2 13B는 MHA라 `d_kv=d`이고,
8B보다 큰 모델인데도 예측 역전점은 더 짧다. 측정 전에 이 값을 고정하고 실제 역전점이
같은 방향으로 움직이는지 확인하면 모델 구조식의 강한 반증 가능 검증이 된다.

### 6.4 비용 오차에 대한 보장 범위

고정 후보 집합의 모든 계산·통신 event duration `z`에 대해

```text
(1−δ)z ≤ z_hat ≤ (1+δ)z
```

가 성립한다고 하자. recurrence는 음수가 아닌 `max-plus` 식이므로 각 계획의 예측
makespan도 같은 범위에 있다. 따라서 **같은 후보 집합 안에서** 모델이 고른 전체 계획
`q_hat=(N,B,π,p)`과 실제 최적 계획 `q*` 사이에는

```text
T(q_hat) ≤ (1+δ)/(1−δ) · T(q*)
```

가 성립한다. `δ=5%`면 상한은 1.105배다.

이 보장은 다음을 포함하지 않는다.

- 후보 집합에서 빠진 `B` 또는 노드 배치
- 측정하지 않은 공유 NIC contention
- 시간에 따라 변하는 background load
- 비용 오차가 일부 점에서 `δ`를 넘는 경우

따라서 이론 보장과 measured best-of-grid attainment를 함께 보고한다.

---

## 7. 현재 실측이 말해 주는 것

### 7.1 선형 청크 법칙은 반증됐다

`S=7,212`, PP8, 실행 전 worker 단일성 확인, 순서를 반대로 한 교차 2라운드 결과다.
시간 단위는 ms다.

| `B` | 1라운드(내림차순) | 2라운드(오름차순) |
|---:|---:|---:|
| 32 | 196,546 | 198,482 |
| **48** | **187,452** | **198,333** |
| 64 | 206,596 | 203,601 |
| 96 | 216,688 | 208,323 |
| 128 | ~~361,907~~ (기각) | **229,782** |

**1라운드의 `B=128`은 기각한다.** 그 세션의 **첫 실행**이고, 콜드 스타트가 첫 실행만
일관되게 이상치로 만든다는 것은 이미 기록된 사실이다([13 §3](13-methodology.md):
49,992 / 34,077 / 34,824 / 34,418 / 32,414). 다른 네 점은 두 라운드가 4% 이내로
일치하는데 이 점만 57% 벌어진다.

2라운드는 `B=48` 이후 단조 증가하는 깨끗한 곡선이다. 이 실험에서 안전하게 말할 수
있는 결론은 두 가지다.

1. `B=32`와 `B=48`은 2라운드에서 0.1% 차이이므로 **동률**로 봐야 한다.
   최소점을 둘 중 하나로 특정할 수 없다.
2. `B∝S`가 예측한 128은 최소 구간보다 **1.16× 느리다**(229,782 / 198,333).
   선형 법칙은 반증되지만 **손해는 16%이지 두 배가 아니다.**

이 표만으로 최소점을 32 또는 48로 특정할 수는 없었다. 뒤의 §7.3에서 planner가
아직 측정하지 않은 `B=16`을 선택했고, §7.4의 같은 바이너리 재측정에서 16이 더
빠른 것으로 확인됐다. 따라서 이 절은 **선형 법칙을 기각한 중간 실험**으로만 읽는다.
`τ+κB`의 매끄러운 비대칭 곡선이나 `√S` 대안은 §7.2의 비매끄러운 kernel table이
관측된 뒤 최종 선택 모델에서 제외됐다.

### 7.2 노드별 커널 캘리브레이션 — 연산자 순위가 역전된다

`prefill_bench/calibrate.cpp` 로 8노드 전부에서 두 커널 계열을 측정했다
(4스레드, 내부 반복으로 구간당 최소 2×10¹⁰ FLOP, 외부 3회 중앙값).

**측정 자체에서 결함 두 개를 먼저 잡았다.**

1. 매 반복마다 `std::thread` 를 만들면 생성 비용(코어당 50~100 µs)이 통째로
   고정비 `τ` 로 잡힌다. 실제 커널은 상시 스레드 풀 위에서 돈다.
   `B₀ = √(S·τ/((N−1)κ))` 가 `τ` 에 직접 의존하므로 후보 `B` 를 통째로 틀리게 만든다.
   스레드를 한 번만 만들고 내부 반복으로 바꿨다.
2. 타이밍 구간이 9 ms 면 스케줄러 지터로 45 % 까지 흔들린다(`B=16` 300 vs `B=24` 207).
   구간을 키우자 재현된다.

#### (a) `B_min = 16` 이 마이크로벤치에서 독립 확인됐다

`ffn13` 형상(14336×4096), GFLOPS:

| node | B=4 | B=8 | B=16 | **B=24** | B=32 | B=48 | B=64 | B=128 |
|---|---|---|---|---|---|---|---|---|
| 103 | 131 | 133 | 299 | **212** | 279 | 296 | 295 | 295 |
| 104 | 135 | 135 | 306 | **216** | 304 | 303 | 301 | 300 |
| 113 | 111 | 110 | 276 | **178** | 267 | 261 | 266 | 263 |
| 166 | 94 | 94 | 244 | **153** | 238 | 238 | 239 | 237 |
| 191 | 112 | 112 | 291 | **186** | 290 | 286 | 285 | 281 |
| 54 | 130 | 132 | 297 | **212** | 297 | 290 | 295 | 294 |
| 94 | 136 | 135 | 305 | **215** | 304 | 304 | 301 | 301 |
| root | 135 | 136 | 280 | **213** | 276 | 273 | 277 | 280 |

`B≤12`는 효율 구간의 약 절반이고, `B=16`은 처음으로 높은 처리율에 도달한다.
`B≥32`는 대체로 plateau지만 `B=24`에는 별도의 notch가 있다. 따라서 여기서
`B_min=16`은 “그 이상이 모두 매끄럽게 포화한다”는 뜻이 아니라 **효율적인 첫 격자점**이다.
end-to-end에서 얻은 knee가 독립 kernel calibration에서도 다시 나온 것은
end-to-end-sweep-free 선택의 근거가 된다.

#### (b) `c(B)` 는 매끄럽지 않다 — 적합 대신 테이블

**`B=24` 가 이웃(16, 32)보다 낮은 것이 8/8 노드에서 재현된다.** 노이즈가 아니라
커널 특성이다.

> 따라서 `c(B)`를 매끄러운 `τ+κB`로 적합해 최종 `B`를 고르면 안 된다.
> `B₀` 식은 sanity check로만 쓰고, 최종 비용은 측정 격자의 table lookup으로 읽는다.

#### (c) `ρ_k` 편차 23.6 % — 공통 `ρ` 는 쓸 수 없다

`ρ_k = P_q,k(B=32, ffn13) / P_a,k(B=32, prefix=2048)`:

| node | 103 | 104 | 113 | 166 | 191 | 54 | 94 | root |
|---|---|---|---|---|---|---|---|---|
| `ρ_k` | 13.29 | **15.57** | 10.99 | **10.13** | 11.99 | 14.98 | 14.95 | 14.20 |

평균 13.26, 범위 10.13~15.57, **최대 편차 23.6 %.**
§0.4 의 분기에서 **노드별 `{P_q,k, P_a,k}` 를 따로 들고 가는 쪽으로 확정**한다.

#### (d) 연산자별 노드 순위가 역전된다 — ⚠️ **재현 실패, §7.15 참조**

> **이 절의 결론은 철회됐다.** 아래는 당시 관측 기록으로만 남긴다.
> 9노드 동시 수집 재측정에서 τ = +0.111 이 나왔고, 이 절이 DP·placement 를
> 정당화하는 근거로 쓰이던 것을 §7.15 에서 취소했다.


최고 노드를 1.00 으로 정규화한 상대 속도:

| node | GEMM | attention | 차이 |
|---|---|---|---|
| 103 | 0.92 | 0.74 | −0.17 |
| **104** | **1.00** | **0.69** | **−0.31** |
| 113 | 0.88 | 0.81 | −0.06 |
| **166** | **0.78** | **0.93** | **+0.15** |
| 191 | 0.95 | 1.00 | +0.05 |
| 54 | 0.98 | 0.88 | −0.10 |
| 94 | 1.00 | 0.90 | −0.09 |
| root | 0.91 | 0.91 | 0.00 |

```
Kendall tau = −0.29   (일치쌍 10/28)
```

**순위 상관이 음수다.** `166` 은 GEMM 최하위인데 attention 2위, `104` 는 GEMM 1위인데
attention 최하위다.

이것이 왜 결정적인가:

- 기존 프로파일은 `.166` 을 "45.4 ms/layer 로 최악" 이라 판정하고 서브블록을 적게 줬다.
  그러나 그 프로파일은 **S=447 에서 잰 것**이고 그 길이에서는 FFN 이 지배하므로
  사실상 `P_q` 만 반영한다.
- 긴 프롬프트에서 attention 이 절반을 넘으면 **`166` 은 오히려 좋은 노드가 된다.**
  즉 현재 배정 `9,8,6,8,9,9,9,6` 은 긴 프롬프트에서 틀린 노드에 적게 주고 있다.
- **노드 이종성을 스칼라 `s_k` 로 표현할 수 없다는 직접 증거**다. 연산자별 속도가
  다르고 위치에 따라 연산자 비중이 바뀌므로, 위치·연산자 의존 비용을 그대로 들고 가는
  완료시각 DP 가 필요하다.
- 부수 효과: §8.4 에서 인위적으로 만들려던 H2/H3(`P_a` 만 저하 / `P_q` 만 저하)이
  **자연적으로 이미 존재한다.** 통제 실험은 이 자연 대비를 확대하는 용도로 바뀐다.

#### (e) 남은 정리

`P_a` 에 이상치가 있다(`191` 의 prefix=512 가 13.8, 다른 값은 21~24).
반복을 늘려 재측정하고, `T_a(B, prefix)` 의 prefix 의존성을 확정해야 한다.

### 7.3 첫 held-out 결과 — `B` 선택은 적중, joint plan은 미검증

`prefill_bench/derivepp.py` 로 `S=7,212`, `N=8` 의 계획을 계산했다.
입력은 캘리브레이션 TSV 와 모델 config 뿐이고 end-to-end 실행은 하지 않았다.

#### (a) DP 정확성 — brute force 와 6/6 일치

`L=8` 로 줄인 인스턴스에서 모든 연속 분할을 전수 평가한 값과 대조:

| N | B | DP | brute force | frontier |
|---|---|---|---|---|
| 2 | 16 | 1515.376 ms | 1515.376 ms | ≤2 |
| 2 | 32 | 1536.488 | 1536.488 | ≤3 |
| 3 | 16 | 1156.659 | 1156.659 | ≤3 |
| 3 | 32 | 1168.897 | 1168.897 | ≤3 |
| 4 | 16 | 1003.400 | 1003.400 | ≤3 |
| 4 | 32 | 929.692 | 929.692 | ≤4 |

보조정리 1의 증명과 일치하도록 fixed-order DP가 구현됐는지를 확인한 첫 단위시험이다.
다만 현재 simulator와 brute force가 모두 §3.3의 단순 링크식을 공유하므로 직렬 링크를
포함한 최종 recurrence 검증은 아니다.

#### (b) frontier 는 폭발하지 않는다

실제 규모(`L=64, N=8, M=226`)에서 `Φ` 최대 **879**. 연속 분할 수가
`C(63,7) ≈ 5.5×10⁸` 인데 frontier 는 수백 수준이다.

> 관측한 이 인스턴스에서는 exact completion-vector DP를 사용할 수 있다. 그러나
> 이론적으로 frontier가 폭발하지 않는다는 결과는 아니며, 다른 `S`, 모델, placement를
> 포함한 분포와 timeout guard를 확인하기 전에는 fallback을 삭제하지 않는다.

#### (c) 예측이 적중했다 — `B=16`

플래너는 `B=16` 을 최적으로 지목했다. **그 지점은 그때까지 측정한 적이 없었다.**

| B | 예측(s) | 실측 r1 | 실측 r2 |
|---|---|---|---|
| **16** | **149.8** | **172.1** | **166.5** |
| 24 | 168.4 | 190.6 | 197.5 |
| 48 | 159.5 | 192.0 | 198.0 |

`B=16` 이 두 라운드 모두에서 최적이고, 이전 최선보다 **11~16 % 빠르다.**
`B=24` 와 `B=48` 은 예측에서 근접했고 실측에서도 0.2~0.7 % 로 사실상 동률이다 —
**순위가 의미 있는 곳에서 맞았다.**

| 규칙 | 고르는 `B` | 실측 |
|---|---|---|
| 기존 선형 `S/(8N)` | 128 | 229.8 s |
| `√S` | ~60 | 203.6 s (B=64) |
| 이전까지 최선 | 48 | 198.0 s |
| **DerivePP의 B selector** | **16** | **166.5 s** |

선형 규칙 대비 **1.38×**, `√S` 대비 **1.22×**, 이전 최선 대비 **1.19×**.

이 실측은 동일한 기존 노드 순서와 실행 가능한 분할 위에서 `B`만 바꾼 결과다.
문서에는 예측 분할 `p*`를 실제로 배포해 비교한 기록이 없다. 따라서 검증된 것은
**청크 선택과 순위 예측**이며, completion-vector partition이나 placement의
end-to-end speedup은 아직 0이다. 이후 평가에서는 `B-only`, `partition-only`,
`placement-only`, `joint`를 분리한다.

#### (d) 계통 편향 −14~−20 %

| B | 예측 | 실측(r2) | 오차 |
|---|---|---|---|
| 16 | 149.8 | 166.5 | −10.0 % |
| 32 | 167.6 | 198.5 | −15.6 % |
| 48 | 159.5 | 198.3 | −19.6 % |
| 64 | 169.7 | 203.6 | −16.7 % |
| 96 | 178.9 | 208.3 | −14.1 % |
| 128 | 184.3 | 229.8 | −19.8 % |

**무작위라기보다 일관된 과소예측이다.** 빠진 RMSNorm·RoPE·cast·residual이 후보지만,
production attention과 calibration surrogate의 차이, lm_head 형상 대용, 실행기 비용,
링크 protocol 차이도 남아 있으므로 원인을 아직 확정하지 않는다.

> ⚠️ **이 측정값에 맞춰 `γ` 를 적합하면 fitting 이 되어 zero-shot 주장이 무너진다.**
> production-path calibration을 먼저 맞추고, 그 뒤 남는 연산을 직접 추가해 재는 쪽으로 간다.

#### (e) `√S` 서사는 철회한다

과거 `S=2,048`의 최선이 32였다는 결과는 logit backpressure 수정 이전 바이너리에서
나왔다. 서로 다른 시스템 상태의 점을 연결해 `B∝√S`를 만든 것이 오류였다. 현재
바이너리의 재측정 결과는 바로 다음 절에 정리한다. 최종 planner는 `√S` 후보식을
사용하지 않고 calibration grid 전체를 모델로 평가한다.

### 7.4 관측 범위에서는 `B*=B_min`

과거 “512/2,048 설정”으로 부르던 prompt를 현재 바이너리로 다시 측정했다. 아래 표의
`S`는 tokenizer를 거친 **실제 prefill token 수 447/1,789**를 쓴다. 교차 2라운드,
`--n-batches 256` 고정이다.

| S | B=8 | **B=16** | B=32 | B=48 |
|---|---|---|---|---|
| 447 (r1/r2) | 10,723 / 10,747 | **6,601 / 6,142** | 8,040 / 7,679 | 8,981 / 8,786 |
| 1,789 (r1) | 45,255 | **25,828** | 29,752 | 29,714 |
| 7,212 (r2) | — | **166,508** | 198,482 | 197,969 |

**테스트한 세 길이, N=8, 현재 모델·커널·프로토콜에서는 모두
`B*=16=B_min`이었다.** 이 범위 안에서는 관측된 `S` 의존성이 없다.

이것은 캘리브레이션에서 직접 예측되는 결과다(§7.2a). `P_q` 가 `B=16`~`128` 에서
평평하므로 `B` 를 키워 얻을 커널 이득이 없고, `M` 만 줄어 버블이 커진다.

이 결과는 `B_min`을 강제로 반환하는 보편 법칙이 아니다. `N`, 모델 형상, CPU 또는
production kernel이 달라지면 table의 최솟값도 바뀔 수 있다. 그래서 §5의 planner는
`B_min`만 반환하지 않고 calibration grid 전체를 평가한다.

#### 세 번 철회한 법칙

| 시점 | 주장 | 철회 사유 |
|---|---|---|
| 초기 | `B ∝ S` (`S/(8N)`) | `B=128` 이 최소보다 느림 (§7.1) |
| 중기 | `B ∝ √S` | `B*` 가 512→16, 2048→32, 7212→16 로 단조가 아님 |
| **현재 관측** | **세 점에서 `B* = B_min`** | 재측정 3점 일치. 보편 법칙으로 승격하지 않음 |

앞의 두 번은 모두 **로짓 역압 수정 이전 데이터** 위에 세운 것이었다. 그때는 마지막
스테이지가 매 청크마다 로짓 513 kB 를 보냈으므로 `M` 이 큰 작은 `B` 가 부당하게
불리했다(§12).

> **교훈**: 시스템을 고친 뒤에는 그 이전 데이터로 세운 법칙을 전부 다시 세워야 한다.
> 두 번의 철회가 같은 원인에서 나왔다.

### 7.5 `nBatches` 와 `seqLen` — 자유도인가 교란인가

`nBatches` 는 모든 버퍼의 행 수를 정한다(`src/llm.cpp`).

| 버퍼 | nBatches=32 | nBatches=256 |
|---|---|---|
| att 스크래치 (`nBatches×nHeads×seqLen`) | 34.6 MB | **277 MB** |
| logits (`nBatches×vocab`) | 16.4 MB | **131 MB** |
| x/y/z/q/k/v/d 등 | ~5 MB | ~40 MB |

코드를 확인한 결과 **실행 비용은 `nBatches` 에 비례하지 않는다.**

- 전송량: `payloadBytes = xPipeRowBytes × batchSize` — 실제 배치 기준
- 실행 루프: `nBatches` 를 쓰는 곳은 `mergeSum_F32` 의 스트라이드 인덱싱뿐,
  루프는 `batchSize` 로 돈다

따라서 남는 경로는 할당·첫 접촉·TLB 압박이다.

**실측 결과: 고정비 ~500 ms 이고 처리량 효과가 아니다.**

| S | nBatches=32 | nBatches=256 | 차이 |
|---|---|---|---|
| 447 (r1/r2) | 6,431 / 6,083 | 6,766 / 6,698 | **+335 / +615 ms** |
| 7,212 (r1/r2) | 168,019 / 162,130 | 167,433 / 162,975 | −586 / +845 (잡음) |

짧은 프롬프트에서는 8 %, 긴 프롬프트에서는 0.3 % 로 검출되지 않는다.
**할당·첫 접촉 비용**이지 청크당 비용이 아니며, `nBatches` 로 도는 실행 루프가
없다는 위 코드 확인과 일치한다.

#### 그렇다면 24 % 는 어디서 왔나 — 세션 드리프트다

같은 `B=16, S=447` 구성이 세션에 따라 **5,305 ms ~ 6,601 ms** 로 나온다.
`nBatches` 로 설명되는 부분은 2.6 % 뿐이므로 나머지는 세션 간 드리프트다.

> ⚠️ **세션 간 절대값 비교는 무효다.** 이미 [측정 방법론](13-methodology.md)이 요구한 규칙이지만,
> `B` 스윕 해석에서 내가 두 번 어겼다(첫 스윕의 `B=64`/`B=128` 이상치, 그리고
> "nBatches 가 24 % 를 먹는다"는 잘못된 귀속). **모든 비교는 같은 세션 교차 실행으로만
> 한다.**

#### 현재 판단 — 평가에서는 고정하고, 배포에서는 bucket으로 관리한다

현재 한 점만으로 `nBatches`와 `seqLen`이 작을수록 항상 좋다고 확정할 수 없다.
우선 비교 실험에서는 교란을 없애기 위해 다음처럼 고정한다.

```
단일 길이 평가:   nBatches = max(ℬ),   seqLen = 고정된 동일 값
최종 선택 재실행: nBatches = B*,       seqLen = S_exec 또는 사전 등록한 bucket 상한
```

`nBatches=B*`, `seqLen=S_exec`은 모델을 다시 만드는 단일 요청 benchmark에서는
가능하지만, 다양한 길이를 받는 서버에서는 요청마다 graph와 buffer를 다시 만들 수 없다.
실제 Atlas는 `seqLen`과 `nBatches`도 소수의 bucket으로 컴파일하고 메모리 사용량을
함께 보고해야 한다.

**단, 이 주장을 쓰려면 "작을수록 좋다"가 단조인지 확인해야 한다.**
`nBatches ∈ {16, 32, 64, 128, 256}` 스윕이 남아 있다. 비단조면(예: 정렬 효과)
후보 탐색이 필요하다.

#### `seqLen` 은 더 위험하다

att 스크래치가 `nBatches × nHeads × seqLen` 이므로 `--max-seq-len` 도 같은 축이다.
**지금까지의 스윕은 실제 프롬프트가 447 일 때도 `--max-seq-len 1024`, 7,212 일 때
`8448` 을 썼다.** 프롬프트별로 최소값을 쓰지 않으면 같은 손해를 본다.
이것은 **지금까지의 모든 절대 수치에 걸친 잠재 교란**이므로 별도로 측정한다.

### 7.6 계통 편향 — 레이어 단위로 올려도 22%가 남았다

§7.3의 과소예측을 연산자별로 쪼갰다. 단일 노드, `S=448`, `B=16`, 32레이어,
`M=28`에서 FFN은 다음과 같다.

| 예측 경로 | FFN 전체 | 실측 대비 |
|---|---:|---:|
| op 단위 커널 합 | 18.0 s | **−14 %** |
| barrier-free 레이어 macrobench (`20.9 ms × 32 × 28`) | 18.7 s | **−11 %** |
| production stage timing (**웜**, n=2) | **21.1 s** | 기준 |

> ⚠️ **이 표는 한 번 정정됐다.** 처음 쓴 기준값 `24.0 s` 는 **콜드 세션** 값이었고,
> 페이지 캐시가 채워진 뒤 `21.1 s` 로 수렴한다(§7.6a). 그때 격차는 각각
> −25 % / −22 % 로 보였으나 실제로는 −14 % / −11 % 다. **분모가 14 % 부풀려져 있었다.**

중앙값은 레이어 순서대로 연산자를 교대시켰을 때 0.7 s 증가했다. 그러나 두 값은 동일
조건의 paired A/B가 아니다. op 단위 값은 한 형상을 반복한 `calibrate.cpp`, 레이어 값은
가중치 사본 4개를 순회한 `calibrate_layer.cpp`에서 왔고, calibrator 세션 분산도 2.8%다.
관측 차이 3.7%가 이와 비슷하므로 0.7 s 전부를 “연산자 교대 효과”로 귀속하지 않는다.
안전한 결론은 **두 방식 모두 production보다 22% 이상 낮아 캘리브레이션 단위를 단순히
레이어로 올리는 것만으로는 부족하다**는 것이다.

코드 감사에서도 이 벤치가 executor와 동형이 아님을 확인했다. 생성된 각 thread가
RMSNorm부터 residual까지 자기 op 열을 독립적으로 끝까지 실행한다. 사용되지 않던
`Pool` 구조체는 제거했고, 주석도 **연산자 교대는 포함하지만 op 경계 배리어는 없는
macrobench**라고 수정했다. 이 벤치는 방안 A의 barrier-free 대조군으로 보존한다.

#### 가중치 working-set 가설은 기각됐다

`calibrate_layer.cpp`에서 서로 다른 FFN 가중치 사본 수를 늘려 레이어당·청크당 시간을
쟀다.

| 가중치 사본 | working set | FFN 시간 |
|---:|---:|---:|
| 1 layer | 122 MB | 20.90 ms |
| 2 layers | 244 MB | 20.07 ms |
| 4 layers | 488 MB | 20.90 ms |
| 8 layers | 976 MB | 20.89 ms |
| 16 layers | 1,952 MB | 22.19 ms |

working set을 16배로 늘려도 증가는 6%뿐이다. production과의 약 25% 차이를 DRAM
행 지역성이나 한 레이어 가중치 재사용으로 설명할 수 없다. 이 가설은 추가 fitting 없이
기각한다.

#### 남은 후보

| 후보 | 왜 남아 있는가 | 분리 방법 |
|---|---|---|
| executor step 동기화 | production `NnExecutor`는 op마다 모든 worker가 끝나야 다음 step으로 간다. macrobench는 각 thread가 전체 op 열을 독립 실행해 op 사이에 기다리지 않는다 | 같은 executor loop의 no-op step 비용과 production op별 thread 종료시각 측정 |
| thread load-imbalance 노출 | barrier가 있으면 `Σ_o max_t c_{o,t}`, 없으면 `max_t Σ_o c_{o,t}`에 가깝다. 둘의 차이는 빈 barrier 비용으로 잡히지 않는다 | op별 thread completion skew와 step wall time 비교 |
| KV cache write | macrobench attention은 K/V를 미리 채우고 읽기만 한다 | 동일 attention에서 KV append on/off A/B |
| graph-local copy | CAST, MERGE_ADD와 pipe buffer 이동이 macrobench 경로와 다르다 | step trace에서 copy op를 분리하고 bytes/time 기록 |

#### 실제 step 수 — 추정 20개가 아니라 레이어당 26개다

정적 graph 정의는 MoE, pruning, TP 변형 같은 조건부 op를 잘못 포함할 수 있다.
`DLLAMA_DUMP_STEPS=1`을 `NnExecutor`에 추가해 생성된 실행계획에서 직접 세었다.

```text
🧮 [STEPS] total=838 execute_op=838 sync_nodes=0 other=0

layer-local execute steps = 838 − 6 = 832
steps per layer           = 832 / 32 = 26
대략적인 분류             = attention 14 + FFN 12
```

`sync_nodes=0`은 단일 노드 실행이기 때문이다. 분산 graph에서는
`STEP_EXECUTE_OP`와 `STEP_SYNC_NODES`를 별도로 세고 별도 비용으로 모델링해야 한다.
후자는 순수 executor 경계뿐 아니라 실제 network wait/transfer를 포함할 수 있으므로
`h_step` 하나로 합치지 않는다.

#### 새 step 수로 배리어 크기를 다시 계산한다

관측된 FFN 차이는

```text
(24.0 s − 18.7 s) / 28 chunks = 189 ms/chunk
```

전체 레이어의 832개 execute step을 모두 FFN 잔차에 귀속하는 보수적인 상한은 다음과
같다.

```text
10 µs × 832 = 8.32 ms/chunk = 잔차의 4.4%
잔차 전체를 설명하는 데 필요한 평균 = 189 ms / 832 = 227 µs/step
```

하지만 189 ms는 FFN component의 잔차다. FFN 분류 약 12 op/레이어만 직접 대응시키면
더 엄격한 값은 다음과 같다.

```text
FFN steps = 12 × 32 = 384
10 µs × 384 = 3.84 ms/chunk = 잔차의 2.0%
잔차 전체를 설명하는 데 필요한 평균 = 189 ms / 384 = 492 µs/step
```

따라서 새 계측은 결론을 바꾸지 않고 강화한다. 순수 barrier latency가 주원인이라는
표현은 아직 성립하지 않으며, 빈 barrier보다 **동기화가 노출하는 thread 불균형**이 더
중요한 후보일 수 있다.

#### 결정 — C를 진단 gate로 먼저, 실패하면 A로 즉시 전환한다

세 선택지의 최종 판단은 다음과 같다.

| 방안 | 판단 |
|---|---|
| B. 전체 시간을 상수 비율로 보정 | **기각.** 순위를 보존할 수는 있어도 모델·`B`·노드가 바뀌면 검증되지 않은 fitting이 된다 |
| C. 배리어 비용 별도 측정 | **먼저 수행하되 진단용 gate로 제한.** 실제 `NnExecutor`의 atomic/yield step 경로를 그대로 사용해야 한다 |
| A. 실제 실행기로 한 레이어 실행 | **C가 잔차를 설명하지 못하면 채택.** synthetic weight를 쓰면 모델 파일 없이도 가능하며 end-to-end plan sweep이 아니다 |

C는 일반적인 `pthread_barrier` latency를 재는 실험이 아니다. 다음 세 값을 분리한다.

```text
T_null-op(K)    = 같은 NnExecutor로 K개의 no-op STEP_EXECUTE_OP을 실행한 시간
T_null-sync(K)  = no-op synchronizer로 K개의 STEP_SYNC_NODES를 실행한 시간
T_step-sum      = production op를 executor로 실행한 각 step wall time의 합
T_macro         = 같은 op 열을 barrier 없이 실행한 레이어 macrobench 시간

순수 execute-step tax ≈ T_null-op(K) − T_null-op(0)
순수 sync-step tax    ≈ T_null-sync(K) − T_null-sync(0)
동기화된 불균형 tax ≈ T_step-sum − T_macro − 순수 execute-step tax
```

측정 순서는 다음과 같다.

1. **완료:** 단일 노드 graph에서 `838 execute_op`, 반복 레이어당 26개를 확인했다.
2. 분산 graph에서도 stage별 `STEP_EXECUTE_OP`와 `STEP_SYNC_NODES` 수를 따로 기록한다.
3. production과 동일한 thread pool, atomic counter, `yield` loop에서 no-op execute step
   `K∈{0,1,8,32,128,384,448,832,838}`을 잰다. 384/448은 FFN/attention 추정 개수,
   832/838은 반복 레이어/전체 graph 실측 개수다.
4. `STEP_SYNC_NODES`는 no-op synchronizer와 실제 network synchronizer를 분리 측정한다.
5. op별 thread 종료시각과 executor step wall time을 기록한다.
6. 독립적으로 KV append on/off, CAST/MERGE_ADD on/off를 A/B한다.
7. 한 노드·한 `B`에서 얻은 항으로 다른 `B`, prefix, 노드의 잔차를 예측한다.

`DLLAMA_DUMP_STEPS` 출력은 graph 생성 확인용이며 timing run에서는 꺼서 출력 비용이
측정에 섞이지 않게 한다. step 수는 플랫폼 상수 하나가 아니라 `(N,π,p)`에 따라 생성된
stage graph에서 유도되는 값이고, 측정하는 플랫폼 상수는 step 종류별 단위 비용이다.

순수 step 항은 `K`에 대해 거의 선형이고, 측정하지 않은 조건에서 같은 방향으로 잔차를
줄일 때만 `X_k=n_step·h_step,k`로 비용 모델에 넣는다. 잔차의 50%도 설명하지 못하면
“배리어가 주원인”이라는 가설을 기각한다. 작은 실측 항 자체는 보존할 수 있지만,
**held-out component error가 15% 아래로 내려가지 않으면 원인 규명이 끝난 것이 아니므로
A로 넘어간다.**

A에서도 실제 모델 요청을 여러 plan으로 replay하지 않는다. production graph builder,
`NnExecutor`, CPU op, buffer/KV 경로를 그대로 사용하되 synthetic weight로 한 att/ff
서브블록만 짧게 실행한다. 따라서 zero-tuning의 정의인 “수동 튜닝과 end-to-end plan
sweep 없음”은 유지된다. 달라지는 것은 캘리브레이션이 production executor와 동형이
된다는 점이다.

#### C 결과 — 순수 step 비용은 잔차의 0.3 % 다 (기각)

`prefill_bench/bench_steps.cpp` 로 **실제 `NnExecutor`** 에 1×1 CAST no-op step 을
`K` 개 넣고 잰다. 별도 barrier loop 가 아니라 production 의 atomic/yield step 경로를
그대로 쓴다. `NnFakeNodeSynchronizer` 로 network 항은 제외했다.

| K | 8 | 32 | 128 | 384 | 448 | 832 | 838 |
|---|---|---|---|---|---|---|---|
| µs/step | 5.17 | 2.02 | 1.36 | 1.39 | 1.25 | 1.23 | 1.32 |

```text
T(K) = a + h·K      a = 0.025 ms,   h = 1.246 µs/step
```

| | 값 |
|---|---|
| 실측 step 비용 `h` | **1.246 µs/step** |
| FFN 384 step | 0.48 ms/chunk = 잔차 189 ms 의 **0.3 %** |
| 전체 832 step | 1.04 ms/chunk = 잔차의 **0.5 %** |
| 잔차 전체 설명에 필요한 값 | **492 µs/step** — 실측의 **395배** |

kill 기준("잔차의 50 % 도 설명하지 못하면 기각")을 압도적으로 미달한다.

> **순수 step 동기화 비용은 원인이 아니다.** 배리어보다 **thread load-imbalance**
> 를 유력 후보로 둔 판단이 옳았다. `Σ_o max_t c_{o,t}` 와 `max_t Σ_o c_{o,t}` 의
> 차이는 빈 step 비용으로 잡히지 않는다.

작은 실측 항 `h = 1.246 µs/step` 자체는 보존한다(`X_k = n_step · h`).
`n_step` 은 §7.6 의 런타임 덤프에서 유도된다.

**구현 메모**: `NnExecutorDevice` 는 `unique_ptr<NnDevice>` 로 소유권을 가져간다.
스택 객체 주소를 넘기면 소멸 시 delete 를 시도해 죽는다(ASan 으로 확인).

#### 따라서 A 로 전환한다

C 가 잔차를 설명하지 못했으므로 §7.6 의 결정 경로대로 A 를 채택한다.
그리고 `bench_steps.cpp` 가 **실제 `NnExecutor` 로 그래프를 만들어 돌리는 것을 이미
증명**했으므로 A 의 경로가 분명해졌다.

```text
C: NnExecutor + K개 1x1 CAST        -> step tax 만 측정 (완료, 기각)
A: NnExecutor + 실제 att/ff op 열   -> production 과 동형인 비용 함수
```

production graph builder 와 `NnExecutor`, CPU op, buffer/KV 경로를 그대로 쓰되
synthetic weight 로 att/ff 서브블록만 짧게 실행한다. 모델 파일이 필요 없고
end-to-end plan sweep 도 아니므로 zero-tuning 정의는 유지된다.

#### A 결과 — executor 동형으로 잔차의 1/3 을 메웠다

`prefill_bench/calibrate_exec.cpp` 는 production `buildLlmNet` 으로 그래프를 만들고
synthetic weight 를 주입한 뒤 실제 `NnExecutor::forward()` 를 돌린다.
모델 파일이 필요 없고 `(N,B,p)` 를 end-to-end 로 쓸어보지도 않으므로
zero-tuning 정의는 유지된다.

| 캘리브레이션 | 레이어당 (B=16, prefix≈224) | production 대비 |
|---|---|---|
| op 단위 커널 합 | ~24 ms | **−37 %** |
| 레이어 macrobench (배리어 없음) | ~28 ms | −26 % |
| **executor 동형 (방안 A)** | **31.2 ms** | **−18 %** |
| production 실측 | 38.0 ms | 기준 |

`prefix` 축 원자료(레이어 차분, `CAL_LAYERS=4`):

| prefix | 128 | 512 | 2,048 | 4,096 | 8,192 |
|---|---|---|---|---|---|
| ms/layer | 30.07 | 32.80 | 39.99 | 46.27 | 79.33 |

**executor 를 쓰자 −26 % 에서 −18 % 로 줄었다.** 배리어가 노출하는 thread imbalance 가
실재하고, 그 크기가 잔차의 약 1/3 이다. §7.6 이 배리어 자체보다 imbalance 를 유력
후보로 둔 판단이 여기서 확인된다.

#### 구현에서 잡은 두 가지 (재현 시 필수)

**1. `setDecodePhase(false)` 를 반드시 호출한다.**
설정하지 않으면 `lm_head` 가 decode 기본값으로 배치 전체 행을 계산한다
(prefill 은 마지막 1행). 이것 하나로 46.0 ms 가 나왔다 — 실제의 1.5배다.

**2. 레이어 수로 나누지 말고 차분한다.**
그래프에는 embedding·final norm·lm_head 같은 **레이어 무관 상수항** `c0` 가 있다.

```text
T(L) = c0 + L · c_layer      →      c_layer = (T(L2) − T(L1)) / (L2 − L1)
```

측정된 `c0` 는 19~70 ms 로 작지 않다. `T(L)/L` 로 나누면 이 값이 섞인다.

#### 남은 −18 % 의 후보

| 후보 | 왜 남아 있는가 |
|---|---|
| **production 기준값 자체의 세션 드리프트** | 34,013 ms 는 한 세션 값이고, 같은 구성에서 24 % 변동을 관측했다(§7.5). **같은 세션에서 재측정하기 전에는 −18 % 가 진짜인지 알 수 없다** |
| KV cache write | 벤치는 위치를 `prefix−B` 로 고정해 매번 같은 곳에 쓴다. 실제로는 청크마다 새 위치에 append 한다 |
| 파이프 간 복사 | 단일 노드라 `MERGE_ADD`/`CAST` 경로가 분산과 다를 수 있다 |

**첫 번째를 먼저 처리한다.** 나머지를 파기 전에 기준값을 같은 세션에서 다시 재야
한다. 그러지 않으면 §7.5 에서 이미 두 번 저지른 잘못된 귀속을 반복하게 된다.

#### 7.6a 기준값이 콜드였다 — 워밍업에 3라운드가 걸린다

`calibrate_exec` 와 production 을 **같은 세션에서 번갈아** 3라운드 돌렸다.

| 라운드 | production | calibrate_exec | 비 |
|---|---|---|---|
| r1 | 41.91 ms/layer | 30.65 | 0.73 |
| r2 | 36.89 | 31.08 | 0.84 |
| r3 | 34.58 | 31.69 | 0.92 |

**production 만 단조 감소하고 calibrate_exec 는 안정적이다.** production 을 5라운드
더 돌리면 수렴한다.

| r4 | r5 | r6 | r7 | r8 |
|---|---|---|---|---|
| 33.06 | 33.30 | 32.79 | 33.29 | 33.90 |

콜드 41.91 → 웜 33.3, **17 % 차이**다. 4.5 GB 모델 파일의 페이지 캐시가 채워지는
과정이며 `buff/cache` 가 7.3 GB 까지 오르는 것과 일치한다.

##### 격차는 −18 % 가 아니라 −6 % 다

| 기준 | 값 | calibrate_exec 대비 격차 |
|---|---|---|
| production 콜드 (r1) | 41.9 ms | −27 % |
| production 문서 최초값 | 38.0 ms | −18 % |
| **production 웜 (r4~r8 중앙값)** | **33.3 ms** | **−6.0 %** |
| calibrate_exec | 31.2 ms | — |

캘리브레이터 편차 3.4 %, production 편차 3.4 % 이므로 6 % 는 잡음의 2배 수준이다.
**Phase 1 통과 조건("예측 오차 대부분 15 % 이내")을 충족한다.**

##### 웜 기준 연산자별 분해 (n=2)

| 성분 | 웜 실측 | 콜드 실측(기존) | 차이 |
|---|---|---|---|
| prefillMs | 29,993 / 30,006 | 34,013 | −12 % |
| ffnMs | 21,032 / 21,147 | 24,031 | −12 % |
| attnProjMs | 4,782 / 4,660 | 5,237 | −10 % |
| attnMs | 2,386 / 2,405 | 2,688 | −11 % |
| otherMs | 1,112 / 1,122 | 1,489 | −25 % |

모든 성분이 균일하게 내려간다 — 특정 연산이 아니라 **전역 메모리 계층 효과**다.

##### 측정 규율에 추가한다

[측정 방법론](13-methodology.md)의 "rep1 을 버린다"로는 부족하다.
**모델 파일 페이지 캐시가 채워지는 데 3라운드가 걸린다.**

```text
새 규율: 모델을 바꾼 뒤 첫 3회는 워밍업으로 버린다.
         free -m 의 buff/cache 가 안정될 때까지 기다린다.
```

> 이것은 §7.5 가 경고한 잘못된 귀속의 **세 번째** 사례다(앞의 둘: 첫 `B` 스윕의
> 이상치, "nBatches 가 24 % 를 먹는다"). 이번에는 내가 콜드 기준값 위에 세 개의
> 비교(−25 / −22 / −18 %)를 쌓았고, 분모를 고치자 전부 바뀌었다.
> **과거 `prefillMs` 수치 중 워밍업 상태가 기록되지 않은 것은 재해석이 필요하다.**

##### 남은 6 % — 더 파지 않는다

KV write 와 파이프 복사가 후보로 남지만, 잡음의 2배 수준이라 수익이 낮다.
Phase 1 은 통과로 보고 비용 모델을 `calibrate_exec` 출력으로 교체한다.
이후 held-out 오차가 15 % 를 넘으면 그때 재개한다.

#### 7.6b 비용 모델 교체 후 오히려 나빠졌다 — `att_share` 가정이 원인

`calibrate_exec` 출력을 planner 에 넣고 `S=7,212`, `N=8` 을 돌렸다.

| B | 예측 | 실측(웜, 교차 2라운드) | 비 |
|---|---|---|---|
| 16 | 105.3 s | 166.5 ~ 172.1 s | **0.62** |
| 32 | 111.1 s | 198.5 s | 0.56 |
| 48 | 94.6 s | 192.0 ~ 198.0 s | **0.48** |

**−38 ~ −52 % 과소예측이고, 순위가 뒤집혔다.** 예측은 `B=48` 이 최적인데
실측은 `B=16` 이 최적이다. op 단위 모델(−16~−20 %, 순위 적중)보다 **나빠졌다.**

##### 원인 — `Lall` 을 비율로 쪼갠 것

`calibrate_exec` 는 레이어 전체 시간 `Lall(B, prefix)` 만 낸다. 이를 att/ff 로 나눠야
DP 가 서브블록 단위로 배정할 수 있는데, 그 분할을 상수 비율로 했다.

```text
att_share = 0.254   <- 웜 production stage timing, prefix ~= 224 조건
ff  = Lall(B, prefix0) * (1 - att_share)
att = Lall(B, prefix0) * att_share + [Lall(B, prefix) - Lall(B, prefix0)]
```

두 가지가 틀렸다.

1. `att_share` 는 `prefix ~= 224` 에서 잰 값이다. `S=7,212` 에서는 attention 비중이
   훨씬 크다(§6.2 의 `R(S)` 가 0.34 → 1.9).
2. "prefix 증분은 전부 attention 몫" 으로 두면, 기저값 `Lall(B, prefix0)` 안에 이미
   들어 있는 `prefix0` 만큼의 attention 이 `att_share` 로만 계상된다.
   **긴 프롬프트에서 attention 을 덜 세는 방향**이고, 관측된 과소예측·순위 역전과
   방향이 일치한다.

##### 조치 — 가정을 없앤다

비율을 고쳐 맞추는 것은 fitting 이다. 대신 **`calibrate_exec` 가 att/ff 를 나눠서
직접 재도록** 바꾼다.

```text
현재: Lall(B, prefix)                    -> 비율 가정 필요
변경: Latt(B, prefix),  Lff(B)           -> 가정 불필요
```

executor 는 그래프 전체를 한 번에 돌므로 세그먼트/op 종류별 시간을 뽑아야 한다.
production 의 `--stage-timing` 이 쓰는 `getLastForwardOpBreakdown()` 경로를 그대로
사용한다. 그러면 캘리브레이션과 production 이 **같은 계측기**를 쓰게 되어
비교 자체가 동형이 된다.

> **교훈**: executor 동형으로 올려 −6 % 를 얻었지만, 그 출력을 DP 가 쓸 수 있는
> 형태로 **가공하는 단계에서 새 가정을 넣었다.** 측정을 정밀하게 만들어도
> 그 뒤 가정 하나가 이득을 통째로 되돌릴 수 있다.

##### 부수 성과 — planner 성능 문제 해결

실규모(`M=451`)에서 계획 생성이 40분에도 끝나지 않았다. 병목을 세 번 잘못 짚은 뒤
프로파일링으로 찾았다.

| 시도 | 근거 | 결과 |
|---|---|---|
| `M` 순차 루프 벡터화 | prefix max 닫힌 형태 | 필요했지만 부족 |
| 상태를 `(Φ,M)` 행렬로 묶기 | numpy 호출 수 `Φ` 배 감소 | 필요했지만 부족 |
| 비용 계산 닫힌 식 + 캐시 | `stage_cost` 가 `O(e−i)` | 필요했지만 부족 |
| **프로파일링** | — | **`np.asarray` 629 k 회 = 92 s 중 63 s** |

dominance 가지치기가 `kept` 리스트를 **매 후보마다 배열로 재구성**하고 있었다
(`O(Φ²·M)` 복사). 미리 할당하고 뷰로 비교하자 **4배** 빨라졌다(92 s → 23 s).

```text
last_j = C_j + max_{i<=j}(a_i - C_i)      # recurrence 닫힌 형태
n_att(a,b) = ((b-a) + [a 짝수]) // 2      # 구간 내 att 개수 닫힌 식
kept 는 미리 할당하고 뷰로 비교            # O(Phi^2 M) 복사 제거
```

세 번 연속 추측이 빗나갔다. **성능 문제도 측정으로 접근해야 한다.**

#### 7.6c 레이어 수 회귀를 폐기하고 레이어별 직접 수집으로 간다

##### 회귀가 재현되지 않았다

같은 조건(`root, B=32, prefix=8192`)이 실행마다 달랐다.

| 실행 | 조건 | 값 |
|---|---|---|
| 콜드 | 54 °C 시작, 짧음 | **103.6 ms** |
| 웜 | 짧음 | 146.2 ms |
| 뜨거움 | 전체 45점 수집 | **302.5 ms** |

**2.9배다.** 온도는 54 → 76 °C 로 올랐지만 주파수는 2400 → 2200 MHz(8 %)에 그쳤다.
**8 % 주파수 저하로 2.9배를 설명할 수 없다.** 열은 기여하지만 지배 요인이 아니다.

##### 측정 시간을 늘리면 오히려 나빠진다

`inner` 를 5 에서 40 으로 늘려 각 3회:

| inner | 시도1 | 시도2 | 시도3 | CV |
|---|---|---|---|---|
| 5 | 131.4 | 178.2 | 149.7 | 15 % |
| **40** | 174.2 | **92.6** | 116.2 | **32 %** |

**긴 쪽이 더 불안정하다.** 랜덤 잡음은 줄지만 체계적 편향이 커져 순효과가 악화된다.

##### 원인 — 루프 순서가 `L` 과 시간·온도를 함께 움직인다

```text
for prefix = 128 -> ... -> 8192      (뒤로 갈수록 뜨겁다)
  for B = 4 -> ... -> 128
    for L = 1 -> 2 -> 3 -> 4          (뒤로 갈수록 뜨겁다)
```

`L=1` 은 차가울 때, `L=4` 는 뜨거울 때 측정된다. 회귀는 이 시간 드리프트를
**레이어 효과로 오인**한다. `inner` 를 키우면 `L=4` 가 더 길고 더 늦게 돌아
교란이 커진다 — 위 표가 그것이다.

##### 두 번째 구조적 결함 — `rest` 비례 배분

`normUs + otherUs` 를 att/ff 에 비례 배분했는데, 그 안에는

- 레이어 비례 항 (residual, cast, merge)
- **레이어 독립 항** (embedding, final norm, lm_head)

이 섞여 있다. 레이어 독립 항까지 배분하면 `ta/tf` 에 `c0` 가 다시 섞이고,
그 `c0` 를 회귀로 빼려 하니 스스로를 방해한다.

##### 해법 — 한 forward 에서 레이어별로 직접 읽는다

`NnExecutor::getLastForwardStepTimes()` 를 추가해 step 별 시간을 그대로 꺼낸다.
`opConfig->index` 가 `layerIndex` 이므로 레이어별로 모을 수 있고,
`block_` 접두가 없는 op(embedding, final, lm_head)는 자연히 `c0` 로 분리된다.

```text
제거된 것            이유
──────────────────────────────────────────────────────────────
레이어 수 회귀        한 forward 에서 모든 레이어를 같은 열·메모리 상태로 수집
rest 비례 배분        op 이름으로 att/ff/c0 를 직접 분류
c0 추정              layerIndex 로 자연히 분리
R^2 판정             적합을 안 하므로 불필요
```

##### 정합성 검사가 내장된다

step 시간의 합은 forward 벽시계와 같아야 한다. 실측:

| prefix | 128 | 512 | 2,048 | 4,096 | 8,192 |
|---|---|---|---|---|---|
| stepsum / wall | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 |

**이 검사가 회귀에는 없었다.** 그래서 1.87배 과소한 값을 그대로 믿었다.

```text
prefix=128, B=32, L=4
  레이어별 직접:  forward 0.534 s   (레이어당 ~169 ms + c0 48 ms)
  회귀 추정:      forward 0.286 s   (레이어당 66 ms + c0 22 ms)   <- 1.87배 과소
```

회귀는 열 드리프트를 기울기에 흡수시켜 **레이어 비용을 절반 이하로** 잡고 있었다.

##### 남은 검증 항목

- `L` 순서 정/역방향 교차 — 레이어 간 편차(att 49/58/68/49 ms)가 실제인지 변동인지
- **`P ≪ Q` 조건** — 현재 캘리브레이터는 `P = Q` 로 고정한다. production 의 초기
  청크는 `prefix` 가 작고 `seqLen` 은 크다. §7.5 의 `Q` 실험은 `attnMs` 에
  영향이 없다고 했지만 그것은 `S=448` 단일 조건이었다.
- 45점 수집 순서 무작위화 + 주기적 anchor 재측정(`B=32, prefix=2048`, ±5 % 이탈 시 무효)

### 7.7 링크 캘리브레이션 — 클러스터가 이종 링크였다

`prefill_bench/bench_link.cpp`(전용 echo 서버/클라이언트)로 64 B~4 MB 범위에서
왕복을 재고 `D(x)=α+x/β` 를 적합했다. `iperf` 같은 대용량 스트리밍은 `α` 를 못 잡고,
ssh 왕복은 프로토콜 처리 시간이 섞여 무효다.

| 경로 | α (ms) | β (MB/s) | R² |
|---|---|---|---|
| 16 GB 군 내부 (105/113/166/191) | 0.083~0.085 | **115.7~117.0** | 1.0000 |
| 8 GB 군 내부 (103/94/104/54) | 0.084~0.087 | **115.5~116.9** | 1.0000 |
| **군 간 (양방향 5개 경로 전부)** | 0.126 | **11.7** | 1.0000 |

측정에 사용한 크기 점들에서는 affine fit의 `R²`가 1.0000이었다. 그러나 이는 같은
데이터에 대한 적합도이며 실제 pipeline protocol의 정확성을 뜻하지 않는다. 논문에는
별도의 held-out 메시지 크기 MAPE와 최대 잔차도 함께 보고한다.

#### RAM 이 아니라 토폴로지다

모든 노드가 `1000Mb/full` 로 협상돼 있고 8 GB 노드끼리도 115.6 MB/s 가 나온다.
메인 스위치와 두 하위 스위치를 잇는 **업링크가 100 Mb** 다. RAM 용량과의 상관은
16 GB 4대와 8 GB 노드를 각각 다른 스위치에 꽂은 배치의 부산물이지 인과가 아니다.

#### 영향은 작지만 모델에는 필수다

활성값이 q80(1.06 B/원소)이라 경계 전송이 작다.

```
B=48, d=4096 → 209 kB
  1 GbE 경계:  0.083 + 209/115.7 = 1.9 ms
  100 Mb 경계: 0.126 + 209/11.7  = 18.0 ms
현재 스테이지 순서는 업링크를 정확히 1회 횡단 → M=151 × 16.1 ms = 2.4 s (전체의 1.3 %)
```

여기서 계산한 추가 비용은 약 2.4 s인데 기존 계측의 `syncXferMs`는 69.6 ms였다.
두 값은 **그대로는 일치하지 않는다.** `syncXferMs`가 wire time 전체를 재는지,
비동기 send queue에서 실제 전송시간이 빠지는지부터 확인해야 한다. 따라서 현재 말할 수
있는 것은 “기본 순서에서 모델상 약 1.3%”까지이며, 실제 기여는 event trace 전에는
확정하지 않는다. 그래도 단일 `β`를 쓰면 군 간 경계 비용을 10배 잘못 입력하므로
pairwise link model은 필요하다.

#### placement를 모델 변수로 올린다

§2.1에서 “링크 비대칭이 실측에서 클 때만 후속 확장”이라 유보했다. 대역폭 비가
10배이므로 비용 모델에서 순서를 무시할 수 없다.
현재 순서 `113,166,191,103,94,104,54` 는 업링크를 1회만 건너는데 이는 우연이고,
순서를 섞으면 최대 7회까지 건널 수 있다.

다만 현재 순서에서 예측 영향은 1.3%라 placement speedup이 크다고 미리 주장하지 않는다.
연산자별 노드 배치와 링크 횡단을 **함께** 최적화했을 때의 end-to-end 이득으로 판정한다.

**부수 효과**: §8.5 네트워크 축 실험을 `tc` 없이 **자연 조건**으로 할 수 있다.
스테이지 순서만 바꾸면 횡단 횟수를 0~7 로 통제할 수 있다. §7.2d 의 연산자 순위
역전과 같은 구조 — 인위적으로 만들려던 조건이 이미 존재한다.

### 7.8 기존 결과를 어떻게 사용할 것인가

기존 distributed-llama 대비 최대 42×는 전체 시스템 정상화의 결과이며 DerivePP
단독 이득이 아니다. 논문에서는 다음 계층으로 원인을 분리한다.

| 단계 | 구성 | 보고 목적 |
|---:|---|---|
| 1 | Original distributed-llama | 역사적 출발점 |
| 2 | Sanitized baseline | 컴파일·repack·버그 수정까지 반영한 공정한 기준선 |
| 3 | Fixed PP + 기존 heuristic | 기존 scheduling policy |
| 4 | Analytical `N,B` | 노드 수와 청크 선택 기여 |
| 5 | + analytical partition `p` | att/ff 경계 선택 기여 |
| 6 | + placement `π` | 노드 subset/order와 링크 인지 기여 |
| 7 | DerivePP 전체 | `(N,B,π,p)` 공동 계획의 end-to-end 결과 |
| 8 | Measured best-of-grid | 사전 고정한 측정 grid의 실측 최선 |

42×는 별도의 system journey로 보고할 수 있지만, 알고리즘 speedup은 반드시 2번 또는
3번을 분모로 사용한다.

---

### 7.9 J1b — partition-only A/B (결정적 negative)

**설계.** 이전 J1 은 plan 순서와 세션 드리프트가 완전히 교락돼 폐기했다(그리고
root 에서 다른 벤치를 동시에 돌려 오염시켰다). J1b 는 매 plan 사이에 `P0` 를 끼운
anchor 반복 + 정순/역순 설계이며, `flock` 으로 배타 실행을 강제한다. 각 plan 의
기준선은 **인접 anchor 의 선형보간**으로 잡아 드리프트를 상쇄한다.

조건: llama3-8B q40, N=8, B=32, S=1789, 56 chunks, 17 회 실행.

| plan | 정순 | 역순 | 중앙값 | 설명 |
|---|---|---|---|---|
| P1 | 0.929 | 1.003 | 0.966 | 속도 비례 |
| P2 | 0.967 | 1.030 | **0.999** | DerivePP DP |
| P3a | 1.039 | 1.006 | 1.022 | 인접 조정 |
| P3b | 0.954 | 1.006 | 0.980 | |

anchor `P0` 9회: CV **0.6%**, 드리프트 range 2.4%.

**결론.** 어떤 분할도 균등을 유의미하게 이기지 못한다. DP가 고른 `P2` 의 이득은
**정확히 0**(0.999)이다. anchor CV 가 0.6% 로 3% 차이도 분해 가능한 해상도이므로
"측정이 나빠 못 잡았다"는 설명은 성립하지 않는다.

**구조적 이유.** 8 GB/16 GB 군이 **같은 CPU** 라 노드 성능 이종성이 작고, 링크
이종성은 100 Mb uplink 구간에만 있다. 64 sub-block → 8 stage 문제에서 균등이 이미
거의 최적이며, DP 가 찾는 재배치량이 비용 모델의 예측 오차보다 작다.

**잠정 관찰(n=2, 주장 아님).** 비균등 plan 의 회차 간 변동이 균등보다 크다
(P1 7.6%, P2 7.7%, P3b 5.8% vs P0 9회 2.4%). 불균형 stage 가 일시적 교란에 더
민감하다는 해석이 가능하나 표본이 부족하다.

**논문 영향.** "분할을 최적화한다"는 축은 기여로 쓸 수 없다. 남는 것은 `(N, B)`
축이며 거기서는 실측 효과가 크다(`B=128` vs `B=16` 이 1.38×). §13 기여 문장과
§10 개발 계획을 축-판별 중심으로 다시 써야 한다.

---

### 7.10 J2 — B 축 A/B (결정적 positive)

J1b 와 **동일한 anchor 설계**(정순+역순 17회, `flock` 배타 실행). 분할은 균등으로
고정했다 — J1b 가 분할 무관을 보였으므로 그것이 올바른 통제다.
조건: llama3-8B q40, N=8, S=1789, 균등 분할. anchor 는 **distributed-llama 기본값
`nBatches=32`**(`src/app.cpp:418`)이므로 primary baseline 과 일치한다.

| B | chunks | 정순 | 역순 | 중앙값 |
|---|---|---|---|---|
| 16 | 112 | 1.151 | 1.120 | **1.136** |
| 24 | 75 | 0.958 | 0.940 | 0.949 |
| 32 | 56 | — | — | 1.000 (기본값) |
| 64 | 28 | 0.946 | 0.878 | 0.912 |
| 128 | 14 | 0.771 | 0.758 | 0.765 |

anchor `B32` 9회: CV **1.35%**, range 4.0%.
기본값 대비 최선 **1.136×**, 최선/최악 **1.485×**.

**두 축의 대비가 핵심이다.**

| | 효과 크기 | anchor 노이즈 | 정순/역순 |
|---|---|---|---|
| partition (§7.9) | 0.97~1.02 | 0.6% | 불일치 |
| `B` (§7.10) | 0.765~1.136 | 1.35% | 일치 |

`B` 축은 효과가 노이즈의 10~25배이며 방향 간 재현된다. 같은 방법론·같은 해상도로
잰 두 축이 정반대 결론을 주므로, §7.9 의 평평함은 **측정 실패가 아니다**.

**비단조성.** `B=24` notch 가 양방향에서 재현된다(0.958 / 0.940). `16→24` 에서
나빠졌다가 `32` 에서 회복하므로 단조도 볼록도 아니다. 닫힌 형태로 `B*` 를 유도할 수
없고 **캘리브레이션 테이블이 필요하다**는 직접 증거이며, §5 의 `B_ε`·`B₀` 기각
(§12.2)과 일관된다.

**주의.** 1.38× 는 과거 관측이었고 여기서 최선/최악 1.485× 로 갱신됐지만, `B=128` 은
아무도 고르지 않는 값이므로 **primary 수치로 쓰지 않는다**. 논문에서는 세 비교를
분리한다 — 기본 설정 대비 **1.136×**(primary), 고정 `B=32` 대비(component ablation),
과거 선형 heuristic 의 `B=128`(잘못된 규칙의 반증).

---

### 7.11 AxisCert 의 축별 차분 오차 `δ_x`

절대 예측 구간으로 게이트를 정의하면 무력화된다 — `S=7212` 절대 오차가 30% 이므로
`potential = U(q0) − L(q)` 는 모든 축에서 크게 나온다. 결정에 필요한 것은 절대
정확도가 아니라 **계획 간 비교의 교정 정확도**다.

    e_x(q; q0) = log[T_meas(q)/T_meas(q0)] − log[T_pred(q)/T_pred(q0)]
    δ_x = max_q |e_x|

캘리브레이션 `artifacts/cal_perlayer_0819`, 예측은 `prefill_bench/eval_plans.py`
(DP 탐색 없이 주어진 계획만 평가).

**분할축 (J1b, S=1789 N=8 B=32)**

| q | 예측 R̂ | 실측 R | e |
|---|---|---|---|
| P1 | 0.9401 | 0.9660 | +0.0272 |
| P2 | 1.0302 | 0.9990 | −0.0307 |
| P3a | 1.0843 | 1.0220 | −0.0592 |
| P3b | 1.0001 | 0.9800 | −0.0203 |

`δ_p = 0.059` (6.1%)

**B 축 (J2, S=1789 N=8 균등분할)**

| q | 예측 R̂ | 실측 R | e |
|---|---|---|---|
| 16 | 1.0944 | 1.1360 | +0.0373 |
| 24 | 0.9300 | 0.9490 | +0.0202 |
| 64 | 1.0143 | 0.9120 | −0.1063 |
| 128 | 0.8420 | 0.7650 | −0.0958 |

`δ_B = 0.106` (11.2%)

**`δ_B > δ_p` 가 실증됐다.** `B` 가 바뀌면 chunk 수와 executor step tax 가 함께
바뀌어 공통 모드가 완전히 상쇄되지 않는다. 축별 `δ` 가 필요하다.

#### 게이트 적용 결과 — 분할축에서 거짓 양성

| 축 | `max_q log R̂` | `δ_x` | 낙관적 상한 | 판정 | 실제 |
|---|---|---|---|---|---|
| B | +9.4% | 11.2% | +21.7% | active | **active (+13.6%)** ✓ |
| p | +8.4% | 5.9% | +15.0% | active | **inactive (1.000)** ✗ |

**AxisCert 는 현재 모델의 차분 정확도로 분할축을 해석적으로 잘라내지 못한다.**
비용 모델이 분할 이득을 과대평가(+8.4% vs 실제 +2.2%)하고 `δ_p` 가 그것을 덮을 만큼
작지 않기 때문이다. 실제로 그 축은 J1b 라는 **end-to-end A/B 로** 껐다. 이는
"sweep 없이 축을 자른다"는 주장의 한계로 정직하게 보고해야 한다.

완화 요소: 거짓 양성은 거짓 음성보다 훨씬 덜 해롭다. 축 활성화는 "탐색한다"는
뜻이지 틀린 계획을 고른다는 뜻이 아니다. 관측된 범위에서 **거짓 음성은 0** 이며,
게이트는 안전 방향으로 보수적이다.

#### 모델 편향의 방향이 축마다 반대다

| 축 | 예측 스프레드 | 실측 스프레드 | 기울기 |
|---|---|---|---|
| 분할 | 1.153× | 1.058× | 0.39 (민감도 **과대**) |
| B | 1.300× | 1.485× | 1.51 (**과소**) |

단일 `δ` 로는 어느 쪽도 맞출 수 없다.

#### 순위는 맞는다 — regret 0

| 축 | 예측 argmin | 실측 최선 | regret |
|---|---|---|---|
| B | B=16 | B=16 | 0 |
| 분할 | P3a | P3a | 0 |

모델은 **순위는 맞히고 크기를 틀린다**. Phase 1 통과 기준("table-selected B 가
measured best plateau 안")은 충족이다.

⚠️ **정정.** §7.9 의 "DP 이득 0" 은 `P2` 기준이었는데 `P2` 는 **옛 캘리브레이션**의
산출물이다. 새 캘리브레이션에서 5개 후보 중 예측 최소는 `P3a`(실측 1.022)다.
분할축의 실제 이득은 0 이 아니라 **약 2%** 이며, 다만 재현성이 약하다(1.039/1.006)
— anchor CV 0.6% 대비 역순 표본이 노이즈 경계에 있다.

---

### 7.12 `S=7212` 는 측정이 불안정하다 — 긴 프롬프트 주장의 선행 블로커

**⚠️ 이 절은 한 번 잘못 결론냈다가 정정한 것이다.** 처음 3회만 보고 "과거 36% 변동은
콜드 스타트였다" 고 적었으나, 9회를 모으니 반증됐다.

동일 조건 `P0` (llama3-8B, N=8, B=32, 균등분할, 226 chunks) 9회:

    190.8  206.1  206.4  198.9  203.3  203.1  200.8  249.4  180.8   (초)

CV **8.7%**, range **38.0%**.

함정은 **안정 구간이 실재한다**는 것이다. 연속 3회만 보면 CV 0.56% 가 나온다
(203.3 / 203.1 / 200.8). 그 구간을 보고 일반 결론을 내리면 틀린다 — 실제로 그렇게
틀렸다. 간헐적이며 콜드 스타트와 무관하게 세션 중간에도 발생한다.

**J4(`S=7212` partition A/B)는 이 때문에 판정 불가다.**

| plan | 정순 | 역순 | 차이 |
|---|---|---|---|
| Pspeed | 341.9 s | 241.7 s | +41% |
| Pdp | 251.8 s | 198.9 s | +27% |

같은 plan 이 회차에 따라 27~41% 다르고 anchor 자체가 38% 흔들리므로, 정순에서
관측된 `Pdp` 0.807 / `Pspeed` 0.591 을 분할 효과로 귀속할 수 없다.
**`δ_p(S=7212) = 61%` 도 무효다** — 예측이 아니라 측정이 흔들린 것이다.

**원인 후보.** 온도는 아니다 — 역순(57~62 °C)과 정순(56~61 °C)이 거의 같은데 결과가
38% 다르다. 남은 후보:

- 네트워크 산발 스톨 — J4 정순 `Pspeed` 에서 실제로 `NET_STALL 58 s → NET_TIMEOUT`
  후 drain 실패가 관측됐다. 그 실패 모드의 약한 버전(타임아웃엔 못 미치는 스톨)이
  38% 변동을 설명할 수 있다. **가장 유력하다.**
- 메모리 단편화/페이지 캐시 — `S=7212` 는 KV 가 크고 반복 실행 시 상태가 누적된다
- 워커 일부의 산발적 저하 — 노드별 stage 시간을 찍어야 확인 가능

**영향.** 이 불안정성은 긴 프롬프트에 관한 **모든** 주장의 선행 조건이다. 해결 전에는
분할축의 길이 의존성도, 모델/길이 축 일반화도 검증할 수 없다. 다음 단계는 A/B 가
아니라 **진단** — `P0` 반복 중 노드별 stage 시간과 네트워크 스톨 로그를 함께 수집한다.

**유효성 범위.** §7.9(분할축 평평)와 §7.10(B 축 1.136×)은 `S=1789` 에서 각 17회,
anchor CV 0.6~1.35%, 정순/역순 일치로 얻었다. 그 길이에서는 유효하며, 여기 불안정성은
`S=7212` 에 국한된 관측이다.

### 7.12b `S=7212` 불안정성 진단 — drain 과 크래시 후유증

§7.12 의 후속. A/B 가 아니라 **원인 규명**이 목적. 동일 조건 `P0` 6회를 돌리며
네트워크 스톨·메모리·온도를 함께 기록했다(`diag7212.sh`).

| rep | prefillMs | drain | prefill−drain | 스톨수 | 최대스톨 | free MB |
|---|---|---|---|---|---|---|
| 1 | 204,892 | 25,466 | 179,426 | 18 | 24.0 s | 7398 |
| 2 | 218,079 | 35,022 | 183,057 | 20 | 34.0 s | 7391 |
| 3 | 212,705 | 40,212 | 172,493 | 23 | 40.0 s | 7331 |
| 4 | 200,300 | 22,989 | 177,311 | 17 | 22.0 s | 7335 |
| 5 | 205,111 | 39,901 | 165,209 | 21 | 38.0 s | 6993 |
| 6 | 196,633 | 21,838 | 174,795 | 16 | 20.0 s | 7004 |

| 지표 | CV | range |
|---|---|---|
| prefillMs | 3.50% | 10.9% |
| drain | **25.0%** | **84.1%** |
| prefill−drain | 3.22% | 10.8% |

#### 스톨의 정체는 네트워크 장애가 아니라 drain 이다

`NET_STALL` 은 2초마다 같은 하나의 대기를 반복 보고한 것이다(`stalled=2000 →
40012ms`, `pending=48B`). 해소 직후 로그가 `[DRAIN] chunk=0 t=40212.5ms` 이므로,
root 가 226 청크를 모두 밀어넣은 뒤 파이프라인이 비워지길 기다리는 시간이다.

메모리(free ~7 GB 일정)와 온도(56~61 °C 일정)는 변동과 무관하다.

**drain 을 빼도 안정되지 않는다.** drain 과 파이프라인 시간이 음의 상관이기
때문이다 — root 가 앞서 달리면 큐가 쌓여 drain 이 커지고 root 시간은 줄어든다.
총합이 각각보다 보존적이다.

#### 38% 변동의 실제 원인은 크래시 후유증이다

깨끗한 세션에서는 `S=7212` 도 **CV 3.5%** 로 측정된다. §7.12 의 range 38% 는 길이의
성질이 아니었다. J4 역순 회차는 직전 실행이 `NET_TIMEOUT` 으로 크래시한 **뒤에**
시작됐고, 그 세션의 anchor 만 249.4 / 180.8 로 튀었다. 정순 anchor 는
203.3 / 203.1 / 200.8 (CV 0.56%) 로 멀쩡했다.

→ 오염원은 프롬프트 길이가 아니라 **실패한 실행 다음 세션**이다. 통제 가능하다.

#### 파이프라인이 실제로 불균형이다

균형 잡힌 파이프라인이면 drain ≈ `(N−1)·C` = 7 × 0.86 s ≈ **6 s** 여야 한다.
관측값은 22~40 s 로 **4~7배**다. root 가 앞서 달려 큐가 쌓인다는 뜻이다.

두 가지 함의가 있다.

1. `S=7212` 에서 분할이 실제로 중요할 수 있다. J4 정순의 방향(`Pdp` 0.807,
   `Pspeed` 0.591)과 일치한다. 단 재현이 필요하다(§7.12).
2. **§12.4 의 balanced-capacity 전제가 이 길이에서는 성립하지 않는다.** 그 절의
   적용 조건 서술을 이 데이터로 보강해야 한다 — 보조정리는 `S=1789` 처럼 균형이
   확인된 경우에만 CP 를 기각한다.

---

### 7.12c J4-R — `S=7212` 에서도 분할축 비활성 (사전등록 판정)

§7.12 의 재실행. 이전 J4 의 교락(NET_TIMEOUT 크래시 후 세션 계속 사용)을 제거하고,
실패 시 세션 전체를 `invalid_session` 으로 분리하는 중단 규칙을 넣었다.
보관 9회(`P0×5, Pdp×2, Pmid×2`), warm-up 1회 폐기.

보정은 **기하평균 paired ratio** — 세션 드리프트가 곱셈적이고 문서의 residual 정의도
로그 공간이므로 선형 보간보다 일관된다.

    R(q) = sqrt(T_P0,전 · T_P0,후) / T_q

| plan | 예측 R̂ | 회차별 R | 기하평균 | 판정 |
|---|---|---|---|---|
| Pdp `9,7,8,…` | 1.058 | 1.017, 0.955 | **0.986** | 비활성 |
| Pmid `10,6,8,…` | 1.021 | 1.038, 0.964 | **1.000** | 비활성 |

anchor `P0` 5회: 184.5 / 182.0 / 186.9 / 191.9 / 195.8 s. CV 2.65%, **단조 증가 +6%**.
인접 anchor 보정이 이를 흡수하며, 잔여 회차별 폭(0.955~1.038)은 사전등록한 비활성
구간 `0.95~1.05` 안이다.

**결론: 분할축은 `S=1789` 와 `S=7212` 양쪽에서 비활성이다.** 역전점 `S*≈3,354` 의
양편에서 확인됐으므로 att share 0.22 와 0.68 구간을 모두 덮는다.

#### 두 개의 철회

1. **"긴 프롬프트에서 분할축 활성"** — 철회. 이전 J4 의 `Pdp` 0.807 / `Pspeed` 0.591 은
   정순부터 오염된 세션이었다.
2. **`δ_p(S=7212) = 61%`** — 철회. 깨끗한 데이터로는:

| S | δ_p |
|---|---|
| 1789 | 5.9% |
| 7212 | **7.1%** |

**차분 정확도는 regime 의존적이지 않다.** 절대 오차는 길이에 따라 무너지는데
(비율 0.912 → 0.573, §7.14) 차분 오차는 6~7% 로 유지된다. §7.11 의 핵심 주장을
**강화**한다 — 축을 자르는 데 필요한 것은 절대 예측이 아니라 쌍 비교의 교정 정확도다.

#### 게이트는 두 길이 모두에서 거짓 양성

| S | `max_q log R̂` | `δ_p` | 낙관적 상한 | 판정 | 실제 |
|---|---|---|---|---|---|
| 1789 | +8.4% | 5.9% | +15.0% | active | 비활성 |
| 7212 | +5.8% | 7.1% | +13.6% | active | 비활성 |

두 길이에서 **같은 방향**으로 틀린다(`e` 가 모든 계획에서 음수 — 모델이 분할 이득을
일관되게 과대평가). 무작위 오차가 아니라 **체계적 편향**이므로 보정 가능성이 있다.
다만 현재 `δ_p`, `δ_B` 는 같은 후보군을 설명하는 데 쓴 **in-sample 값이므로
certificate 가 아니다.** AxisCert 구현 전에 다음 계약이 선행되어야 한다.

    offline : calibration + 독립 paired validation → 축별 δ_x 와 certificate
    online  : certificate 로 축을 fixed / enumerate / blocked 결정
    평가    : certificate 생성에 쓰지 않은 held-out 조건에서
             false-negative, false-positive, regret 측정

#### 남은 관측 — drain 단조 증가

anchor 의 drain 이 세션에 걸쳐 17.9 → 29.2 s 로 증가한다(`Pdp` 역순 회차에서는 41.4 s).
같은 분할·같은 프롬프트이므로 tail 청크 비용으로는 설명되지 않는다. 판정에는 영향이
없었으나(paired 보정이 흡수) **원인 미규명**이다. `drain_lifecycle.sh` 로
"이어서 실행" vs "완전 재시작" 을 비교해 세션 상태 누적인지 자원 경합인지 가른다.
그 전까지 `S=7212` 수치는 논문 핵심 그래프에서 제외한다.

---

### 7.13 플래너 확장성 — 원인은 `keep` 과 격자 탐색의 곱

`S=7212, N=8` 을 기본 인터페이스(`--keep 4000`, `B` 9개 격자)로 돌리면 **RSS 8.95 GB**
에서 OOM 직전에 도달해 중단해야 했다(16 GB 노드, 9분 경과).

단일 조합만 풀면 상황이 다르다.

| 조건 | 시간 | peak RSS |
|---|---|---|
| `S=7212, N=8, B=32, keep=500` | 64 s | **0.60 GB** |
| `S=7212, N=8, keep=4000, B 9개` | >9분 | 8.95 GB (중단) |

원인은 완료벡터 차원(`M`: 56 → 226, 4배)이 frontier 상한 `keep` 및 `B` 격자 크기와
곱해지는 것이다. 따라서 "planner 가 긴 프롬프트에서 확장되지 않는다"는 진술은
**과하다** — `keep` 을 낮추면 0.60 GB / 64 s 로 풀린다. 다만 최적성 보장은 잃는다.

**정확한 진술은 이렇다.** 실제 사용 시나리오에서는 `B` 를 모르는 상태로 시작하므로
`(N, B)` 격자 전체를 돌아야 하고, 그때 비용은 조합 수에 비례한다. 축을 미리 자르면:

| | 결과 |
|---|---|
| 분할축 활성 + B 격자 | `B` 조합마다 DP. `keep` 을 키우면 8.95 GB 에서 실패 |
| 분할축 비활성 | 균등 분할 고정 + `B` 테이블 조회. DP 0회 |

즉 §7.11 의 거짓 양성은 이론적 흠이 아니라 **탐색 비용을 실제로 발생시킨다**.
게이트가 닫히면 DP 를 한 번도 돌리지 않는다.

부수적으로, 닫힌 형태 `F_j = C^Σ_j + max_{i≤j}(a_i − C^Σ_i)` (§3)를 이미 갖고 있으므로
완료벡터 전체 대신 `(C^Σ, prefix-max)` 요약만 전파하는 압축이 가능해 보인다.
미검증이며 우선순위는 낮다.

**교훈 (측정 위생).** `keep=500`/`1500` 재시도가 빈 출력을 냈다. 파일 리다이렉트 시
Python stdout 이 블록 버퍼링이라, 중간에 죽으면 계산 결과가 통째로 사라진다.
장시간 계산은 반드시 `python3 -u` 또는 `flush=True` 로 돌린다.

---

### 7.14 `S=7212` 예측 — 절대 오차가 길이에 따라 급격히 나빠진다

`artifacts/cal_perlayer_0819` 기준, `S=7212, N=8, B=32`:

| plan | 예측(s) | R̂ (vs P0) |
|---|---|---|
| P0 (균등) | 118.13 | 1.000 |
| Pdp (DP, keep=500) | 111.69 | 1.058 |
| P3a | 113.42 | 1.042 |
| Pspeed | 123.79 | 0.954 |

**DP 가 고른 분할은 `9,7,8,8,8,8,8,8`** — 균등에서 서브블록 **1개만** 옮긴다.
모델 스스로 긴 프롬프트에서는 균등이 거의 최적이라고 말하는 셈이다.

**절대 오차.**

| S | 예측 P0 | 실측 P0 | 비율 |
|---|---|---|---|
| 1789 | 23.16 s | 25.38 s | 0.912 |
| 7212 | 118.13 s | 206.2 s | **0.573** |

길이가 4배가 되면 예측 비율이 0.912 → 0.573 으로 무너진다. attention 비용의
prefix 의존 성장을 모델이 과소평가한다는 뜻이며, §7.7 의 미해결 잔차와 같은 방향이다.
**이것이 절대 구간 기반 게이트를 쓸 수 없는 결정적 이유다** — 그러나 §7.11 이 보였듯
차분 정확도는 훨씬 낫다.

AxisCert 게이트 적용: `max_q log R̂ = +5.8%`, `δ_p(S=1789) = 5.9%` → 낙관적 상한
**+12.2%** → "active".

⚠️ **이 길이에서는 검증할 수 없다.** J4 가 실측을 주려 했으나 §7.12 의 측정 불안정
(anchor range 38%)으로 판정 불가였다. `δ_p(S=7212)` 는 추정 불가이며, 위 게이트 판정의
정오도 확인되지 않았다. `S=7212` 진단이 선행되어야 한다.

---

### 7.15 노드 이종성 원인 진단 — 순위 역전 근거 철회

§7.2(d)의 `τ = −0.29` 를 근거로 "스칼라 노드 속도 모델은 반증됐다" 고 주장해 왔다.
그 주장을 **철회한다.**

#### (a) 재현 실패

`artifacts/cal_perlayer_0819` — 9노드를 **동시에** 수집(같은 시각·같은 열 조건),
layer 0 인공물 제외, 공통 `(B, prefix)` 90개 조건.

    Kendall τ (GEMM 순위 vs attention 순위) = +0.111   (일치 20, 불일치 16)
    이전 기록                                = −0.29    (일치 10/28)

부호가 뒤집혔다. 이전 값은 노드별로 다른 시각에 수집된 프로파일이었을 가능성이 크다.

#### (b) DVFS·열 — 단일 전역 설명이 아니다

104(Rev 1.0/8 GB), 113(Rev 1.1/16 GB), 166(Rev 1.1/16 GB) 에서
`ondemand → performance → ondemand` 로 정책을 바꿔가며 각 5회.
주파수는 스냅샷이 아니라 `cpufreq/stats/time_in_state` 차분의 **가중평균**으로 측정.

| 정책 | 실효 평균 주파수 |
|---|---|
| ondemand | 2397 ~ 2400 MHz |
| performance | 2400 MHz |

**`ondemand` 에서도 부하 중에는 이미 최대 주파수였다.** throttling 없음(68~70 °C).
그리고 세 정책 구간 모두에서 104 = GEMM 최속 / attention 최저가 유지됐다.

따라서 정확한 진술은 "DVFS/열이 배제됐다" 가 아니라
**"ondemand 와 performance 양쪽에서 편차가 유지돼, DVFS/열은 관측된 편차의 단일한
전역 설명으로 보이지 않는다"** 이다.

부수 관측: 복원성 검사에서 GEMM 은 ±1.6% 로 안정한 반면 attention 은 최대 6.7%
표류했다. attention 이 환경 상태에 더 민감하다.

#### (c) 메모리 경로 — 상관 없음

`prefill_bench/membench.c` (STREAM triad 대역폭 + pointer-chase 지연), 9노드 순차 측정.

| 지표 | vs GEMM | vs attention |
|---|---|---|
| STREAM 1스레드 | +0.23 | +0.02 |
| STREAM 4스레드 | +0.38 | −0.00 |
| CHASE 2 MB | +0.24 | −0.19 |
| CHASE 128 MB | −0.03 | +0.30 |

**뚜렷한 상관이 없다**(|r| ≤ 0.38). 3노드 부분집합에서는 `CHASE 128 MB` vs attention
이 −0.98 이었으나 9노드에서 +0.30 으로 부호까지 뒤집혔다 — **표본 수 효과였다.**

반례도 명확하다. `113` 은 attention 이 0.758 로 가장 느린데 DRAM 지연은 119.3 ns 로
가장 빠른 축이다. `166` 은 GEMM 최하위(0.876)인데 STREAM 은 상위권이다.

(`L3 2 MB` 지연은 94/96/166 이 60~61 ns, 나머지가 91~109 ns 로 뚜렷이 갈리는 군집이
있으나 연산자 성능과 대응하지 않는다. 미해석.)

#### (d) 수정된 결론

> 동일 Pi 5 노드들 사이에 GEMM 은 0.876~1.000 의 처리율 차이가 관측됐다
> (attention 의 0.758~1.000 중 하한은 §7.15(f)에서 캘리브레이션 인공물로 확정됐다). 그러나 9노드 재측정에서 두 연산자의 rank correlation 은
> `τ = +0.111` 로, 이전의 음의 상관은 재현되지 않았다. STREAM 대역폭과 pointer-chase
> 지연도 두 연산자 처리율과 뚜렷한 상관을 보이지 않았다. 따라서 현재 데이터는
> **노드 간 성능 편차는 확인하지만, 연산자별 이종성의 원인이나 placement-aware
> scheduling 의 필요성은 아직 뒷받침하지 못한다.**

⚠️ **철회는 반대 명제의 입증이 아니다.** `τ = +0.111` 은 스칼라 모델을 **증명하지
않는다.** rank-1 분리성 오차 1.12%(§7.x)를 근거로 쓰려면, 그것이 **같은 캘리브레이션
조건·같은 노드 집합·같은 `(B, prefix)`** 에서 나온 값인지 확인하고 held-out 에서도
유지되는지 봐야 한다. 현재는 둘 다 미확인이다.

#### (e) 남은 것

#### (f) `113` 의 attention 0.758 — 캘리브레이션 인공물로 확정

**조건별 분해가 먼저 하드웨어 가설을 약화시켰다.** `113` 의 느려짐은 균일하지 않다.

| prefix | 113 / 타노드 중앙값 |
|---|---|
| 128 | 1.091 |
| 512 | 1.399 |
| 2048 | 1.338 |
| 4096 | 1.210 |
| **8192** | **1.009** |

KV 트래픽이 가장 큰 `prefix=8192` 에서 정상이다. 메모리·캐시 특성이 원인이라면
정반대여야 한다. 같은 `prefix=2048` 안에서도 `B=32` 는 2.18배, `B=64` 는 0.99배로
인접 조건 간 편차가 크다 — 하드웨어 특성은 이렇게 튀지 않는다.

**clean 재측정 (B=32, prefix=2048 집중, 10회, control `96` 동시 측정):**

| rep | 113 att(ms) | 96 att(ms) | 비율 |
|---|---|---|---|
| 1 | 44.91 | 34.04 | 1.319 |
| 2 | 35.09 | 34.45 | 1.019 |
| 3 | 51.50 | 34.08 | 1.511 |
| 4 | 36.41 | 35.20 | 1.034 |
| 5 | 35.96 | 35.60 | 1.010 |
| 6 | 35.15 | 35.29 | 0.996 |
| 7 | 36.02 | 35.92 | 1.003 |
| 8 | 34.69 | 35.60 | 0.975 |
| 9 | 43.64 | 35.74 | 1.221 |
| 10 | 43.86 | 36.30 | 1.208 |

**중앙값 1.027** (범위 0.975~1.511). 10회 중 6회가 control 과 동일하다.
`96` 은 같은 창에서 ±3% 로 안정한 반면 `113` 은 48% 범위다 — 두 노드를 **동시에**
측정했으므로 세션 공통 요인이 아니라 `113` 고유의 산발적 사건이다.
**FFN 은 영향받지 않는다**(50.5~56.1 ms 일정). attention 만 튄다.

context switch 는 원인이 아니다 — 느린 회차가 423~451/s 로 오히려 **낮고**, 빠른
회차가 666~707/s 로 높다. 상관이 반대다.

**결론: `0.758` 은 하드웨어 이종성이 아니라 캘리브레이션 인공물이다.** 전체 그리드
45조건 중 일부가 간헐적 사건에 걸려 중앙값을 끌어내렸다.

이로써 실재하는 노드 편차는 다음으로 줄어든다.

    이전:  GEMM 0.876~1.000 (12%),  attention 0.758~1.000 (24%)
    수정:  GEMM 0.876~1.000 (12%),  attention 편차는 대부분 측정 잡음

**미규명 잔여**: `113` 에서만 attention 이 간헐적으로 1.2~1.5배 느려지는 사건의 정체.
온도·context switch·FFN 어느 것과도 대응하지 않는다. 다만 논문 주장에는 영향이 없다 —
이것은 "노드 이종성" 이 아니라 **특정 노드의 측정 신뢰도** 문제이며, 대응은 반복 수
증가 또는 중앙값 대신 하위 분위수 사용이다.
- **STREAM 4스레드 저하** — `streamcheck.c` 로 affinity 고정·분리 배열·barrier·바이트
  집계를 검증한 뒤에도 9.59 → 6.36 GB/s(−34%)가 남는다. 다만 **원인 미규명이므로 논문
  주장으로 올리지 않는다.** CoRePP 기각은 `261.4 → 270.2 ms`, `G = 0.967×` 라는 직접
  측정으로 이미 충분하다(§12.3).

---

### 7.16 llama.cpp RPC — 외부 baseline 측정

**왜 필요한가.** 지금까지 비교 대상이 *우리가 고친* distributed-llama 뿐이라
"원래 그 구현이 느렸던 것 아니냐" 는 반론에 무방비였다. llama.cpp RPC 는 널리
쓰이는 공개 CPU 분산 구현이므로 독립적 baseline 이 된다.

조건: Llama-3 8B **Q4_0 GGUF**(우리 q40 과 동일 양자화), 스레드 4, 165.x LAN,
`llama-bench -p <tokens> -n 0` (prompt processing 만, 생성 제외, 모델 로드 제외).
`-p` 를 우리 프롬프트 토큰 수에 맞춰 길이 불일치를 제거했다.

#### (a) 노드를 늘려도 전혀 빨라지지 않는다

| nodes | P=447 t/s | 대비 | P=1789 t/s | 대비 |
|---|---|---|---|---|
| 1 (로컬) | 17.49 ± 0.21 | 1.00× | 17.16 ± 0.02 | 1.00× |
| 2 | 7.23 ± 0.07 | 0.41× | 6.79 ± 0.00 | 0.40× |
| 4 | 7.22 ± 0.05 | 0.41× | 6.81 ± 0.00 | 0.40× |
| 8 | 6.77 ± 0.00 | 0.39× | 6.47 ± 0.00 | 0.38× |

**2 → 8 노드에서 완전히 평탄하다**(오히려 소폭 하락). 프롬프트를 4배 늘려도 비율이
거의 그대로다(0.41/0.41/0.39 → 0.40/0.40/0.38). 즉 이것은 **고정 오버헤드 문제가
아니라 순차 실행이라는 구조**다 — (b)의 소스 확인과 일치한다. 반복 편차는 ±0.00~0.02.

#### (b) 원인 — 소스에서 확인됨

`ggml/src/ggml-rpc/ggml-rpc.cpp` 의 backend 인터페이스:

    set_tensor_async  = NULL
    get_tensor_async  = NULL
    cpy_tensor_async  = NULL
    graph_plan_create = NULL

**비동기 전송 경로가 하나도 구현돼 있지 않다.** `ggml_backend_rpc_graph_compute` 는
그래프를 직렬화해 보내고 **블로킹 대기**한다. 여러 RPC 디바이스가 있어도 분할된
서브그래프가 **순차 실행**되며 통신이 계산과 중첩되지 않는다. 노드를 8배 늘려도
총 계산 시간이 그대로인 이유다.

이것이 본 연구의 마이크로배치 겹침 실행과의 **구조적 차이**이며, 측정과 소스
양쪽으로 뒷받침되므로 논문에서 주장 가능하다.

#### (c) 절대 성능 격차는 귀속에 실패했다 — 주장하지 않는다

RPC 가 로컬보다 2.4배 낮은 이유를 `extra_bufts` 미지원(ARM SDOT repack 커널을
못 씀)으로 설명하려 했으나 **숫자가 맞지 않는다.**

| 구성 | P=447 t/s |
|---|---|
| 로컬, repack 있음 | 17.49 |
| 로컬, `GGML_CPU_REPACK=OFF` | **2.62 ~ 3.42** |
| RPC 2노드 | 7.23 |

repack 부재가 원인이라면 RPC 가 2.6~3.4 근처여야 하는데 **오히려 2.1~2.8배 빠르다.**
가설 기각. 실제 원인은 미규명이므로 **논문에서 절대 격차는 주장하지 않고 확장성
부재만 보고한다.** 그것만으로 논지에는 충분하다.

#### (d) 부수 발견 — repack 이 6.7배다

`17.49 → 2.62` (repack off). CPU LLM prefill 에서 **단일 요인으로 가장 크다.**
본 연구도 `nn-repack` 과 `block_q4_0x4` ARM SDOT 경로를 쓰므로 llama.cpp 의
repack 있는 쪽과 같은 급이다. **이 사실을 논문에 명시해야 한다** — 명시하지 않으면
5.25× 가 repack 유무 차이로 오해될 수 있다.

#### (e) 공정성 각주

- `-ngl 99` 는 모든 레이어를 RPC 디바이스에 올린다. root 는 계산에 거의 참여하지
  않으므로 `N=2` 는 실질적으로 "원격 1노드 실행" 이다. 본 연구가 root 를 stage 0 으로
  쓰는 것과 구조가 다르다.
- RPC 는 개발사가 README 에서 **proof-of-concept, fragile, insecure** 로 규정한
  상태이며 실행 시 경고를 출력한다. 성숙도를 감안해 서술한다.

#### (f) 공식 distributed-llama — 스위치 안에서는 확장하고, 업링크에서 붕괴한다

⚠️ **이전 서술 정정.** "기존 distributed-llama 는 노드를 늘려도 나아지지 않는다" 고
적었으나 **부정확했다.** 깨끗한 조건에서 재측정하면 같은 스위치 안에서는 거의
선형으로 확장한다.

조건: 공식 upstream 바이너리(md5 `1125dfb9`, TTFT 계측만 추가), 같은 모델·양자화·
스레드 수·프롬프트. 지표는 upstream 이 출력하는 `wall_prefill_ms`.
회차: warm-up 1 + 본측정 2 (N=1 만 3회).

| S | N | t/s | llama.cpp 단일 대비 | 회차 편차 |
|---|---|---|---|---|
| 447 | 1 | 1.84 | 0.11× | 2.0% |
| 447 | 2 | 3.88 | 0.22× | 0.3% |
| 447 | 4 | **7.63** | **0.44×** | 1.0% |
| 447 | 8 | 2.08 | 0.12× | 0.1% |
| 1789 | 2 | 3.35 | 0.20× | 0.4% |
| 1789 | 4 | **7.06** | **0.41×** | 1.3% |
| 1789 | 8 | 2.08 | 0.12× | 0.1% |

**재현 확인.** 1차 측정(원자료 소실)과 v2 재측정(`artifacts/official_baseline_v2/`,
전체 로그·manifest·checksum 보존)의 차이는 `S=447` 에서 −1.7~+0.4%, `S=1789` 의
`N=8` 에서 −0.4% 다. `S=1789` 의 `N=2,4` 만 11~13% 낮아졌는데 v1 의 해당 조건 편차가
컸던 반면(최대 19%) v2 는 0.4~1.3% 이므로 **v2 를 채택한다.** 핵심 수치 `0.44×` 는
양쪽에서 동일하다.

**`N=4` 까지는 확장하고(N=1 대비 4.15×) `N=8` 에서 붕괴한다.** 두 길이 모두에서
재현되며 반복 편차는 0.2% 다(447: 214,990/215,373 ms · 1789: 865,379/863,343 ms).

**원인은 링크 이종성이다.** `N≤4` 는 16 GB 군(switch A)만 쓰고, `N=8` 은 8 GB 군
(switch B)을 포함해 **100 Mb 업링크**를 건넌다.

    군 내부  115.7 MB/s
    군 간     11.7 MB/s      10배 (§7.4 실측)

공식 구현은 파이프라인 단계마다 활성값을 동기 전송하므로, 느린 링크를 건너는 순간
그 구간이 병목이 된다. **§7.4 에서 측정한 링크 비대칭이 실제로 시스템을 무너뜨리는
것을 확인한 것**이며, 원래 서술("확장 안 됨")보다 강한 결과다.

#### (g) 두 공개 구현 모두 단일 노드를 넘지 못한다

| 시스템 | 최선 구성 | t/s | llama.cpp 로컬 단일 대비 |
|---|---|---|---|
| llama.cpp 로컬 | N=1 | 17.49 | 1.00× |
| llama.cpp RPC | N=2 | 7.23 | **0.41×** |
| 공식 distributed-llama | N=4 | 7.63 | **0.44×** |
| 공식 distributed-llama | N=8 | 2.08 | 0.12× |

**어떤 노드 구성으로도 단일 노드 llama.cpp 를 넘지 못한다.** 최선이 0.44× 다.
노드를 8대 붙여도 한 대만 못하다.

실패 방식은 서로 다르다 — RPC 는 **비동기 경로 부재**로 처음부터 평탄하고, 공식
dllama 는 **링크 이종성**에서 붕괴한다. 두 독립 구현이 각각 다른 이유로 같은 결론에
도달하므로, "commodity CPU cluster 에서 분산 prefill 이 실제로 확장되지 않는다" 는
문제 설정이 뒷받침된다.

⚠️ 측정 원자료는 재부팅으로 `/tmp` 가 초기화되며 소실됐다. 값은
`artifacts/external_baseline/raw/measured.tsv` 에 보존했으나 **로그는 없다.**
재현이 필요하면 재측정해야 한다.

### 7.17 세 시스템 확장 곡선 — 본 연구만 단일 노드를 넘는다

동일 하드웨어(4×16 GB + 4×8 GB Pi 5), 동일 모델(Llama-3 8B), 동일 양자화(Q4_0/q40),
동일 스레드 수(4), 동일 프롬프트 토큰 수, 데이터 경로 165.x LAN.
지표는 모두 **모델 로드 제외 · 생성 단계 미포함**의 prefill 시간.

본 연구는 균등 분할 + `B=32`(dllama 기본값) 고정 — **엔진 자체의 scale-out** 만 본다.
`B=16`(AxisCert 선택)의 추가 이득은 별도로 측정한다(그림 2).

#### S=1789 (llama.cpp 로컬 단일 = 17.16 t/s 기준)

| N | 본 연구 t/s | 자기 N=1 대비 | **llama.cpp 대비** | 공식 dllama | llama.cpp RPC |
|---|---|---|---|---|---|
| 1 | 11.66 | 1.00× | 0.68× | — | — |
| 2 | 22.89 | 1.96× | 1.33× | 2.97 | 6.79 |
| 4 | 44.49 | 3.82× | 2.59× | 6.11 | 6.81 |
| 8 | **77.44** | **6.64×** | **4.51×** | 2.07 | 6.47 |

#### S=447 (llama.cpp 로컬 단일 = 17.49 t/s 기준)

| N | 본 연구 t/s | 자기 N=1 대비 | **llama.cpp 대비** | 공식 dllama | llama.cpp RPC |
|---|---|---|---|---|---|
| 1 | 13.65 | 1.00× | 0.78× | 1.81 | — |
| 2 | 26.54 | 1.94× | 1.52× | 3.80 | 7.23 |
| 4 | 45.98 | 3.37× | 2.63× | 7.66 | 7.22 |
| 8 | **72.32** | **5.30×** | **4.14×** | 2.08 | 6.77 |

#### 세 가지 결론

**1. 본 연구만 단일 노드를 넘는다.**

| 시스템 | 최선 | llama.cpp 로컬 단일 대비 |
|---|---|---|
| 본 연구 (N=8) | 77.44 t/s | **4.51×** |
| 공식 distributed-llama (N=4) | 7.66 t/s | 0.44× |
| llama.cpp RPC (N=2) | 7.23 t/s | 0.41× |

**2. `N=8` 에서 붕괴하지 않는다.** 공식 dllama 가 무너진 바로 그 구성(8 GB 군을
포함해 100 Mb 업링크를 건넘)에서 본 연구는 `4.28× → 7.45×` 로 확장 폭이 오히려
커진다. 마이크로배치 겹침이 느린 링크의 전송을 계산과 중첩시킨다는 설계 의도가
실측으로 확인된다.

**3. 긴 프롬프트일수록 확장이 좋아진다.** 5.30×(S=447) → **6.64×**(S=1789, 효율 83%).
버블 `(N−1)/(M+N−1)` 이 `M` 과 함께 줄어드는 모델 예측과 방향이 일치한다.

#### ⚠️ 정정과 한계

**단일 노드 성능이 이전 기록보다 낮다.**

`N=1` 은 warm-up 1회로 부족했다(54.6 / 36.5 / 31.1 s, 편차 **76%** — 3회차까지 계속
빨라졌다). warm-up 을 **3회**로 늘려 재측정한 값을 위 표에 반영했다.

| | 이전 기록 | warm-up 1회 | **warm-up 3회 (채택)** |
|---|---|---|---|
| S=447 | 29,454 ms (llama.cpp 의 94%) | 36,457 ms (70%, 편차 76%) | **32,747 ms (78%, 편차 7%)** |
| S=1789 | — | 172,096 ms (61%, 편차 17%) | **153,424 ms (68%, 편차 5%)** |

재측정으로 편차가 76% → 7% 로 안정됐고 `자기 N=1 대비` 배수는 7.45× → **6.64×** 로
낮아졌다. 이전 기록의 94% 는 재현되지 않는다 — 본 연구 단일 노드는 llama.cpp 의
**68~78%** 다.

**헤드라인 `llama.cpp 대비 4.51×` 는 이 문제의 영향을 받지 않는다** — 분모가
llama.cpp 로컬 단일이기 때문이다. `N≥2` 의 편차는 2~8% 로 안정적이다.

원자료: `artifacts/external_baseline/raw/measured.tsv`.

---

### 7.18 그림 2 (B=16) — B 축 이득이 `S=1789` 에서 재현되지 않는다

§7.17(그림 1, `B=32` 고정)과 **동일 조건에서 `B` 만 16 으로** 바꿔 측정.
각 조건 warm-up 1 + 본측정 3회. OOM 없이 완주(여유 메모리 13 GB 유지).

| N | S=1789 B16/B32 | S=447 B16/B32 |
|---|---|---|
| 1 | 1.006 | 1.002 |
| 2 | 1.031 | 1.043 |
| 4 | 1.006 | 1.097 |
| 8 | **1.013** | **1.132** |

#### J2 와의 불일치 — anchor 재측정으로 해소 (v2)

그림 2 는 `S=1789, N=8` 에서 B16/B32 = **1.013×** 를 냈고 J2(§7.10)의 1.136× 와
어긋났다. anchor 설계로 재측정해 확정했다
(`artifacts/b_axis_v2/`, S=1789, N=8, 순서 `B32,B16,B32,B16,B32,B16,B32,B16,B32`,
warm-up 1회 폐기, 기하평균 paired ratio).

| 회차 | B16 | anchor 기하평균 | R |
|---|---|---|---|
| seq2 | 22.25 s | 25.75 s | 1.157 |
| seq4 | 22.33 s | 25.73 s | 1.152 |
| seq6 | 23.20 s | 26.01 s | 1.121 |
| seq8 | 22.54 s | 25.46 s | 1.130 |

**기하평균 1.140** (범위 1.121~1.157), anchor CV 2.00%.

| 측정 | 값 |
|---|---|
| J2 (anchor 17회) | 1.136 |
| **anchor 재측정 (9회)** | **1.140** |
| 그림 2 (독립 세션) | 1.013 ← **인공물** |

**그림 2 가 틀린 이유.**

| | B=32 중앙값 | B=16 중앙값 |
|---|---|---|
| anchor 세션 | 25.7 s | 22.5 s |
| 그림1/2 세션 | 23.1 s | 22.8 s |
| 차이 | **11%** | 1.3% |

`B=32` 측정이 세션에 따라 11% 흔들리고 `B=16` 은 안정적이다. 그림 2 는 `B=32`
곡선(그림 1)과 `B=16` 곡선을 **다른 세션에서** 재고 그 비율을 계산했으므로,
`B=32` 의 세션 드리프트가 그대로 비율에 들어갔다.

**교훈.** 두 설정의 비율을 주장하려면 반드시 **같은 세션 안에서 교차 배치**해야
한다. 곡선을 따로 재서 나누는 것은 세션 간 비교가 되어 드리프트를 흡수하지 못한다.
`S=447` 에서 그림 2 가 1.132× 로 J2 와 부합했던 것은 그 세션의 드리프트가 작았던
우연이다.

#### 확정: `B` 축은 두 길이 모두에서 활성

| S | B16/B32 | 근거 |
|---|---|---|
| 447 | 1.132× | 그림 2 (§7.18) |
| 1789 | **1.140×** | anchor 재측정 (본 절), J2 1.136× 와 일치 |

§0.6 의 "길이 의존 가능성" 단서는 **철회한다** — 두 길이 모두에서 1.13~1.14× 다.

#### 헤드라인은 영향받지 않는다

| S | llama.cpp 단일 대비 | 분해 (엔진 × B선택) |
|---|---|---|
| 1789 | **4.57×** | 4.51× × 1.013× (그림2 동일세션 측정) |
| 447 | **4.68×** | 4.14× × 1.132× |

⚠️ 위 분해의 `B` 선택 항은 **그림 2 의 세션 간 비교값**이다. anchor 기준으로는
`S=1789` 도 1.140× 이므로, 같은 세션에서 잰다면 엔진 항이 그만큼 낮게 나온다.
**곱은 보존되므로 최종 4.57× 는 유효하다** — 분해 비율만 설계에 따라 달라진다.

`B` 선택 이득이 1.0 이든 1.14 든 헤드라인 **4.5~4.7×** 는 유지된다. 시스템 기여가
지배적이고 `B` 선택은 부수적이다. **논문 중심을 시스템에 둔 판단이 이 데이터로
뒷받침된다** — 알고리즘 축의 이득은 재현성이 흔들리지만 시스템 축은 세 시스템
비교에서 견고하다.

---

### 7.19 13B — capacity 수치는 확보, 분산 실행은 미달성

**목적.** 속도 비교가 아니라 *8 GB 단독 노드에서 실행 불가능한 모델을 이종 CPU
클러스터에서 prefill 한다* 는 capacity 결과. 8B 는 16 GB 노드 한 대로 돌아가므로
"왜 굳이 분산?" 이 성립하지만, 13B 에서는 그 질문이 사라진다.

모델: `dllama_model_llama-13b_q40.m` (7.89 GB).
구조가 8B 와 다르다 — `d`=5120, `d_ff`=13824, **`d_kv`=5120(MHA, 8B 의 5배)**,
40 레이어(서브블록 80), vocab 128000.

#### 확보된 capacity 수치

| 구성 | RequiredMemory | 8 GB 노드(가용 7 GB) |
|---|---|---|
| N=1 (단일) | **9,167 MB** | **불가** |
| N=2 | **4,875 MB** | 가능 |

**분산이 노드당 요구 메모리를 절반 이하로 낮춘다.** 이 값은 모델 로딩 단계에서
계산·출력되므로 이후 크래시와 무관하게 유효하다.

단일 노드(root, 16 GB) 실행은 정상 동작했다 — `S=447`, 14 청크, 57.1 / 79.8 s.

#### ⚠️ 미달성 — `N≥2` 에서 segfault

`N=2` 에서 root 가 **재현적으로 SIGSEGV** 한다.

| 시도 | 결과 |
|---|---|
| S=447, N=2 | segfault (core dump) |
| S=112(짧은 프롬프트), N=2 | segfault — **프롬프트 길이 무관** |
| S=112, N=2, `--wave-pipeline 0` | segfault — **wave 경로 문제 아님** |

8B 에서는 `N=1~8` 이 모두 정상이므로 **모델 구조 차이와 관련된 우리 구현의 결함**이다.
후보는 `d_kv` 5배(MHA), 서브블록 80개, vocab 128000(llama3 토크나이저는 128256)이나
**어느 것인지 특정하지 못했다.** 원인 국소화에 실패했으므로 여기서 중단하고 기록만
남긴다 — 추적하려면 ASan/gdb 수준의 디버깅이 필요하다.

#### 정직한 서술 범위

    쓸 수 있음:  13B 단일 노드 요구량 9.2 GB > 8 GB 노드 가용량 7 GB.
                 2노드 분할 시 노드당 4.9 GB 로 감소.
                 단일 노드(16 GB) 실행은 정상 동작.

    쓸 수 없음:  "13B 를 클러스터에서 prefill 했다"  ← 미달성

capacity 메시지의 완성형("8 GB 노드들만으로 13B 를 돌린다")은 이 결함을 고쳐야
가능하다. **논문 헤드라인(8B, 4.57~4.68×)은 영향받지 않는다.**

원자료: `artifacts/model13b/`.

---

---

## 8. 평가 설계

### 8.1 지표와 oracle 용어

| 용어 | 정확한 정의 |
|---|---|
| analytic optimum | 비용 모델과 후보 집합 안에서의 최솟값. 하드웨어 oracle이 아님 |
| measured best-of-grid | 측정 전에 고정한 `(N,B,π,p)` 후보 중 가장 빠른 실측값 |

```text
attainment       = T_best-of-grid / T_DerivePP       # 1에 가까울수록 좋음
prediction error = abs(T_pred−T_meas)/T_meas
speedup          = T_baseline/T_DerivePP
```

평균만 보고하지 않고 길이별 값, geometric mean, worst case, 선택 실패 횟수를 함께 낸다.

### 8.2 모델 축 zero-shot 검증

3B/8B/13B에 대해 측정 전에 다음 내용을 timestamp가 남는 파일에 등록한다.

```text
모델 구조: d, d_ff, d_kv, L
캘리브레이션: T_q, T_a, 검증된 executor tax X, B_min, 링크 α/β
예측: S_k* 범위, 각 길이의 N*, B*, 노드 순서 π*, 경계 p*, 예상 TTFT
```

그 뒤 end-to-end 결과를 측정한다. 결과를 본 다음 식이나 상수를 바꾼 경우에는 같은
모델의 결과를 zero-shot이라고 부르지 않고 calibration/ablation으로 분리한다.
70B는 단일 노드 메모리 한계를 넘는 최종 시연으로 사용하되, 속도 일반성 검증을
대체하지는 않는다.

### 8.3 `ρ`의 통제된 섭동

thread 수와 activation buffer type을 바꾸면 Q4 GEMM과 F32 attention이 서로 다른
폭으로 변할 수 있다. 그러나 “`ρ`만 변한다”고 가정하지 않고 매 조건에서 모든 비용
함수를 다시 측정한다.

```text
조작
  → T_q, T_a, B_min, 링크/변환 비용 재측정
  → 바뀐 값으로 S_k* 범위, B*, 경계를 사전 예측
  → end-to-end 결과와 비교
```

핵심 관찰은 조작 자체가 아니라, 측정된 원인 변수가 바뀌었을 때 계획 전환이 식의
예측 방향과 크기를 따르는가이다.

### 8.4 자연적으로 관측된 연산자 이종성

| 실험 | 비교 | 사전 예측 |
|---|---|---|
| H0 | 단일 scalar 속도로 만든 기존 분할 | 짧은 `S`에서는 GEMM 순위와 대체로 일치 |
| H1 | 노드별 `T_q,T_a`를 사용한 분할 | 긴 `S`에서 attention이 빠른 노드의 배정량 증가 |
| H2 | fixed order vs subset-placement DP | 연산자 적합도와 느린 링크 횡단을 함께 개선 |
| H3 | 길이 `S` 변화 | 동일 노드의 상대적 유불리와 경계가 예측대로 전환 |

이미 자연 상태에서 GEMM–attention 순위 상관이 음수이므로 인위적 slowdown보다 이
조건을 먼저 사용한다. 코어·주파수 제한은 모델이 더 큰 이종성에서도 견디는지를 보는
보조 실험으로만 둔다.

### 8.5 네트워크 축

먼저 실제 두 스위치 군을 이용해 업링크 횡단 수가 다른 stage 순서를 만든다. 그 뒤
필요할 때만 `tc`로 100 Mb/s, 500 Mb/s, 1 Gb/s를 추가한다.
검증 대상은 단순한 TTFT 증가가 아니라 다음이다.

- recurrence가 링크 직렬화로 생긴 지연 전파를 맞히는가
- 느린 링크에서 `N*` 또는 경계가 예측대로 바뀌는가
- 공유 NIC 가정이 깨지는 지점을 trace가 보여 주는가
- placement DP가 느린 링크 횡단 수와 compute 적합도를 함께 trade-off하는가

### 8.6 단일 물리 플랫폼에서 가능한 주장

가능한 표현:

> 하나의 물리적 플랫폼에서 계산 처리율, 연산자별 이종성, 링크 조건을 통제된 방식으로
> 변화시켜 DerivePP의 인과적 예측이 넓은 운용 영역에서 성립함을 검증했다.

금지할 표현:

> 여러 CPU 마이크로아키텍처에 대한 이식성을 실증했다.

자동 캘리브레이션으로 다른 CPU에 적용할 수 있다는 것은 설계상 함의다. 두 번째 CPU가
없다면 cross-microarchitecture portability는 limitation에 명시한다.

---

## 9. Schedule Atlas

서버는 모델을 로드할 때 stage별 가중치 배치를 정해야 하므로 요청마다 경계를 바꾸기
어렵다. Atlas는 지원할 길이 범위에 대해 planner를 미리 실행하고 같은 계획이 연속되는
길이를 하나의 구간으로 합친 표다.

```text
[1, 180]       → N=1, B=..., nodes/order π_A, partition p_A, buffer bucket q_A
[181, 620]     → N=4, B=..., nodes/order π_B, partition p_B, buffer bucket q_B
[621, 2048]    → N=8, B=..., nodes/order π_C, partition p_C, buffer bucket q_C
...
```

런타임에는 `S`가 속한 구간을 lookup한다. 다만 `ceil(S/B)`, 패딩, causal `S²` 항,
이산 `N/B/p` 때문에 계획 비용을 **piecewise-linear**이라고 단정하지 않는다.
정확한 표현은 “이산적인 plan transition을 갖는 piecewise-defined atlas”다.

Atlas의 실용성은 두 운영 방식 중 하나로 평가한다.

1. 자주 쓰는 길이 bucket별로 모델 instance를 미리 로드한다.
2. `N,B`만 요청별로 바꾸고, `π,p,seqLen,nBatches`는 대표 길이별 소수 plan으로 제한한다.

가중치 재배치 비용이 TTFT보다 크다면 모든 요청마다 완전히 다른 `p`를 선택하는 설계는
실용적이지 않다. 이 비용을 제외한 채 lookup overhead만 보고해서는 안 된다.

---

## 10. 수정된 개발 계획

각 Phase는 “코드를 작성했다”가 아니라 **산출물과 통과 조건을 만족했을 때** 끝난다.

### Phase 0 — Baseline freeze `완료`

목표: 스케줄러 외의 변화가 성능 수치를 오염하지 않게 한다.

- attention 커널 실험 4개 기각 및 원복
- 정확성 gate 편차 0.000% 확인
- logit backpressure deadlock 수정, `S=8192` 완주
- benchmark commit, 빌드 플래그, 모델 hash, 실행 명령 고정

산출물: 재현 가능한 sanitized baseline과 변경 목록.

### Phase 1 — 측정 타당성 gate `핵심 통과, 잔여 항목 있음`

목표: calibration 숫자와 실제 runtime event가 같은 대상을 재도록 만든다.

| 우선순위 | 작업 | 현재 판단 |
|---:|---|---|
| 1 | **완료:** `DLLAMA_DUMP_STEPS`로 단일 노드 runtime graph 계측 | 838 execute op, 레이어당 26개(대략 att 14/ff 12), sync node 0 |
| 2 | **완료:** 동일 `NnExecutor` 경로의 no-op execute-step benchmark (`bench_steps.cpp`) | `h = 1.246 µs/step`. FFN 384 step = 잔차의 **0.3 %** → **순수 step tax 기각** |
| 3 | 분산 graph의 execute/sync step 수 분리 및 no-op/real synchronizer 비교 | `STEP_SYNC_NODES`를 compute barrier와 network event로 분리 |
| 4 | op별 thread 완료시각과 step wall time 계측 | 순수 step tax와 load-imbalance tax를 분리 |
| 5 | KV append on/off, CAST/MERGE_ADD on/off A/B | **보류.** 잔차가 6 % 로 잡음의 2배 수준이라 수익이 낮다. held-out 오차가 15 % 를 넘으면 재개 |
| 6 | **완료:** synthetic-weight executor calibrator (`calibrate_exec.cpp`) | 웜 기준 격차 **−6.0 %** → **15 % gate 충족**. `setDecodePhase(false)` 와 레이어 차분이 필수 |
| 7 | production attention kernel을 calibration에서 직접 호출 | 현재 surrogate는 절대시간 예측에 불충분 |
| 8 | Q/O, K/V, FFN up/down, lm_head 형상을 분리 측정 | 현재 `proj` 하나로 대용 중 |
| 9 | `sendActivation`의 start/end/recv-ready trace 추가 | echo의 RTT/2와 실제 protocol을 연결 |
| 10 | `nBatches`와 `seqLen` 교차 실험 완료 | 평가 고정값과 배포 bucket 결정 |
| 11 | 온도·주파수·background load·worker 단일성 기록 자동화 | 세션 드리프트 통제. **워밍업 3라운드 규율 추가**(§7.6a) |
| 12 | `94` 링크 timeout과 `191/prefix=512` 이상치 재측정 | calibration table 정리 |

통과 조건:

- raw production op와 kernel calibrator의 동일 `(B,prefix,shape)` 시간 차이가 대부분 5% 이내
- executor tax를 포함한 att/ff component 예측 오차가 held-out 조건에서 대부분 15% 이내
- `n_step·h_step`을 쓸 경우 `h_step`은 잔차 회귀가 아니라 no-op executor에서 독립 측정
- 청크별 `compute_end/send_end/recv_ready`의 순서가 trace와 recurrence에서 일치
- 같은 세션 교차 실행의 coefficient of variation과 이상치 규칙을 사전 고정

이 Phase가 끝나기 전에는 과소예측 원인을 특정하거나 `γ`를 추가하지 않는다. C가 실패하면
절대오차 기준을 낮추지 않고 A로 전환한다.

**현재 상태**: C 기각 → A 채택 → 웜 기준 격차 −6.0 %.
`γ` 를 끝까지 넣지 않고 통과했다. 남은 항목(3, 4, 7~10, 12)은 분산 조건과
형상 세분화이며, **비용 모델을 `calibrate_exec` 출력으로 교체하고 planner 재검증을
먼저 한다.** 교체 후 held-out 오차가 15 % 를 넘으면 남은 항목으로 돌아온다.

### Phase 2 — fixed-placement planner 완성 `부분 완료`

이미 완료된 것:

- 부분 청크와 causal geometry
- fixed-order completion-vector DP와 dominance pruning
- 작은 6개 인스턴스의 brute-force 일치
- 관측한 실규모 인스턴스에서 `Φ≤879`
- held-out `S=7,212`에서 `B=16` 순위 적중

남은 것:

1. §3.1의 링크 직렬화 `E_{k,j}`와 `D_return`을 simulator·DP·brute force에 구현
2. 무작위 작은 문제 1,000개와 adversarial `D≫C` 문제 대조
3. pruning on/off 결과 일치
4. `N=1..N_max`와 전체 calibrated `B` grid 평가
5. Python 10분을 줄이기 위한 vectorized/C++ 구현
6. frontier timeout과 메모리 guard 유지

통과 조건: exact recurrence 기준 brute force와 100% 일치하고, 최대 지원 입력의
atlas compilation 시간이 실용 범위에 들어온다.

### Phase 3 — placement × partition 공동 DP `새 핵심`

목표: ~~노드별 연산자 순위 역전~~(§7.15 철회)과 10배 링크 비대칭을 이용한다. 순위 역전 근거가 사라졌으므로 이 목표는 링크 비대칭만 남은 상태로 재검토가 필요하다.

1. §4.6의 `DP[U][v][i]` 구현
2. `K≤6,L≤12`에서 모든 node permutation × partition 전수열거와 일치
3. 고정 순서, compute-only 최적 순서, link-only 최적 순서, joint 순서 비교
4. 실제 8노드에서 frontier·시간·메모리 측정
5. 필요하면 스위치 군 연속 제약 variant의 최적성 gap 측정
6. PipeEdge/EdgeShard와 heterogeneous flow-shop·pipeline placement 문헌의 상태·목적함수 대조표 작성

통과 조건: joint planner가 fixed placement보다 예측상 다른 `π,p`를 선택하고,
그 변화가 실제 stage trace와 end-to-end 결과에서 재현돼야 한다. 계획이 같거나 이득이
3% 미만이면 placement는 완전한 핵심 기여가 아니라 모델 완전성/보조 기여로 낮춘다.

### Phase 4 — 시스템 통합과 정확성

목표: `(N,B,π,p,padding,buffer bucket)`이 실제 graph와 weight placement에 정확히
반영되게 한다.

1. planner plan manifest를 CLI/config로 전달
2. graph builder와 weight loader의 node/subblock owner assert
3. att|ff 중간 bridge와 node reorder 정확성 검증
4. 부분 청크, padding, 다른 `N`, 군 간 링크 순서의 회귀 테스트
5. 모든 plan에 동일한 logit/perplexity gate 적용

### Phase 5 — 핵심 end-to-end 평가

모든 비교는 같은 세션에서 순서를 교차하고, 다음 ablation을 분리한다.

권장 실행 단위는 `warm-up 1회 폐기 → ABBA 또는 무작위 순서의 paired block 5개 이상`이다.
절대시간의 서로 다른 세션 평균을 직접 나누지 않고, block별 paired speedup의 중앙값과
bootstrap confidence interval을 보고한다. 온도나 주파수가 사전 범위를 벗어난 block은
결과를 본 뒤가 아니라 미리 정한 규칙으로 기각한다.

| 구성 | 검증하는 것 |
|---|---|
| fixed baseline | sanitized 실행 기준 |
| `B` only | 이미 확인된 table-based chunk 선택 |
| partition only | 노드별 `T_q,T_a`의 경계 선택 기여 |
| placement only | subset/order와 링크 모델 기여 |
| partition + placement | 공동 DP의 상호작용 |
| full DerivePP | `N,B,π,p` 전체 |
| measured best-of-grid | 사전 등록한 실측 후보의 상한 |

길이는 짧은 구간, 각 노드의 `S_k*` 범위, 중간, 장문, held-out `S=4,096`을 포함한다.
prediction error, attainment, geometric-mean speedup, worst-case slowdown을 함께 보고한다.

### Phase 6 — 모델·조건 일반성

1. production-path calibration 후 3B/13B 계획 사전 등록
2. 모델별 kernel shape calibration과 `S_k*` 검증
3. 자연 operator heterogeneity와 스위치 군 횡단 실험
4. 필요할 때만 thread/frequency와 `tc` 조건 추가
5. 가능하면 70B feasibility 시연

### Phase 7 — Atlas와 artifact

- `(π,p,seqLen,nBatches)`를 포함한 길이 bucket plan 컴파일
- calibration 및 atlas compilation 시간·메모리 보고
- raw logs, rejected points, preregistration, planner source 공개
- full experiment manifest와 one-command reproduction 제공

---

## 11. Kill criteria

기준은 실험 후 유리하게 바꾸지 않는다.

| 지표 | 판정 기준 |
|---|---|
| kernel calibration equivalence | raw production op 대비 대부분 5% 초과 → kernel calibrator 수정 |
| executor calibration equivalence | held-out att/ff component 오차 대부분 15% 초과 → 순수 barrier 항 기각, executor-backed layer calibration으로 전환 |
| barrier 주원인 가설 | 독립 측정한 execute-step tax가 FFN 잔차의 50% 미만 설명 → 주원인 주장 기각. 유효한 작은 항은 보존하되 thread-skew/A/KV/copy 진단 계속 |
| `R_k(S)` 오차 | 노드·모델별 20% 초과 → 구조식 수정 또는 일반성 주장 축소 |
| `S_k*` 오차 | 30% 초과 → 모델축 역전점 주장 철회 |
| `B` 선택 | held-out 길이에서 table 선택이 실측 최소 plateau 안에 위치. `B=B_min` 법칙은 주장하지 않음 |
| stage/chunk 비용 | 큰 체계적 잔차가 남으면 DP 전에 비용 모델 수정 |
| 예측 TTFT | executor·KV·copy 항 확정 후 대부분 15% 이내. 그 전 수치는 순위 검증으로만 사용 |
| best-of-grid attainment | `<90%` 수정 / `90~95%` 유효 / `>95%` 강함 |
| 기존 heuristic 대비 | 전체 길이 geometric mean 1.2× 이상 |
| 서브레이어 분할 단독 | `<3%` 보조로 강등 / `3~8%` 보조 / `>8%` 독립 기여 |
| placement 단독 | `<3%` 보조로 강등. 링크 모델의 정확성 결과는 별도 유지 |
| joint `π,p` | fixed-order partition보다 일관된 개선이 없으면 핵심 기여 문구 축소 |
| controlled `ρ_k` | `S_k*` 이동 방향 일치, 크기 오차 30% 이내 |
| 이종성 | 병목 노드와 경계 이동 방향 일치 |
| 네트워크 | 계획 전환 또는 TTFT 오차 20% 이내 |
| exact DP | 직렬 link recurrence를 포함한 작은 brute-force 문제 100% 일치 |
| frontier `Φ` | 모델·길이·placement별 최대·중앙값·메모리·pruning 시간 공개 |

결정적 위험은 현재 1.19~1.38× 결과가 `B` heuristic 하나만 고치는 결과로 끝나는 것이다.
TPDS 기여가 되려면 `(N,B,π,p)`의 공동 결정이 여러 길이,
모델 구조, 이종성, 링크 조건에서 일관되게 measured best-of-grid에 접근해야 한다.

---

## 12. 기각 기록

### 12.1 커널과 런타임 아이디어

같은 가설을 다시 세우지 않기 위해 성능 기여와 분리해 보존한다.

| 시도 | 결과 |
|---|---|
| 정확한 KV tile skip | 달성 가능한 상한이 S=2048에서 17/13,762,560, 사실상 0.00% |
| QKᵀ 4×4 register blocking | microbench 1.81×, in-situ 이득 0 |
| AV 4×4 blocking | V tile이 L2를 넘어 1.71× 악화 |
| KV layout 변환 | 4 kB stride 손해가 4%뿐 |
| query fusion G=2/4 | 1.00×. 바깥 `t` loop가 이미 K 재사용 |
| stage-local microbatch fusion | 실제 pipeline 단위가 `G·B`가 되어 큰 `B`와 동일 |
| SlackAssist-PP | 같은 scheduling slack을 두고 경쟁하고 환경 의존성이 큼 |

이 결과들은 DerivePP의 contribution이 아니라 baseline hygiene와 design-space pruning의
근거로 짧게 보고한다.

---

### 12.2 측정에서 나온 기각 (스케줄링)

| 주장 | 반증 |
|---|---|
| `B ∝ S` (`S/(8N)`) | `B=128` 이 `B=16` 보다 1.38× 느림 |
| `B ∝ √S` | 현재 바이너리의 세 길이에서 모두 16. 과거 32점은 다른 시스템 상태 |
| `B_ε`(버블 예산)가 청크를 직접 정한다 | 비매끄러운 kernel table을 설명하지 못함 |
| `B₀`(고정비 균형)가 청크를 직접 정한다 | `B=24` notch를 설명하지 못함 |
| `B*=B_min`이 보편 법칙이다 | 아직 반증도 확증도 안 됨. 현재 한 모델·N=8·세 길이의 관측일 뿐 |
| `nBatches=256` 이 24 % 를 먹는다 | 실측 2.6 %. 나머지는 **세션 드리프트** |
| 링크 차이는 RAM/기기 차이 | 전 노드 `1000Mb/full`, 8 GB 군 내부도 115.6 MB/s |
| ~~노드 이종성을 스칼라 `s_k` 로 표현 가능~~ | ~~연산자 순위가 역전 (τ = −0.29)~~ → **이 반증을 철회한다(§7.15).** 9노드 재측정 τ = +0.111. 단 이것이 스칼라 모델을 **입증하지도 않는다** |
| op 단위 대신 barrier-free 레이어 단위로 재면 절대오차가 해결된다 | 18.0 s→18.7 s로만 증가, production 24.0 s보다 여전히 22% 낮음 |
| weight working set/DRAM 행 지역성이 25% 격차의 주원인 | 122 MB→1.95 GB로 16배 늘려도 20.90→22.19 ms, 6% 증가뿐 |
| 10 µs barrier가 20% 격차를 설명한다 | 전체 832 step을 모두 귀속해도 8.32 ms/chunk로 잔차의 4.4%. FFN 약 384 step만 대응시키면 3.84 ms, 2.0% |

### 12.3 코어 분할 재진입 파이프라인 (CoRePP)

**아이디어.** stage 안에서 직렬로 도는 Attention–FFN을 코어를 나눠 서로 다른
microbatch에 대해 동시 실행한다. `FFN(l,j) ⊥ Attention(l,j+1)`이므로 의존성은
실제로 깨지지 않는다 — 구조 자체는 타당하다.

**기각 근거.** 이득의 필요조건이 성립하지 않는다. 두 lane이 모두 코어 수에
선형으로 확장되면 분할 이득은 정확히 0이고, 이득은 **확장이 나쁜 쪽**에서만 나온다.
`calibrate_perlayer`로 1/2/3/4 스레드 확장 곡선을 재면:

| prefix | att 확장(4코어) | ff 확장(4코어) | 직렬 | 최적 동시 | G |
|---|---|---|---|---|---|
| 8192 | 3.41× (85%) | 2.51× (63%) | 261.4 ms | 270.2 ms | **0.97** |

긴 prefix에서 attention이 작업의 79%(207 vs 54 ms)를 차지하므로, 코어를 떼어
FFN lane에 주면 attention이 느려지는 손실이 FFN 이득을 넘는다. 게이트 기준
`max_r G ≥ 1.10`에 미달이며, 방향이 이득이 아니라 **손해**다.

이득이 가능한 구간은 두 lane의 작업량이 비슷해지는 `att share ≈ 0.5`
(`prefix ≈ 3,000`, 역전점 `S* ≈ 3,354`와 일치) 부근뿐이다. 이는 prefill 전체가
`prefix: 0 → S`를 훑는 동안의 **일부 구간**이고, 그 구간의 상한조차 위 표의 다른
행들이 오염돼 확인되지 않았다. 실제 동시 실행은 DRAM/L3 경합으로 상한보다 낮다.

**측정 오염 (기록).** `prefix=512`/`2048` 행은 쓸 수 없다. `CAL_LAYERS=2`로 줄이면서
layer 0 인공물이 평균의 절반을 차지해 `1코어→2코어 8.7×`(물리적 상한 2×) 같은 값이
나왔다. 짧은 prefix 판단이 필요하면 `CAL_LAYERS=8`로 재측정해야 한다.

**부산물 (기각 아님).** FFN이 **3코어에서 포화**한다. 4번째 코어가 52.7 → 54.1 ms로
오히려 악화되며, `prefix=512`와 `8192` 양쪽에서 재현된다. Pi5의 4번째 코어가 FFN
GEMM에 기여하지 않는다는 관측으로, 자원 모델 논의에서 별도로 쓸 수 있다.

---

### 12.4 CP island — capacity-contraction 상한 (구현 없이 판정)

`prefill_bench/cp_gate.py`. 구현 전에 낙관적 상한만으로 판정한다.

#### Balanced-capacity contraction lemma

동일한 물리 worker `N` 대로 이루어진 **이미 균형 잡힌** 파이프라인을 가정한다.
각 physical stage 의 정상상태 service time 이 `c` 이고, 두 worker 를 하나의
CP/TP group 으로 묶어도 group 의 service time 이 `c` 보다 **작아지지 않는다면**,
logical stage 수를 `N → N−1` 로 줄여 얻는 이상적 speedup 은

    T_N / T_{N−1}  ≤  (M + N − 1) / (M + N − 2)

이며, 순수 bubble 이득은 **`1 / (M + N − 2)`** 이다.
`M=7, N=8` 이면 `1/13 = 7.69%`, `M=226, N=8` 이면 `1/232 = 0.43%`.

**적용 조건 (반드시 명시).** 이 보조정리는 다음 경우 성립하지 않는다.

- 기존 PP 가 불균형인 경우
- CP/TP 가 stage service time 을 **실제로 줄이는** 경우
  (cache·메모리 대역폭·커널 효율이 바뀌어 `c` 자체가 내려가는 경우 포함)
- global TP 처럼 파이프라인 의존 그래프 자체가 달라지는 경우

따라서 "모든 stage 내 병렬화가 같은 이유로 죽는다"는 **너무 넓은 주장이며 쓰지
않는다**. 정확히는:

| | |
|---|---|
| CoRePP (§12.3) | logical stage 수가 그대로 → 선형 확장이면 이상적 이득이 **0** |
| CP2 island | 위 조건이 성립할 때 **bubble 상한만** 남음 |
| TP | 동일한 local island 구조로 쓸 때만 같은 보조정리 적용 |

균형 가정의 근거는 J1b(§7.9)이지만, 그것은 `S=1789` 의 **static** partition 관측이다.

**⚠️ `S=7212` 에서는 균형 전제가 실제로 깨진다.** §7.12b 가 측정으로 보였다 — drain 이
균형 시 기대값 `(N−1)·C ≈ 6 s` 대비 22~40 s 로 **4~7배**다. 즉 root 가 앞서 달려 큐가
쌓이는 불균형 상태다. 따라서 이 보조정리에 근거한 CP 기각은 **`S=1789` 처럼 균형이
확인된 조건에 한정**되며, 긴 프롬프트에서는 CP 가 균형 개선으로 이득을 낼 여지가
아직 배제되지 않았다(반대로 island 이 균형을 더 깨뜨릴 위험도 함께 있다).
긴 길이의 CP 판정은 §7.12 의 재측정이 선행되어야 한다.

#### 통신 회계

exact CP 에서 KV 를 shard 로 유지하려면 helper 가 **모든 `B` query** 에 대해 자기
shard 를 처리해야 한다. 레이어당 최소 payload:

    owner  → helper : Q             ≈ B·d
    helper → owner  : partial O     ≈ B·d       (+ softmax 통계, 무시할 수준)
    owner  → helper : 새 K/V append ≈ B·2·d_kv

Q 전송과 partial 회수는 **서로 다른 의존 단계**라 full-duplex 로 상쇄되지 않는다.
`B=32` 에서 레이어당 327,680 원소 = 348 KB.

query 를 `B/2` 씩 나누는 대안은 각 query 가 반대편 KV shard 도 봐야 하므로 prefix
비례 KV ring/all-gather 가 되살아난다. **"B/2 query 전송 + KV 이동 없음" 을 동시에
만족하는 프로토콜은 존재하지 않는다.**

#### 결과

| M | S 대응 | 이상적 상한 `1/(M+N−2)` | intra-switch | inter-switch |
|---|---|---|---|---|
| 7 | 224 | +7.69% | +4.08% | −19.8% |
| 28 | 896 | +2.94% | −0.51% | −23.3% |
| 56 | 1789 | +1.61% | −1.79% | −24.3% |
| 226 | 7212 | +0.43% | −2.93% | −25.2% |

**판정.** CP2 island 의 매우 낙관적인 capacity-contraction 상한은 긴 프롬프트에서
**0.43%** 에 불과하다. 사전 정의한 게이트("낙관적 상한 10% 이상이면 프로토타입")를
전 구간에서 통과하지 못하며, 가장 유리한 조건(`M=7`)에서도 +7.69% 다. 실제 통신을
넣으면 `M≥28` 에서 부호가 음수가 된다.

placement 제약도 확인된다 — island 을 inter-switch 링크에 걸면 −20% 이상으로
즉사한다. 다만 **intra-switch 도 통과하지 못하므로** placement 최적화로 살릴 수 없다.

**남은 정밀화.** 위 payload 는 하한 성격의 추정이다. 실제 wire protocol 을 정확히
회계하면 통신은 늘어날 뿐이므로 상한은 더 내려간다. 즉 이 판정을 뒤집으려면
보조정리의 **적용 조건이 깨져야** 한다 — 구체적으로 CP 가 stage service time 을
실제로 줄인다는 증거가 필요하다. 그 증거가 나오기 전에는 구현하지 않는다.

---

## 13. 논문 기여 문장

> DerivePP 는 저속 Ethernet 으로 연결된 commodity CPU 클러스터에서 microbatch wave
> pipeline 을 통해 통신을 계산과 중첩하며, 기존 공개 분산 구현들이 로컬 CPU 성능의
> 0.41~0.44× 에 머문 환경에서 **4.57~4.68×** 를 달성한다.

### A. 실증된 기여

1. **저속·비대칭 Ethernet 에서 확장되는 CPU prefill wave pipeline.**
   프롬프트를 microbatch 로 나눠 layer stage 에 wave 로 흘려보내, 느린 링크를
   건너는 activation 전송을 다른 stage 의 계산과 중첩한다. 링크가 10배 비대칭인
   조건(군 내부 115.7 MB/s vs 군 간 11.7 MB/s)에서 8노드 **6.64×**(효율 83%).

2. **공개 CPU 분산 구현 두 종류와의 공정한 비교** (§7.16, §7.17).
   동일 하드웨어·모델·양자화·스레드 수·프롬프트 토큰 수, 모델 로드 제외,
   생성 단계 미포함. 실패 원인을 **소스로 확인**했다 — RPC 는 비동기 경로 부재,
   공식 dllama 는 업링크 붕괴.

3. **계산·통신 중첩의 실증.** 공식 distributed-llama 가 붕괴한 **동일한 구성**
   (8 GB 군을 포함해 100 Mb 업링크를 건넘)에서 본 연구는 `3.82× → 6.64×` 로
   확장 폭이 오히려 커진다.

4. **AxisCert — 보조 계획 절차.**
   > 효과 없는 최적화 축을 배제하고, 플랫폼에서 실제로 유효한 실행 설정만 선택한다.

   4.6× 를 만든 핵심 알고리즘이 **아니다.** 축을 자르는 데 필요한 것이 절대 지연
   예측이 아니라 **계획 쌍 비교의 교정 정확도** `δ_x` 임을 보인다(§7.11) —
   절대 오차가 길이에 따라 무너져도(0.912 → 0.573) `δ_p` 는 5.9~7.1% 로 유지된다.

5. **효과가 없었던 축의 구조적 분석.** sublayer DP(§7.9, §7.12c), CoRePP(§12.3),
   CP island(§12.4). 단순 negative 나열이 아니라 **왜 볼 필요가 없는지**를
   balanced-capacity contraction lemma 로 보인다.

6. **연산 계층 — SharedPack-SDOT** (`research/19`, `research/21`).
   wave 파이프라인 위에서 N=8 TTFT 를 **1.436×**(통합 8쌍, 범위 1.403~1.465)
   추가 단축하며 로짓이 비트 단위로 동일하다.

   > **주의 1.** 이 1.436× 를 4.57~4.68× 에 곱하지 않는다. 베이스라인·세션·조건이
   > 다르다. 최종 시스템 대 llama.cpp 는 동일 조건에서 다시 재야 한다.
   >
   > **주의 2.** N=1 operator 개선 1.205× 보다 큰 이유는 **아직 분해되지 않았다.**
   > `syncWait` 감소는 앞 stage 완료시각에서 파생된 결과이므로 별도 이득으로
   > 더하면 이중 계산이다. 분해 설계는 `research/21` §5.
   >
   > **주의 3.** 기법 자체("pack once, reuse")는 FBGEMM·MKL packed API·llama.cpp
   > `repack.cpp` 에 선행이 있다. 독립 기여로 세우려면 `research/21` §4 의
   > 자동 결정(graph 위치·소유권·생명주기·cache gate)까지 필요하다.
   > **현재는 DerivePP 의 연산 기여로 제시한다.**

7. **측정 방법론.** anchor 반복 + 정순/역순 + `flock` 배타 실행으로 CV 0.6~1.35%.
   실패 세션은 `invalid_session` 으로 분리 보존. 모든 수치는 `artifacts/<run-id>/`
   에 manifest·checksum 과 함께 고정.

   여기에 **검증 계층의 반증 사례**를 더한다(`research/19` §8.7): 커널이 심각하게
   깨진 상태(`kBlocks=1`, 버퍼 3.5배 부족)에서도 **E2E 로짓 해시가 비트 단위로
   일치해 통과했다.** 따라서 출력 해시는 커널 정확성의 검증기가 아니며,
   커널이 소비하는 중간 표현을 직접 대조해야 한다. 검증 계층을
   `packed bytes -> GEMM output -> stage output -> final logits` 로 나누고,
   검증기가 "실행했고 0건" 과 "실행하지 않음" 을 artifact 만으로 구별하도록
   종료 시 항상 요약을 출력한다.

   > bit-identical 정확성 자체는 **기여가 아니라 요구사항**으로 취급한다.
   > *"다른 엣지 가속은 정확도를 희생한다"* 는 비교는 근거가 없으므로 쓰지 않는다.

### B. 검증 후보

8. **Position-dependent pipeline formulation** — max-plus 완료시각 모델. 형식은
   서 있으나 절대 예측이 길이에 따라 무너진다(§7.14).
9. **Completion-vector partitioning DP** — 정확성은 brute force 6/6 일치이나
   **이득이 확인되지 않았다.** 분할축이 두 길이 모두에서 비활성이다.
10. **Operator/topology-aware placement `π`** — 근거였던 순위 역전이 재현되지
   않았다(τ −0.29 → +0.111, §7.15). **현재 기여로 제시할 근거가 없다.**

### C. 반증·기각

CoRePP(§12.3), CP island(§12.4), 커널 최적화 4종·KV skip(§12.1),
`B ∝ S`·`B ∝ √S`(§12.2), 연산자 순위 역전에 근거한 스칼라 모델 반증(§7.15).

---

## 14. TPDS 제출 판단

| 완성 상태 | 판단 |
|---|---|
| 해석식과 소수 측정점만 | 제출 단계 아님 |
| 비용 모델 + heuristic lookup | Pi 전용 경험 규칙으로 보일 위험 |
| **세 시스템 공정 비교 + 4.57~4.68× + 실패 원인 소스 확인** | **현재 위치.** 시스템 논문으로 성립 |
| 위 + 공식 baseline 로그·manifest 완비 | 제출 가능 |
| 위 + 13B capacity + held-out 모델 검증 | 강한 제출 |

**연산 계층의 novelty 경로** (`research/21` §6). SharedPack 을 어디까지 밀지에 따라
기여의 위치가 달라진다.

| 수준 | novelty |
|---|---|
| 현재 구현 그대로 | 독립 알고리즘 기여는 **약함** — 선행(FBGEMM / MKL packed API / llama.cpp `repack.cpp`)과 구별되지 않는다 |
| DerivePP 내부의 연산 기여 | **충분히 강함** — 현재 여기로 제시한다 |
| graph-hoisted packing + cache gate + cross-projection reuse + lifetime 관리 | **중간 이상** |
| 위 + pipeline critical-path 모델 + 자동 선택 + 다중 모델·CPU 검증 | **주요 기여 후보** |

**우선순위 (2026-08-19 기준).**

1. ~~문서 헤드라인을 시스템 중심으로 재정렬~~ — 완료(§0, §13, §14)
2. **공식 distributed-llama 재측정 + 전체 로그·manifest 보존** — 1차 측정의 원자료가
   재부팅으로 소실됐다. `0.44×` 비교표가 논문 핵심이므로 **최우선**이다.
   `B` 결과가 1.01× 여도 헤드라인은 유지되지만, 공식 baseline 로그가 없으면
   비교표 전체가 공격받는다.
3. `S=1789` 의 `B=16/B=32` **anchor 재측정** — §7.18 의 불일치 해소
4. 랩미팅 자료 갱신
5. **13B feasibility** — 속도가 아니라 *8 GB 단독 노드에서 실행 불가능한 모델을
   이종 CPU 클러스터에서 prefill 한다* 는 capacity 결과. 별도 메시지로 둔다.
6. held-out 모델 검증 (시간 여유 시)

**주의.** `N` 축은 아직 A/B 를 하지 않았다. 확장 곡선의 `N` 은 노드 구성과 교락돼
있으므로(`N≤4` 는 16 GB 군만, `N=8` 은 8 GB 군 포함) 순수 `N` 효과가 아니라
**deployment plan `(N, π)`** 비교로 표기해야 한다.
