# CPU Prefill 연산 경로 — 무엇이 병목이고 어떻게 SharedPack까지 왔는가

이 문서는 [18-compute-path.md](../18-compute-path.md)를 처음 읽는 사람을 위한 해설이다.
DerivePP의 pipeline scheduling이 아니라, **한 CPU 노드 안에서 prefill 연산 자체를 어떻게
빠르게 만들 것인가**에 초점을 둔다.

가장 중요한 현재 결론은 다음 한 문장이다.

> Llama-3 8B의 짧은 prefill에서는 Q4×Q8 GEMM이 전체 시간의 약 84%를 차지한다.
> Down GEMM의 4-thread 성능 저하는 긴 `K` 계산 자체가 아니라, 동일 activation을
> 스레드마다 따로 pack해 네 개의 독립 cache footprint를 만드는 구현에서 발생했다.

---

## 1. 먼저 알아야 할 연산 구조

Transformer의 FFN은 다음 세 행렬곱을 사용한다.

```text
입력 activation
   ├─ Gate projection (W1) ── SiLU ─┐
   └─ Up projection   (W3) ──────────×── hidden activation
                                         │
                                         └─ Down projection (W2) ── 출력
```

Llama-3 8B에서 행렬 형상은 대략 다음과 같다.

| 연산 | 입력 차원 `K` | 출력 차원 `N` | 의미 |
|---|---:|---:|---|
| Gate / Up | 4,096 | 14,336 | model dimension을 FFN hidden dimension으로 확장 |
| Down | 14,336 | 4,096 | FFN hidden dimension을 model dimension으로 축소 |

Gate/Up과 Down은 `K×N`이 같으므로 weight 원소 수와 MAC 수가 거의 같다. 그런데
production 측정에서는 Down이 Gate/Up보다 약 1.5배 느렸다. 이 차이가 연산 경로 연구의
출발점이었다.

### Q4×Q8은 무엇인가

```text
weight      : Q4_0, 원소당 약 4 bit
activation  : Q8_0, 원소당 8 bit + block scale
핵심 명령   : ARM SDOT
출력        : F32
```

네 token row를 한 번에 처리할 때 activation은 `block_q8_0x4` 레이아웃으로 재배치된다.
이 문서에서 **pack**은 Q80 값을 새로 근사하는 양자화가 아니라, 이미 존재하는 Q80 네 행을
SDOT 커널이 읽기 좋은 순서로 재배치하는 것을 뜻한다.

---

## 2. 왜 scheduling보다 연산 경로를 우선하게 되었는가

`S=447`, `B=32`, 단일 노드에서 executor가 측정한 전체 prefill 시간은 34,346.5 ms였다.

| 구간 | 전체 시간 비중 |
|---|---:|
| Down GEMM | 29.44% |
| Gate + Up GEMM | 39.28% |
| Q/K/V/O 등의 projection GEMM | 약 15.4% |
| **Q4×Q8 GEMM 합계** | **약 84%** |
| attention core | 9.15% |
| 나머지 elementwise/cast/norm 등 | 약 6.7% |

어떤 구간의 비중을 `f`, 그 구간의 가속률을 `r`이라 하면 전체 가속률은 Amdahl 식으로
계산한다.

```text
E = 1 / ((1-f) + f/r)
```

전체의 84%인 GEMM을 평균 1.25배 가속하면 다음과 같다.

```text
E = 1 / (0.16 + 0.84/1.25) ≈ 1.20×
```

따라서 CPU 연산 연구의 1차 목표를 Q4×Q8 GEMM으로 잡은 것은 단순한 직관이 아니라
시간 비중에서 나온 결정이다.

---

## 3. 먼저 탈락한 연산 아이디어

### 3.1 WCEP — 실제 weight 배열에 맞춘 산술 합성

WCEP는 Q4 weight 값의 반복 구조를 이용해 SDOT 대신 더 짧은 exact 산술 DAG를 만들려는
시도였다. 실제 weight만 보면 symbolic operation이 약 45.8% 줄어드는 것처럼 보였다.

그러나 tile 내 값 빈도를 보존하고 위치만 섞은 matched null도 같은 절약을 냈다.

```text
actual weight 대비 tile-hist null 효과 = -0.37 %p
95% CI = [-0.39, -0.35] %p
```

즉 절약은 학습된 weight의 특별한 공간 구조가 아니라 Q4 값 분포 자체에서 나왔다.
weight-aware novelty 가설은 기각됐다. 자세한 과정은
[17-weight-compiled-prefill.md](../17-weight-compiled-prefill.md)를 따른다.

### 3.2 QCFuse — FFN 중간 F32 버퍼 제거

QCFuse는 Gate/Up 결과, SiLU, 곱셈, Down 입력 양자화를 하나의 producer 연산으로 묶어
대형 F32 중간 버퍼를 없애려는 시도였다.

실제 제거 대상의 합은 전체 prefill의 2.45%뿐이었다.

```text
block_mul       1.12%
block_act       0.65%
block_cast_d2   0.56%
block_cast_y3   0.11%
합계            2.45%

무한 가속 상한 = 1 / (1 - 0.0245) = 1.025×
```

F32 버퍼는 커 보이지만 각 elementwise op는 원소당 한두 번만 접근한다. 반면 GEMM은
각 출력 원소에 대해 수천~수만 번의 MAC을 수행한다. 그래서 중간 버퍼 비용은 GEMM에
묻혔다.

> 여기서 기각된 것은 graph-level F32 materialization 제거다. GEMM 내부의 activation
> pack 비용은 `block_matmul_*` 시간 안에 들어 있으므로 이 측정으로 기각되지 않았다.

---

## 4. Down은 왜 느려 보였는가

### 4.1 형상 교차 실험

총 `K×N`을 거의 고정하고 Gate/Up 형상에서 Down 형상까지 `K`만 늘렸다.

```text
K= 4096, N=14336
K= 6144, N= 9556
K= 8192, N= 7168
K=10240, N= 5732
K=12288, N= 4776
K=14336, N= 4096
```

단일 스레드는 모든 형상에서 약 85 GOPS로 같았다. 따라서 긴 `K`가 산술적으로 비효율적인
것은 아니었다. 하지만 기존 production packing을 사용한 4-thread 결과는 다음처럼
떨어졌다.

| `K` | 16-row packed panel | 4-thread GOPS |
|---:|---:|---:|
| 4,096 | 68 KiB | 304.0 |
| 8,192 | 136 KiB | 291.6 |
| 10,240 | 170 KiB | 289.2 |
| 12,288 | 204 KiB | 248.8 |
| 14,336 | 238 KiB | 189.7 |

처음에는 이를 “긴 K의 activation panel이 cache에 들어가지 않는다”라고 해석했다. 그래서
K를 작은 panel로 나누는 KP-SDOT을 제안했다.

### 4.2 이 해석에 무엇이 빠졌는가

기존 커널은 output column을 스레드별로 나눈다. 모든 스레드가 batch 전체 activation을
필요로 하지만 op 내부 barrier가 없기 때문에 다음처럼 동작했다.

```text
thread 0 : Q80 전체를 private scratch 0에 pack
thread 1 : Q80 전체를 private scratch 1에 pack
thread 2 : Q80 전체를 private scratch 2에 pack
thread 3 : Q80 전체를 private scratch 3에 pack
```

K-sweep은 모든 점에서 이 **duplicate packing 정책을 고정**하고 있었다. 따라서 `K`가
증가할수록 panel 크기뿐 아니라 네 개의 독립 packed footprint도 함께 커졌다.

즉 당시 관측은 다음 두 효과를 분리하지 못했다.

```text
K 증가에 따른 순수 계산/working-set 효과
스레드별 중복 pack 사본이 커지는 효과
```

이것이 교락(confounding)이다.

---

## 5. 교락을 어떻게 분리했는가

같은 weight, activation, GEMM, thread 수를 유지하고 packing 정책만 바꿨다.

```text
dup     : 각 스레드가 자기 주소에 activation 전체를 pack
shared  : 한 번 만든 packed activation 하나를 모든 스레드가 읽음
```

### K=14,336, B=32

| thread 수 | dup GOPS | shared GOPS | shared/dup |
|---:|---:|---:|---:|
| 1 | 84.5 | 84.1 | 1.00× |
| 2 | 169.8 | 170.9 | 1.01× |
| 3 | 245.5 | 252.5 | 1.03× |
| **4** | **198.7** | **333.8** | **1.68×** |

### 4-thread에서 K/panel 크기를 다시 변화

| 16-row panel | dup GOPS | shared GOPS |
|---:|---:|---:|
| 68 KiB | 318.4 | 329.6 |
| 102 KiB | 312.3 | 330.9 |
| 136 KiB | 305.9 | 327.1 |
| 170 KiB | 297.5 | 327.4 |
| 204 KiB | 252.9 | 330.9 |
| 238 KiB | 198.7 | 333.8 |

shared는 68~238 KiB에서 327~334 GOPS로 평탄하다. 반면 dup만 318→199 GOPS로
무너진다.

### 인과적으로 무엇을 말할 수 있는가

```text
잘못된 결론:
  K가 길거나 panel이 크면 본질적으로 느리다.

현재 결론:
  현재 커널의 4-thread duplicate-packing 조건에서 K가 길어지면 네 개의 독립
  packed footprint가 커지고 확장성이 무너진다. packed activation 하나를 공유하면
  같은 긴 K에서도 붕괴가 사라진다.
```

따라서 `K`가 완전히 무관하다는 표현도 부정확하다. `K`는 baseline에서 중복 footprint의
크기를 키우는 **증폭 변수**다. 직접 원인은 `K` 산술 자체가 아니라
`K × 4 threads × duplicate packing`의 상호작용이다.

“트래픽이 정확히 4배”라고도 쓰지 않는다. shared에서도 네 스레드의 논리적 read는 남는다.
차이는 동일 cache line을 공유할 수 있는 한 사본인지, 서로 다른 주소의 네 사본인지다.
PMU cache counter가 없으므로 정확히 어느 cache level이 지배했는지는 아직 단정하지 않는다.

---

## 6. 왜 KP-SDOT을 중단했는가

KP-SDOT은 K를 작은 panel로 나눠 activation working set을 줄이는 방법이었다. 하지만 shared
packing에서는 238 KiB panel도 333.8 GOPS를 냈다. 즉 panel 크기를 줄이지 않고도 병목이
사라졌다.

```text
문제                 실제 처방
─────────────────────────────────────────────────
큰 K panel 자체       처방 불필요
네 개의 packed 사본   하나의 shared packed buffer
```

KP-SDOT prototype에서 panel 경계마다 F32 accumulator를 저장하고 다시 읽어도 세 panel
크기 모두 bit-identical이라는 사실은 확인했다. 이는 보조 결과로 보존하지만 production
우선순위에서는 제외한다.

---

## 7. Microbenchmark 상한에서 production 결과까지

Down microbenchmark에서:

```text
dup     198.7 GOPS
shared  333.8 GOPS
kernel-only speedup = 1.68×
```

그러나 shared buffer는 timed loop 밖에서 미리 만들었다. 따라서 333.8 GOPS에는 다음 두
비용이 빠져 있다.

```text
pack을 한 번 수행하는 시간
pack을 마친 뒤 네 worker가 만나는 barrier/executor 경계
```

production 시간은 반드시 다음으로 평가한다.

```text
T_shared-production = T_pack-once + T_executor-boundary + T_GEMM(shared)
```

Down 비중 29.44%에 kernel-only 1.68배를 적용하면 Down만으로 전체 약 1.129배가 상한이다.
Gate/Up과 Q/K/V/O까지 단순 합성한 사전 예상은 약 1.16배였다.

이후 별도 pack op를 production graph에 넣고 Q/K/V, Gate/Up, Down으로 확장했다.

```text
Down pack-inclusive    약 1.64×
pack 세 op 합          84.2 ms, 새 누적 op 시간의 0.30%
기존 누적 op profile   34,346.5 → 28,511.0 ms = 1.205×
N=1 B=16/B=32          테스트 6회 logits bit-identical
```

이 초기 1.205배 비교는 사전 예상을 넘었지만 서로 다른 세션에서 얻었고 decode-inclusive였다.
현재는 동일 바이너리의 `DLLAMA_SHARED_PACK=0/1`, decode 전 `[PREFILL ONLY]` snapshot과
paired anchor를 사용해 최종 수치를 재확정 중이다. 그러므로 1.205배는 구현 성공의 강한
증거지만 최종 E2E 논문 수치로 아직 고정하지 않는다.

---

## 8. 이 문서를 읽은 뒤 확인할 질문

다음 질문에 답할 수 있으면 [19-sharedpack-implementation.md](../19-sharedpack-implementation.md)와
[13-sharedpack-sdot.md](13-sharedpack-sdot.md)로 넘어갈 준비가 된 것이다.

1. `block_cast_y3`와 GEMM 내부 Q80→Q8×4 pack은 왜 다른가?
2. QCFuse의 2.45% 상한은 왜 SharedPack을 기각하지 못하는가?
3. 기존 K-sweep만으로 cache-residency를 원인이라고 단정할 수 없었던 이유는 무엇인가?
4. `shared` 대조군이 K=14,336에서도 평탄하다는 사실은 무엇을 반증하는가?
5. 333.8 GOPS를 production 성능으로 바로 인용하면 안 되는 이유는 무엇인가?
6. 초기 1.205배를 동일 바이너리 paired A/B로 다시 확인해야 하는 이유는 무엇인가?
