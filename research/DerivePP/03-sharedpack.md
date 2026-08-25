# 03. SharedPack-SDOT — 노드 내 연산 계층

파이프라인이 서고 나서 **"스케줄링 알고리즘이 아니라 스케줄링 위에 올릴 최적화된
CPU 연산"** 으로 우선순위를 옮겼다. 그 결과물이다.

## 3.1 문제 — 커널이 같은 일을 4번 한다

Q4_0 repack GEMM 은 이렇게 돈다.

```
가중치   block_q4_0x4   (미리 재배치돼 있음)
활성화   block_q8_0  →  block_q8_0x4 로 재배치 필요
연산     ggml_gemm_q4_0_4x4_q8_0   (ARM SDOT)
```

문제는 **스레딩 구조**다. 스레드는 **출력 열**로 나뉜다. 그런데 모든 스레드가
**배치 전체의 활성화**를 필요로 한다. executor 는 op 경계에서만 동기화하고
op 내부 배리어가 없다. 그래서 **각 스레드가 자기 스크래치에 전체를 중복 변환한다.**

```
스레드 0:  활성화 전체 → 자기 packed copy → 자기 output column
스레드 1:  활성화 전체 → 자기 packed copy → 자기 output column
스레드 2:  ...
스레드 3:  ...
```

## 3.2 진단 — 비용은 pack 연산이 아니라 cache footprint 다

처음엔 `K` 가 길어서(Down, K=14336) 느린 줄 알았다. **K-panelization(KP-SDOT)** 을
설계했는데, dup/shared 대조를 해보니 원인이 달랐다.

| 스레드 수 | duplicate vs shared |
|---|---|
| 1~3 | 1.00~1.03× (차이 없음) |
| **4** | **1.68×** |

| 경로 | panel 68~238 KiB 구간 |
|---|---|
| shared | **327~334 GOPS 로 평탄** |
| duplicate | **318 → 199 GOPS 로 붕괴** |

**pack 자체의 연산량은 전체의 0.30% 에 불과하다.** 문제는 4개의 독립 cache
footprint 가 생겨 4스레드 확장이 무너지는 것이다.
`K` 는 교락 변수였다 — KP-SDOT 은 없는 문제를 푸는 것이었으므로 **중단했다.**

## 3.3 해법 — pack 을 executor op 로 승격

pack 을 별도 op(`OP_PACK_Q80X4`)로 분리한다. 그러면 **executor 의 op 경계가
barrier 를 제공**하므로, 네 스레드가 서로 다른 batch-row group 을 **하나의 버퍼**에
병렬로 쓸 수 있다.

```
block_pack_yq   →  matmul_q / matmul_k / matmul_v        (K = dim)
block_pack_yq2  →  matmul_w1 / matmul_w3   (Gate/Up)     (K = dim)
block_pack_dq   →  matmul_w2               (Down)        (K = hiddenDim)
```

`NnMatmulOpConfig.prepackedBufferIndex` 로 연결하고, 없으면(`NN_NO_PREPACK`)
기존 per-thread 경로로 fallback 한다.

packed 버퍼는 **원본 Q80 버퍼와 같은 shape** 로 선언한다. `block_q8_0x4` 는
`NnBlockQ80` 의 재배열이라 바이트 수가 정확히 같기 때문이다.

```
nBatches × kBlocks × 34  ==  (nBatches/4) × kBlocks × sizeof(block_q8_0x4)
```

덕분에 **버퍼 폭이 곧 K** 가 되어 폭을 직접 계산하다 틀릴 여지가 사라진다.

## 3.4 효과

**N=8, S=447, wave on. 앵커 설계(웜업 폐기 + 정/역 교차 + 기하평균 쌍 비율).**

| 세션 | 바이너리 | 쌍 비율 | R |
|---|---|---|---|
| 1 | `0c3accb8` | 1.4355 1.4654 1.4411 1.4410 | 1.4457 |
| 2 | `a6c22236` | 1.4481 1.4107 1.4426 1.4033 | 1.4260 |

**통합 8쌍 R = 1.436, 범위 1.403~1.465, log-SD ±1.4%**

### 수치의 역할 고정

| 수치 | 어디에 쓰는가 |
|---|---|
| **1.436×** | **논문 headline.** 통합 8쌍 기하평균 |
| 1.4457× / 1.4260× | 재현 세션별 결과 |
| **1.426×** | **성분 분해와 recurrence 의 대상.** 세션 2 |

> 아래 성분 표는 stage timing 을 보존한 **두 번째 세션(R=1.426×)** 의 값이다.
> 통합 headline 1.436× 와 다른 것은 오류가 아니라 대상이 다르기 때문이다.

관측된 성분 변화 (세션 2):

| 성분 | BASE | SP | 관측 비율 |
|---|---|---|---|
| prefill 전체 | 7,612 | 5,338 | **1.426×** |
| syncWait | 3,725 | 2,105 | 1.770× |
| non-wait residual | 3,887 | 3,233 | 1.202× |

**정확성: 로짓이 BASE 와 비트 단위로 동일**하고, packed 표현 대조 2,688회 중
불일치 0 이다.

## 3.5 이 표를 읽을 때의 함정

**`syncWait` 를 별도 이득으로 더하면 이중 계산이다.**
`syncWait` 는 독립 비용이 아니라 **앞 스테이지 완료시각에서 파생되는 결과**다
([02](02-wave-pipeline.md) 의 재귀). 앞 스테이지의 `C_{k,j}` 가 줄면 `F_{k−1,j}` 가
당겨지고 그 결과로 뒤 스테이지 대기가 줄어든다.

그리고 다음은 엄밀하지 않으므로 **쓰지 않는다.**

> ~~스테이지가 빨라져 wave overlap 이 좋아지고 bubble 이 줄었다~~

스케줄과 마이크로배치 수가 그대로면 bubble 의 **슬롯 수나 비율이 자동으로 준 것이
아니다.** 줄어든 것은 각 슬롯의 길이일 수 있고, 그것은 별개의 주장이다.

**현재 안전한 표현:**

> SharedPack 은 stage service-time 분포와 critical path 를 변화시켰으며,
> 그 결과 관측된 `syncWait` 가 3,725 → 2,105 ms 로 감소했다.
> N=1 operator 개선 1.205× 보다 큰 N=8 TTFT 개선 1.426× 가 관측됐지만,
> **그 원인은 아직 분해되지 않았다.**

## 3.6 novelty 위치 — 솔직하게

**"pack once, reuse" 는 선행 기술과 구별되지 않는다.**

| 선행 | 이미 하는 것 |
|---|---|
| FBGEMM | packing 을 GEMM 의 명시적 구성요소로 분리, packed 행렬 재사용 |
| Intel MKL packed GEMM API | 동일 입력을 여러 GEMM 호출에서 재사용 |
| llama.cpp `repack.cpp` | Q8_0 4행 interleave 변환 코드가 이미 존재 |

그래서 **독립 알고리즘 기여로 주장하지 않는다.** 현재 위치는
**DerivePP 내부의 연산 기여**다.

독립 기여로 키우려면 "언제·어디서·누가 소유하고 언제 해제하는가" 를 **자동으로
결정**하는 데까지 가야 한다 → `research/21` 의 CTGHP 설계.

**bit-identical 정확성도 novelty 가 아니라 요구사항으로 취급한다.**
*"다른 엣지 가속은 정확도를 희생한다"* 는 비교는 근거가 없어 쓰지 않는다.

## 3.7 개발 중 가장 값진 사고

구현 도중 `size2D(floatType, y, x)` 의 축을 두 곳에서 반대로 읽었다.

```
1) packQ80x4Forward 가 K 대신 행 수(32)를 읽어  kBlocks = 1
2) d_pack 폭을 w2Slice.d(=4096, 출력)로 잡아 Down 의 K(14336)보다 3.5배 작음
```

그런데 **일부 구성에서 E2E 로짓 해시가 BASE 와 비트 단위로 일치해 통과했다.**

| 구성 | 깨진 빌드의 결과 |
|---|---|
| N=8 **wave on** | 해시가 달랐다 — 잡혔다 |
| **N=1** | BASE 와 동일한 해시 — **통과** |
| **N=8 wave off** | BASE 와 동일한 해시 — **통과** |

> **일부 구성에서의 최종 해시 일치는 중간 커널 정확성을 보장하지 않는다.**
> (왜 그 두 구성에서 통과했는지는 규명하지 못했고, 수정 후에는 무의미해졌다.)

이 반증 사례가 [07-measurement.md](07-measurement.md) 의 검증 계층을 만들었다.
