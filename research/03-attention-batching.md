# 발견 #2: prefill attention이 decode용 커널을 재사용해 KV 캐시를 반복 스트리밍한다

**분류: baseline 정정 (논문 기여 아님)** — [02-baseline-dotprod-fix.md](02-baseline-dotprod-fix.md) 와 동일 취급.

---

## 1. 관측

발견 #1(dotprod)을 수정한 뒤 프롬프트 길이를 늘려가며 측정:

| 프롬프트(실측 토큰) | prefill | ms/tok | attn (O(S²)) | gemm | **attn 비중** |
|---|---|---|---|---|---|
| 110 | 56.9 s | 496 | 0.28 s | 56.3 s | 0.5 % |
| 447 | 76.0 s | 170 | 9.3 s | 65.5 s | 12.2 % |
| **1789** | **624 s** | 349 | **355.6 s** | 264.2 s | **57.0 %** |

교차점(attn = gemm)은 **S ≈ 1400~1500**, 즉 SBC 실사용 프롬프트 구간 안쪽이다.

> 초기에 110 토큰만 보고 "짧은 프롬프트에서 attention은 병목이 아니다(H2)"라고 판단했으나,
> **1789 토큰에서 역전된다.** H2는 S < 500 에서만 성립한다.

## 2. 원인

`multiheadAtt_F32` (`nn-cpu-ops.cpp`) 의 시그니처는 인자로 `pos` **하나**를 받는다.
즉 **쿼리 위치 1개를 전제로 설계된 decode용 커널**이고, prefill 은 이것을
`for (batchIndex)` 루프로 배치 크기만큼 반복 호출한다.

추가로 커널 내부는 쿼리 헤드 단위로 순회하는데, GQA에서 `kvMul = nHeads/nKvHeads = 4`
이므로 **같은 KV 헤드를 4번 반복해서 읽는다.**

결과적으로 KV 캐시가 **(batchSize × kvMul) 번** 스트리밍된다.

### 메모리 트래픽 추정 (S=1789, Llama-3-8B)

- 쿼리 토큰 1개당: 32 헤드 × headDim 128 × 4 B × (K+V) = 32.8 KB × pos
- 레이어당 Σ over pos ≈ 32,768 × S²/2 = **52 GB**
- × 32 레이어 = **약 1.7 TB**
- Pi5 실효 대역폭 ~8 GB/s → **약 210 초**

실측 355초와 같은 자릿수. **attention 은 compute-bound 가 아니라 memory-bandwidth bound 다.**
(GEMM 경로는 110 GFLOPS를 내는데 attention 은 유효 2.4 GFLOPS)

## 3. 수정

`multiheadAttBatch_F32` 신설. 연산 순서를 유지하므로 **수치 결과는 동일**해야 한다.

- **(A) GQA 그룹화** — 같은 KV 헤드를 공유하는 `kvMul` 개 쿼리 헤드를 함께 처리 → KV 읽기 `1/kvMul`
- **(B) 쿼리 타일링** — `t` 루프를 바깥으로 빼 배치 전체가 KV 한 번 읽기를 공유 → KV 읽기 `1/batchSize`

스레드 분할은 KV 그룹 수가 스레드 수 이상일 때만 그룹 단위(=A 적용),
아니면 기존처럼 헤드 단위(=B 만 적용). TP로 헤드가 쪼개진 구성에서 스레드가 놀지 않도록.

`batchSize == 1`(decode)은 재사용 대상이 없으므로 **기존 경로를 그대로 유지**한다.
→ PiPP(decode 기준)의 기존 측정치는 영향을 받지 않는다.

## 4. "이게 SBC 문제인가, distributed-llama 문제인가"

**distributed-llama 구현 문제다.** 두 발견 모두 그렇다.

| | 성격 | 근거 |
|---|---|---|
| 발견 #1 `-mtune=native` | 순수 빌드 설정 버그 | llama.cpp 는 ARM 에서 `-mcpu=native` 사용 |
| 발견 #2 attention | decode 전용 커널을 prefill 이 물려받음 | llama.cpp/ggml 은 attention 을 `mul_mat(K,Q) → softmax → mul_mat(V,att)` 즉 **진짜 GEMM 2개**로 표현. flash-attention 계열 op 도 보유 |

distributed-llama 는 애초에 "홈 클러스터에서 **토큰 생성(decode)** 을 텐서 병렬로 가속"이
목표였고 prefill 은 부산물이었다. **"SBC라서 안 맞는" 게 아니라 "이 엔진이 decode 최적화만
되어 있어서"** 다.

→ 따라서 **논문 기여가 아니라 baseline 위생**이다. 이 구분을 흐리면
   "구현 미숙을 이긴 것"이라는 공격을 받는다.

## 5. 그래도 남는 진짜 체제(regime) 논거

구현을 완벽히 고쳐도 사라지지 않는 부분:

Pi5 **machine balance**
- 연산 ~110 GFLOPS (int8 dotprod GEMM, 실측)
- 메모리 대역폭 ~8–10 GB/s
- → 균형점 **약 11–14 FLOP/byte**

완벽히 타일링된 attention이라도 KV 캐시를 쿼리 타일당 최소 1회는 읽어야 한다.
타일 크기 T일 때 산술 강도 ≈ **T/2 FLOP/byte** (fp32 KV). 균형점 도달 조건:

> **T ≈ 22 ~ 28 쿼리 토큰**

이 값이 앞서 측정한 **sgemm 포화점(배치 32)** 과 거의 같은 자리다. 즉

> **Pi5에서 prefill 청크는 32 이상이어야 GEMM과 attention이 동시에 memory-bound를 벗어난다.**

machine balance의 함수로 표현되므로 **다른 하드웨어로 전이 가능한 규칙**이고,
"체제 인지 플래너"의 첫 번째 구성요소가 될 수 있다.

또한 이 관점은 **블록 단위 로컬 어텐션(Star Attention 계열)이 SBC에서 GPU보다 이득이
큰 이유**를 설명한다 — GPU에서는 O(S²) *연산량* 을 줄이는 것이지만,
SBC에서는 O(S²) *메모리 트래픽* 을 줄이는 것이고 SBC는 대역폭이 훨씬 더 희소하다.

## 6. baseline 전략 (세 가지 모두 필요)

1. **distributed-llama 를 고쳐서 쓴다** — 발견 #1 ✔, 발견 #2 (검증 중)
2. **llama.cpp 를 외부 기준선으로 대조** — 같은 노드/모델/양자화로 `llama-bench` prompt
   processing tok/s 를 재서 우리 수정본이 근접함을 보인다.
   **이게 있어야 "네 baseline 이 원래 느린 거 아니냐"는 질문이 그래프 하나로 끝난다.**
   → 아직 미착수. 로컬에 llama.cpp 없음
3. 분산 실험 baseline 은 **수정본 + llama.cpp RPC 둘 다**

## 7. 검증 상태

- [x] 원인 규명 (메모리 트래픽 추정이 실측과 같은 자릿수)
- [x] `multiheadAttBatch_F32` 구현 및 빌드
- [ ] 정확성 검증 (동일 시드/온도에서 출력 토큰 일치) — 진행 중
- [ ] 성능 측정 (S=512, S=2048) — 진행 중
- [ ] llama.cpp 대조
- [ ] 워커 노드 재배포 (`prefill_bench/redeploy_workers.sh`)
