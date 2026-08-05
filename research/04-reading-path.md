# 단계별 학습 경로

각 단계마다 **"이 단계가 답하는 질문"** 과 **"우리 측정과의 연결"** 을 붙였다.
★ = 정독, ○ = 통독, △ = 필요할 때 참조

---

## Stage 0 — 왜 prefill과 decode는 다른 기계인가 (약 1주)

> **질문**: 우리가 쓴 "machine balance 11–14 FLOP/byte, 타일 T≈22–28" 논법은 어디서 온 표준 형태인가?

| | 논문 | 왜 |
|---|---|---|
| ★ | **Roofline: An Insightful Visual Performance Model** (Williams et al., CACM 2009) | 산술 강도 / 균형점 개념의 원전. 우리 논문의 phase diagram이 결국 roofline의 변형이다 |
| ★ | **Efficiently Scaling Transformer Inference** (Pope et al., MLSys 2023) [arXiv:2211.05102](https://arxiv.org/abs/2211.05102) | 추론 병렬화와 산술 강도를 처음으로 체계적으로 정리. TP/PP 선택 기준의 원형. **분산 설계 논의의 공통 언어** |
| ★ | **LLM Inference Unveiled: Survey and Roofline Model Insights** [arXiv:2402.16363](https://arxiv.org/html/2402.16363v4) | 위 두 개를 LLM 추론에 직접 적용한 형태. 우리가 §5에서 한 계산의 정석 버전 |
| ★ | **FlashAttention** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) → **FlashAttention-2** [arXiv:2307.08691](https://arxiv.org/pdf/2307.08691) | **우리가 방금 구현한 것의 원본 사고방식.** "attention은 compute가 아니라 IO 문제다"가 핵심 주장. tiling + online softmax |

> **연결**: 발견 #2(attention이 memory-bound)는 FlashAttention의 문제의식을 CPU/SBC로 옮긴 것이다.
> 단 GPU는 HBM↔SRAM, 우리는 DRAM↔L2다. **계층은 다르지만 논법은 같다** — 이 대응 관계를
> 논문 서론에서 명시하면 리뷰어가 바로 이해한다.

---

## Stage 1 — 서빙 시스템의 계보 (1–2주)

> **질문**: prefill을 어떻게 스케줄링해왔는가? 우리가 재발명하지 않으려면 뭘 알아야 하나?

| | 논문 | 왜 |
|---|---|---|
| ○ | **From Attention to Disaggregation: Tracing the Evolution of LLM Inference** [arXiv:2511.07422](https://arxiv.org/pdf/2511.07422) | **여기부터 시작.** 계보 전체 지도. 이걸 먼저 훑고 아래를 골라 읽는 게 효율적 |
| ★ | **Orca** (OSDI 2022) | continuous batching. 요청 단위가 아니라 iteration 단위로 스케줄링한다는 전환 |
| ★ | **vLLM / PagedAttention** (SOSP 2023) [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | KV 캐시를 페이지로 관리. **KV 캐시가 1급 자원이라는 관점**의 출발 |
| ★ | **SARATHI** [arXiv:2308.16369](https://arxiv.org/pdf/2308.16369) → **Sarathi-Serve** (OSDI 2024) | **chunked prefill의 원조.** 우리 청크 크기 논의의 직계 조상 |
| ○ | **DistServe** (OSDI 2024) / **Splitwise** (ISCA 2024) | prefill/decode 분리(P/D disaggregation). "두 단계는 다른 기계다"를 시스템으로 밀어붙인 사례 |
| △ | **Taming the Titans: A Survey of Efficient LLM Inference Serving** [arXiv:2504.19720](https://arxiv.org/pdf/2504.19720) | 사전처럼 참조 |

> **연결**: 우리 "청크 크기는 연산 효율이 아니라 통신/파이프라인 관점에서 정하면 된다"는 결론은
> Sarathi 계보 위에 SBC의 machine balance를 얹은 것이다. Sarathi를 안 읽고 쓰면 바로 걸린다.

---

## Stage 2 — 우리 환경: CPU / 엣지 (약 1주)

> **질문**: CPU 추론에서 이미 해결된 것은 무엇이고, distributed-llama는 왜 그걸 놓쳤나?

| | 대상 | 왜 |
|---|---|---|
| ★ | **llama.cpp / ggml 소스** — 특히 `ggml_compute_forward_flash_attn_ext`, `ggml_mul_mat` | **논문이 아니라 코드지만 이 단계에서 가장 중요.** attention을 GEMM 2개로 표현하는 방식을 직접 봐야 발견 #2의 의미가 잡힌다 |
| ★ | **T-MAC** [arXiv:2407.00088](https://arxiv.org/pdf/2407.00088) | LUT 기반 저비트 CPU 커널. Pi에서 GPU급 성능. **CPU 커널 설계의 사고방식** |
| ★ | **Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving** [arXiv:2507.18454](https://arxiv.org/pdf/2507.18454) | **CPU에서 prefill/decode를 분리 컴파일.** 우리 문제의식과 가장 가까운 CPU 논문 |
| ○ | **Profiling LLM Inference on Apple Silicon: A Quantization Perspective** [arXiv:2508.08531](https://arxiv.org/pdf/2508.08531) | CPU/통합메모리에서의 roofline 실측 사례. 방법론 참고 |
| ○ | **Efficient Inference for Edge LLMs: A Survey** [TST 2025](https://www.sciopen.com/article/10.26599/TST.2025.9010166) | 엣지 쪽 지형도 |
| △ | **Cloud to Edge: Benchmarking LLM Inference on SBCs** [arXiv:2604.24785](https://arxiv.org/html/2604.24785) | SBC 벤치마크 수치 비교용 |

> **연결**: 발견 #1·#2가 "distributed-llama 구현 문제"라는 우리 판단의 근거가 여기 있다.
> **llama.cpp 코드를 실제로 읽어야** 그 주장을 논문에 쓸 수 있다.

---

## Stage 3 — 분산: SBC 클러스터 (1–2주)

> **질문**: 저사양 클러스터 분산 추론의 현재 SOTA는 어디까지 왔나? 우리 빈자리는 어디인가?

| | 논문 | 왜 |
|---|---|---|
| ★ | **Galaxy** (INFOCOM 2024) [arXiv:2405.17245](https://arxiv.org/pdf/2405.17245) | TP+SP 하이브리드 + 타일 단위 통신-연산 오버랩. **가장 직접적인 경쟁군이자 타겟 학회의 모범 답안** |
| ★ | **Prima.cpp** [arXiv:2504.08791](https://arxiv.org/abs/2504.08791) | piped-ring parallelism, 이종 홈 클러스터. **decode 목표**라는 점이 우리와의 차이 |
| ★ | **TPI-LLM** [arXiv:2410.00531](https://arxiv.org/pdf/2410.00531) | 메모리 제약 하 TP. sliding-window 메모리 관리 |
| ★ | **Context Parallelism for Scalable Million-Token Inference** [arXiv:2411.01783](https://arxiv.org/pdf/2411.01783) | pass-KV / pass-Q. **TCP 등 저대역폭에서도 확장됨을 보인 근거** — 우리 exact baseline의 설계도 |
| ○ | **Ring Attention** [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) | CP의 원형 |
| ○ | **EdgeShard**, **PipeEdge** | 이종 디바이스 레이어 분할(DP 기반). 플래너 설계 참고 |
| △ | **Dynamic Micro-Batch & Token-Budget Scheduling** [Sensors 2026](https://doi.org/10.3390/s26041101) | **아이디어 겹침 주의 대상.** related work에서 명시적 구분 필요 |

> **연결**: 우리 EXP-1의 H1(동기 구조가 병목)은 Galaxy가 타일 오버랩으로 푼 문제와 같은 뿌리다.
> 다만 Galaxy는 Jetson **GPU** 보드이고 우리는 CPU 코어 경합이 있다 — 그 차이가 기여점.

---

## Stage 4 — 우리 아이디어의 직계 조상 (약 2주)

> **질문**: "나눠서 계산하고 한 번에 합친다"는 이미 누가 어디까지 했나?

### 4-a. 병렬 컨텍스트 인코딩 (핵심)

| | 논문 | 왜 |
|---|---|---|
| ★★ | **Star Attention** (ICML 2025) [arXiv:2411.17116](https://arxiv.org/pdf/2411.17116) · [code](https://github.com/NVIDIA/Star-Attention) | **가장 중요.** 블록 로컬 인코딩(통신 0) + 쿼리 단계 단일 merge. 우리가 재설계할 대상 |
| ★ | **APE** (ICLR 2025) [arXiv:2502.05431](https://arxiv.org/pdf/2502.05431) | 병렬 인코딩 KV의 분포 정렬. 정확도 회복 기법의 표준 |
| ★ | **Pulsar Attention** [arXiv:2607.20457](https://arxiv.org/html/2607.20457) | anchor를 통계 요약으로 대체. **정면 충돌 위험 — 반드시 차이를 정리해 둘 것** |
| ○ | **Block-Attention**, **CacheBlend**, **EPIC** | RAG 프리픽스 재사용 계열. 같은 문제의 다른 각도 |

### 4-b. KV 생성 연산 자체를 줄이기

| | 논문 | 왜 |
|---|---|---|
| ★ | **SwiftKV** [arXiv:2410.03960](https://arxiv.org/pdf/2410.03960) | SingleInputKV — 상위 레이어 KV를 x_k에서 직접 산출. **depth 분해 아이디어의 원본** |
| ○ | **KVSharer** [arXiv:2410.18517](https://arxiv.org/pdf/2410.18517) / **CommonKV** [arXiv:2508.16134](https://arxiv.org/abs/2508.16134) | training-free 레이어 간 KV 공유. "어느 레이어를 근사해도 안전한가" 지표 |
| ○ | **LazyLLM** [OpenReview](https://openreview.net/forum?id=am5Z8dXoaV) / **Speculative Prefill** [arXiv:2502.02789](https://arxiv.org/pdf/2502.02789) | 토큰 중요도 기반 prefill 축소. PiPP(decode skip)와의 서사 연결 지점 |
| △ | **PrefillOnly** [arXiv:2505.07203](https://arxiv.org/html/2505.07203v1) | "prefill-only 워크로드는 마지막 레이어 KV만 필요" — lm_head 낭비 관찰과 연결 |

---

## Stage 5 — 논문을 "쓰기 위한" 방법론 (병행)

> **질문**: 원형을 제안하지 않으면서 알고리즘을 주장하는 논문은 어떻게 구성되는가?

| | 대상 | 왜 |
|---|---|---|
| ★ | **Alpa** (OSDI 2022) [arXiv:2201.12023](https://arxiv.org/abs/2201.12023) | **"플래너 자체가 기여"인 논문의 정본.** 우리 §3-B(체제 인지 분할 플래너)의 서술 모델 |
| ★ | Galaxy / EdgeShard의 **planning 섹션만** 다시 읽기 | 비용 모델을 어떻게 정식화하고 검증하는지 |
| ○ | **A Systematic Characterization of LLM Inference on GPUs** [arXiv:2512.01644](https://arxiv.org/pdf/2512.01644) | measurement study가 기여가 되는 형태의 예시 |

---

## 권장 진행 방식

1. **Stage 0을 건너뛰지 말 것.** 우리 논문의 핵심 논거(machine balance, phase diagram)가
   전부 여기 언어로 쓰인다. Pope et al.과 FlashAttention은 정독.
2. **Stage 2의 llama.cpp 코드 읽기는 논문 작성 전 필수.** 발견 #1·#2를
   "distributed-llama 구현 문제"라고 쓰려면 근거가 있어야 한다.
3. Stage 3·4는 **related work를 쓰면서** 병행하는 게 효율적이다. 한 번에 다 읽고 잊는 것보다
   각 논문마다 "우리와의 차이 한 문장"을 즉시 기록하는 편이 낫다.
4. Stage 4-a의 Star Attention은 **코드까지** 볼 것. 우리 EXP-2가 이걸 포팅하는 것이다.

## 지금 우리 진행 상황과의 매핑

| 우리 작업 | 대응 Stage |
|---|---|
| 발견 #1 dotprod (완료) | — (구현 이슈) |
| 발견 #2 attention 배치화 (검증 중) | Stage 0 (FlashAttention), Stage 2 (llama.cpp) |
| llama.cpp 대조 (미착수) | Stage 2 |
| EXP-1 비용 분해 스윕 | Stage 0 (roofline), Stage 3 (Galaxy) |
| EXP-2 Star Attention 포팅 | Stage 4-a |
| 플래너 설계 | Stage 5 (Alpa) |
