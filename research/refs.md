# 선행연구 정리

## A. 직접 경쟁군 — 엣지/홈 클러스터 분산 추론

| 논문 | 요지 | 우리와의 간극 |
|---|---|---|
| **Galaxy** (INFOCOM'24) [arXiv:2405.17245](https://arxiv.org/pdf/2405.17245) | TP+SP 하이브리드 + 타일 단위 통신-연산 오버랩, Jetson Nano, 최대 2.5× | GPU 보드 대상, BERT/GPT2-L급 소형 모델, TTFT 지표 아님 |
| **Prima.cpp** [arXiv:2504.08791](https://arxiv.org/abs/2504.08791) | piped-ring parallelism + mmap prefetch, 70B 홈 클러스터 | 목표가 decode token latency, prefill 아님 |
| **TPI-LLM** [arXiv:2410.00531](https://arxiv.org/pdf/2410.00531) | sliding-window 메모리 관리 + 링크 최적화 | 메모리 제약 해소가 주목적 |
| EdgeShard / PipeEdge | 이종 디바이스 인지 레이어 분할(DP) | 정적 분할, 청크 단위 prefill 파이프라이닝 없음 |
| **Dynamic Micro-Batch & Token-Budget Scheduling** (Sensors'26) [doi](https://doi.org/10.3390/s26041101) | 4노드 PP에서 token budget/micro-batch 동적 조정, 버블 55%↓ | **아이디어 골격 겹침 주의.** GPU 기준, CPU-only 아님 |

## B. 병렬 컨텍스트 인코딩 — 우리가 재설계할 대상

| 논문 | 요지 | 전제(우리 체제에서 깨지는 것) |
|---|---|---|
| **Star Attention** (ICML'25) [arXiv:2411.17116](https://arxiv.org/pdf/2411.17116) · [code](https://github.com/NVIDIA/Star-Attention) | 2단계: 블록별 로컬 어텐션(통신 0, anchor block 접두) → 쿼리 단계 global attention 단일 merge. 최대 11×, 정확도 97~100% | **S=128K 전제.** 호스트당 여러 블록 → anchor amortize. SBC(S≤4K, 노드당 블록 1개)에서는 amortize 불가 → 2× 연산 |
| **APE** (ICLR'25) [arXiv:2502.05431](https://arxiv.org/pdf/2502.05431) | 병렬 인코딩 KV의 분포 정렬(shared prefix, attention temperature, scaling), 128K에서 28× | 동일하게 초장문·RAG 전제 |
| **Pulsar Attention** [arXiv:2607.20457](https://arxiv.org/html/2607.20457) | anchor를 통계적 요약으로 대체 | **충돌 주의.** anchor를 *정확도* 각도로 다루므로, 우리는 *통신 vs 재계산 비용* 각도로만 접근할 것 |

## C. Prefill / TTFT 기법

- **SwiftKV** [arXiv:2410.03960](https://arxiv.org/pdf/2410.03960) — SingleInputKV: 상위 레이어 KV를 x_k에서 직접 산출, prefill 연산 30~50%↓. **distillation 필요**(우리는 calibration-only 변형을 검토)
- **Context Parallelism for Million-Token Inference** [arXiv:2411.01783](https://arxiv.org/pdf/2411.01783) — pass-KV / pass-Q ring 변형, **TCP 환경에서도 확장성 유지**. exact baseline 구현 근거
- **SGLang Chunked Pipeline Parallelism** [LMSYS 2026-01](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/) — 긴 프롬프트 TTFT 81%↓
- **SARATHI** [arXiv:2308.16369](https://arxiv.org/pdf/2308.16369) — chunked prefill 원조
- **Cronus** [arXiv:2509.17357](https://arxiv.org/abs/2509.17357) / **Hetis** [arXiv:2509.08309](https://arxiv.org/html/2509.08309) — 이종 GPU 클러스터 prefill 분산
- **Speculative Prefill** [arXiv:2502.02789](https://arxiv.org/pdf/2502.02789) / **LazyLLM** [OpenReview](https://openreview.net/forum?id=am5Z8dXoaV) — 토큰 중요도 기반 prefill 축소
- **PrefillOnly** [arXiv:2505.07203](https://arxiv.org/html/2505.07203v1) — prefill-only 워크로드에서는 마지막 레이어 KV만 필요

## D. 통신 / KV 압축

- **Flash Communication** [arXiv:2412.04964](https://arxiv.org/html/2412.04964v1) — low-bit all-reduce. *prefill all-reduce가 2bsh, decode가 2bh* 라는 수치 인용용
- **Communication Compression for TP Inference** [arXiv:2411.09510](https://arxiv.org/html/2411.09510v3)
- **KVSharer** [arXiv:2410.18517](https://arxiv.org/pdf/2410.18517) / **CommonKV** [arXiv:2508.16134](https://arxiv.org/abs/2508.16134) — training-free 레이어 간 KV 공유. 어느 레이어를 근사해도 안전한지 선택하는 지표로 활용

## E. CPU / SBC 실행 기반

- **T-MAC** [arXiv:2407.00088](https://arxiv.org/pdf/2407.00088) — LUT 기반 저비트 CPU 커널
- **Sandwich** [arXiv:2507.18454](https://arxiv.org/pdf/2507.18454) — CPU 서빙 prefill/decode 컴파일 분리, NUMA 인지 TP
- **분산 prompt caching on Pi** [arXiv:2602.22812](https://arxiv.org/html/2602.22812)
- **SBC LLM 벤치마크** [arXiv:2604.24785](https://arxiv.org/html/2604.24785)
- **distributed-llama** [README](https://github.com/b4rtaz/distributed-llama/blob/main/README.md) — 동기화 오버헤드 실측치 (Pi5 4대 8B: I 267ms / T 62ms)

---

## 인용 시 주의

1. **"GPU는 재계산이 싸고 SBC는 통신이 싸다"는 단순 반전 주장은 성립하지 않는다.**
   재계산 vs 브로드캐스트 비용비는 Pi5+1GbE ≈ 200, A100+NVLink ≈ 110 으로 **양쪽 다 브로드캐스트가 유리**하다.
   Star Attention이 재계산을 택한 이유는 비용이 아니라 **의존성 제거**(완전 독립 실행)다.
   → 살아남는 논거는 **"anchor amortization 전제가 short-prompt/few-block 체제에서 무너진다"** 쪽이다.

2. Sensors'26 논문(§A 마지막)과 Pulsar Attention(§B)은 각각 스케줄링·anchor 축에서 겹친다.
   related work에서 명시적으로 구분해 두지 않으면 novelty 공격을 받는다.
