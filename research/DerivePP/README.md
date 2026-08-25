# DerivePP — 저속 Ethernet CPU 클러스터의 Prefill/TTFT 최적화

이 폴더는 **지금 무엇을 개발하고 있고, 왜 그것을 했으며, 어떤 효과가 있었는지**를
한 곳에 정리한 것이다. 각 문서는 `문제 → 왜 이렇게 풀었나 → 무엇을 만들었나 →
효과 → 한계` 순서로 쓴다.

원자료와 상세 유도는 `research/16` (본편), `research/18~21` (연산 계층)에 있다.
여기는 그 위에 얹은 **읽는 사람을 위한 층**이다.

---

## 한 줄 요약

> 라즈베리파이 5 여덟 대를 100 Mb 업링크가 낀 스위치로 묶어, 8 B 모델의
> **첫 토큰까지 걸리는 시간(TTFT)** 을 단일 노드 llama.cpp 대비 4.57~4.68× 줄였다.
> 기존 공개 분산 구현 두 종은 같은 하드웨어에서 오히려 **0.41~0.44×** 로 느려진다.

---

## 문서 목록

| 문서 | 내용 |
|---|---|
| [01-problem.md](01-problem.md) | 왜 이 문제인가 — 하드웨어 제약, TTFT 가 왜 어려운가, 기존 구현이 왜 실패하는가 |
| [02-wave-pipeline.md](02-wave-pipeline.md) | **핵심 엔진.** microbatch wave 파이프라인 — 통신을 계산 뒤로 숨긴다 |
| [03-tile-alignment.md](03-tile-alignment.md) | 마이크로배치 크기 선택과 타일 정렬 — 계단 비용 함수와 파이프라인 누적 |
| [04-axiscert.md](04-axiscert.md) | AxisCert — 효과 없는 최적화 축을 **근거를 갖고 잘라내는** 절차 |
| [05-sharedpack.md](05-sharedpack.md) | 연산 계층 SharedPack-SDOT — 스레드별 중복 packing 제거 |
| [06-rejected.md](06-rejected.md) | 기각한 아이디어들과 **왜 기각했는가** (이 프로젝트에서 가장 분량이 큰 부분) |
| [07-measurement.md](07-measurement.md) | 측정 방법론과 검증 계층 — 스스로를 속이지 않기 위한 장치 |
| [08-status.md](08-status.md) | 현재 상태, 진행 중인 작업, 남은 리스크 |

---

## 현재 성과 한눈에

| 항목 | 수치 | 근거 |
|---|---|---|
| 로컬 llama.cpp 대비 prefill (S=447) | **4.68×** | `research/16` §0.3 |
| 로컬 llama.cpp 대비 prefill (S=1789) | **4.57×** | 〃 |
| 자기 단일 노드 대비 (8노드) | **6.64×**, 효율 83% | 〃 |
| llama.cpp RPC (N=2) | 0.41× — **느려진다** | §7.16 |
| 공식 distributed-llama (N=4) | 0.44× — **느려진다** | §7.16 |
| SharedPack 추가 이득 (N=8 TTFT) | **1.436×**, 로짓 비트 동일 | `research/19` §9 |

> **주의.** 4.57× 와 1.436× 를 곱하지 않는다. 베이스라인·세션·조건이 다르다.
> 최종 시스템 대 llama.cpp 는 동일 조건에서 다시 재야 한다. → [08-status.md](08-status.md)
