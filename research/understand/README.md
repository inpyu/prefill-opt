# understand/ — DerivePP를 읽기 위한 보조 자료

`research/`의 번호 문서는 시간 순 연구 기록이다. 실험 당시의 가설, 폐기된 설계,
현재와 다른 수치가 함께 남아 있다. 이 폴더는 처음 읽는 사람이 **현재 논문 흐름**을 따라갈
수 있게 만드는 보조 자료다.

새로운 실측의 정본은 [16-derivepp.md](../16-derivepp.md)와 그 artifact에 둔다.
이 폴더는 그 수치의 의미, 수식, 읽는 순서를 설명한다.

## 권장 읽기 순서

```text
00-overview
  → 04-concepts
  → 06-math-pipeline
  → 07-math-measurement
  → 11-axis-cert
  → 16-derivepp
```

| 파일 | 역할 |
|---|---|
| [00-overview.md](00-overview.md) | **여기서 시작** — 현재 문제, 논문 메시지, 최종 개발 계획 |
| [01-code-map.md](01-code-map.md) | 개념과 실제 source/function의 대응 |
| [02-glossary-extra.md](02-glossary-extra.md) | 기호·이름 충돌·추가 용어 |
| [03-glossary-full.md](03-glossary-full.md) | 프로젝트에서 쓴 상세 용어와 측정 맥락 |
| [04-concepts.md](04-concepts.md) | Transformer, KV cache, PP/CP/TP의 원리 |
| [05-math-attention.md](05-math-attention.md) | attention, RoPE, KV cache, 정확성 유도 |
| [06-math-pipeline.md](06-math-pipeline.md) | `M`, `N`, `B`, bubble, pipeline 시간 유도 |
| [07-math-measurement.md](07-math-measurement.md) | warm-up, paired A/B, 정확성·측정 규율 |
| [08-math-executor-calibration.md](08-math-executor-calibration.md) | kernel sum과 production executor 비용의 차이 |
| [09-math-planner.md](09-math-planner.md) | completion-vector recurrence와 dominance pruning |
| [10-calibration-history.md](10-calibration-history.md) | calibration 설계가 바뀐 역사와 제거한 가정 |
| [11-axis-cert.md](11-axis-cert.md) | **현재 핵심** — paired residual 기반 axis activation과 capacity gate |

## 문서 사용 규칙

1. 이 폴더의 문서는 원자료를 대체하지 않는다. 새 수치는 `research/16-derivepp.md`와
   run artifact에 먼저 기록한다.
2. historical 문서의 오래된 설계나 측정값을 인용할 때는 00과 16의 현재 판정으로
   상태를 확인한다.
3. 새로운 planner 기능은 먼저 “어떤 축을 활성화하는가”와 “어떤 paired residual 또는
   구조적 upper bound가 그 결정을 지지하는가”를 적고, 그 다음 구현한다.
