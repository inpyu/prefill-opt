# att_phase — attention 계측기 pilot

harness: `prefill_bench/att_phase_pilot.sh`, 분석: `prefill_bench/att_phase_analyze.py`
바이너리 `6852f155` (= `791fa52` + 미커밋 빌드 수정 1건). 단일 노드, 4스레드.

## 상태: A) 오버헤드 검증 도중 중단 — 2/4 회차

harness 는 A 단계에서 `DLLAMA_ATT_PHASE` 를 `0/1/0/1` 로 네 번 교차한다.
**`ovh_1_off`, `ovh_2_on` 만 끝났고 `ovh_3_off` 실행 중 SD 카드 장애로 멈췄다.**

```
ovh_1_off.log   25143 B   완주
ovh_2_on.log    20480 B   20 KB 정확히에서 잘림 (부분 write)
ovh_3_off.log       0 B   시작만 됨
ovh_4_on.log        - 없음
```

`generality/README.md` 의 규약대로면 0바이트는 "진행 중"이지만, 이 경우는
**실행 환경이 사라진 중단**이다. 재개하려면 A 단계를 처음부터 다시 돌린다.

## 지금까지 나온 것 (확정 아님, 표본 1쌍)

| tag | phase | prefillMs | attMs | 로짓 md5 |
|---|---:|---:|---:|---|
| ovh_1_off | 0 | 129745.41 | 42180.4 | 8b8178a50a97 |
| ovh_2_on  | 1 | 129783.31 | 42407.3 | 8b8178a50a97 |

- prefill 오버헤드 **+0.029%**, attention op 오버헤드 **+0.538%** — 게이트(2%) 대비 여유
- 계측 on/off 로짓 md5 동일 → 계측기가 수치를 바꾸지 않는다
- **단, 1쌍뿐이라 세션 간 변동과 구분되지 않는다.** 4회차를 채우기 전에는 게이트 통과로 쓰지 않는다

## phase 분해(QK/softmax/AV/finalize)는 아직 없다

`[ATT_PHASE]` 블록은 프로세스 종료 시 출력되는데 `ovh_2_on.log` 가 그 앞에서 잘렸다.
B) pilot 분해 단계는 시작조차 안 했다. **AV share 는 아직 모른다** —
RoleSplit go/no-go 판단의 입력이 없다는 뜻이다.

## 참고 — `ovh_1_off` 의 op 프로파일 (S_real=1789)

```
block_multihead_att   42180.4   30.10%      attnMs      37230.70
block_matmul_w3       23589.3   16.84%      gemmMs      86123.09
block_matmul_w1       23582.3   16.83%      attnProjMs  18011.65
block_matmul_w2       23300.8   16.63%      ffnMs       66659.74
block_matmul_wo        7783.0    5.55%      ttftMs     130582.99
```

S_real=1789 에서 attention core 가 op 시간의 30.1%. 05-attention-layer 가 인용한
S_real=7212 의 53% 와 같은 방향이다 (길수록 커진다).
