# 발견 #1: `-mtune=native` 가 prefill 배치 경로를 비활성화하고 있었다

**분류: baseline 정정 (논문 기여 아님)** — 이 구분이 중요하다. §4 참조.
발견일 2026-08-04, EXP-1 계측 직후 스모크 테스트에서 이상 신호를 추적한 결과.

---

## 1. 이상 신호

단일 노드 스모크 테스트(llama3-8b_q40, 110 토큰):

```
Prefill: 2.01 tok/s (496 ms/tok)
Decode:  1.90 tok/s (525 ms/tok)
```

Prefill은 토큰을 묶어 처리하므로 가중치 재사용으로 decode보다 몇 배 빨라야 정상인데
거의 같다. **배치 이득이 0**이라는 뜻.

## 2. 인과 사슬

1. `Makefile` 이 aarch64 에서 `-march=native -mtune=native` 를 함께 지정
2. aarch64 에서 `-mtune=native` 를 붙이면 `-march=native` 가 확장한 CPU 기능셋이
   기본 armv8-a 로 되돌아가 **`__ARM_FEATURE_DOTPROD` 가 사라진다**

   | 플래그 | `__ARM_FEATURE_DOTPROD` |
   |---|---|
   | `-march=native` | 정의됨 |
   | `-march=native -mtune=native` | **사라짐** |
   | `-mcpu=native` | 정의됨 |

   (`-march=native` 는 이 머신에서 `-mcpu=cortex-a76+crc+crypto` 로 해석된다)

3. `src/nn/llamafile/sgemm.cpp:978` 의 `#elif defined(__ARM_FEATURE_DOTPROD)` 분기가
   컴파일에서 제외 → `llamafile_sgemm` 이 Q40×Q80 조합에 대해 **항상 false 반환**
4. `matmulForward_Q80_Q40_F32` (`nn-cpu-ops.cpp:1206`) 가 폴백 루프로 진입 →
   **배치를 토큰 하나씩 matvec 으로 처리**
5. 결과: prefill 이 배치 이득을 전혀 못 받음

## 3. 정량 (마이크로벤치, Cortex-A76 4스레드, Q40 weight × Q80 activation)

`prefill_bench/bench_sgemm.cpp` — 모델 로딩 없이 sgemm 만 측정.

| shape | b=1 | b=8 | b=32 | b=128 |
|---|---|---|---|---|
| qkv/o (d=4096, k=4096) | 31.9 | 72.7 | **105.4** | 109.6 GFLOPS |
| ffn_w1 (d=14336, k=4096) | 33.7 | 90.5 | **106.3** | 109.5 GFLOPS |
| ffn_w2 (d=4096, k=14336) | 26.4 | 85.3 | **104.6** | 101.9 GFLOPS |

- 배치 32에서 **3.2 ~ 4.0×**
- **배치 32 이상은 ~110 GFLOPS 로 포화**

수정 전에는 위 표의 모든 칸이 `FALLBACK`(sgemm 미사용)이었다.

## 4. 수정

`Makefile`:

```make
UNAME_M := $(shell uname -m)
ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
    CXXFLAGS += -mcpu=native      # arch + tune 을 동시에 세팅
else
    CXXFLAGS += -march=native -mtune=native
endif
```

바이너리 크기 459,312 → 527,088 바이트 (dotprod 경로가 실제로 컴파일된 증거).

---

## 5. 연구에 갖는 의미

### (a) Baseline 신뢰성 — 반드시 먼저 처리

이 트리에서 지금까지 나온 **모든 prefill 수치가 3~4배 느린 상태**에서 얻어졌다.
논문에 "TP 대비 N배 개선"을 쓰려면 baseline 이 정상이어야 한다. 아니면
**"구현 미숙을 이긴 것"** 이라는 치명적 공격을 받는다.

- [ ] 워커 노드 전부 재빌드/재배포 필요 (`scripts/rpi_worker_run.sh` 의 `BIN_PATH`)
- [ ] PiPP 논문 수치도 이 영향을 받았는지 확인 필요 (decode 경로는 batch=1 이라
      sgemm 을 안 타므로 영향이 없을 가능성이 높지만, **확인 없이 단정하지 말 것**)

### (b) 비용 모델 갱신

`00-RESEARCH-PLAN.md` §1 에서 노드당 실효 연산을 30 GFLOPS 로 가정했으나
**실측 상한은 ~110 GFLOPS**. 3.7배 차이다.

→ **연산 시간이 3.7배 줄면 통신/연산 비율이 그만큼 나빠진다.**
   즉 통신 병목이 예상보다 심각하며, **H1(동기 구조가 병목)이 성립할 가능성이 커졌다.**
   수정 전 데이터로 H1 을 판정했다면 통신 비중을 크게 과소평가했을 것이다.

### (c) 설계 공간 단순화

배치 32에서 이미 포화 → **prefill 청크를 32 토큰보다 크게 잡아도 연산 효율 이득이 없다.**
따라서 청크 크기는 이제 순수하게 **통신/파이프라인 관점**에서만 결정하면 된다.
(기존 스모크 테스트는 27.5 tok/chunk 로 포화점 바로 아래에서 돌고 있었다.)

### (d) 논문에서의 취급

이건 **기여가 아니라 정정**이다. 혼동하면 안 된다.
- 평가 섹션의 baseline 은 **수정된 빌드**로 다시 측정한다
- 이 발견 자체는 "SBC 환경에서 커널 디스패치 검증의 중요성" 정도로 각주 처리하거나,
  artifact/reproducibility 섹션에 넣는다
- upstream(distributed-llama)에 리포트할 가치가 있다

---

## 6. 검증 상태

- [x] 마이크로벤치로 sgemm 배치 스케일링 확인
- [ ] end-to-end A/B (old vs new, 512/2048 토큰) — 진행 중
- [ ] 다중 노드에서 재확인
- [ ] 워커 노드 재배포
