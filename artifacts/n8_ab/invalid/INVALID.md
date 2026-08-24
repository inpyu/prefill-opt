# ⚠️ 무효 — 실험 중 root 재빌드

wave-off A/B 도중 root 바이너리를 9300a29c → cb82104a 로 재빌드했다.
워커는 9300a29c 였으므로 이후 회차는 바이너리 불일치다.

유효했던 유일한 관측 (warm-up):
    N=8, wave off, BASE:  prefillMs 35369.54,  syncWaitMs 31395.32 (89%!),  md5 8b8178a50a97

두 가지가 시사적이다.
  1. wave off 의 N=8 은 syncWait 이 89% — 사실상 순차 실행이다
  2. 그 hash 가 **N=1 과 같다**(8b8178a50a97). N=8 wave on 의 c52e50e37e65 와 다르다
     -> N=1 vs N=8 의 logits 차이는 분산 자체가 아니라 **wave 모드**에서 온다

**교훈**: 실험이 도는 동안 빌드하지 않는다. flock 은 벤치 간 배타만 보장하고
빌드는 막지 않는다. (요약에 기록된 실수의 반복)
