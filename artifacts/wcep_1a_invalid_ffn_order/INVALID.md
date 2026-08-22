# ⚠️ 무효 — FFN projection 순서 오류

offset self-test 는 `diff=0` 으로 통과했으나 **FFN 세 행렬의 파일 순서가 틀렸다.**

## 원인

실제 파일 순서 (`src/llm.cpp` loadLlmNetWeight):

    Q -> K -> V -> O -> W1 -> W2 -> W3
    W1 = Gate : 4096 -> 14336
    W2 = Down : 14336 -> 4096
    W3 = Up   : 4096 -> 14336

scanner 순서:

    Q, K, V, O, Gate, Up, Down     <- W2 를 Up 형상으로, W3 를 Down 형상으로 해석

## 왜 offset test 가 못 잡았나

세 FFN 행렬의 총 원소 수가 같다.

    4096 x 14336 == 14336 x 4096

따라서 byte offset 은 정확하고 최종 `diff=0` 도 통과한다. 그러나 **output-row 경계와
4x32 tile 구성이 잘못된다.** `tile_hist` 와 `cross_out` 분석이 특히 영향을 받고,
FFN 이 전체 tile 의 대부분을 차지하므로 가중평균 결과를 쓸 수 없다.

**교훈**: byte 크기만 맞추는 self-test 로는 부족하다. **이름·shape 를 loader 와 함께
대조**해야 한다.

## 함께 남은 결함

`proj_hist` null 이 아직 exact 가 아니다 — row 4칸·K block 8칸 간격 표본에 rounding 을
더해 65,536 풀을 근사 생성한다. `sampled_proj_hist` 로 이름을 낮추거나 16-counter
전수 집계로 고쳐야 한다.
