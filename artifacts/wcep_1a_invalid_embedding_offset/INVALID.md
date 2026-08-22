# ⚠️ 무효 — embedding 오프셋 오류

이 디렉터리의 결과는 **사용하면 안 된다.** 기각 판정도 철회됐다.

## 원인

scanner 가 embedding 을 Q4_0 으로 가정해 건너뛰었으나 실제로는 **F32** 다
(`src/llm.cpp`: `tokenEmbeddingSize = size2D(F_32, vocabSize, dim)`).

    scanner 가 건너뛴 크기      295,501,824 B
    실제 F32 embedding       2,101,346,304 B
    오프셋 오차              1,805,844,480 B = 1.68 GiB

따라서 projection weight 가 아니라 **F32 embedding 중간부터** 데이터를 Q4 block 으로
해석했다.

## 결과에서 보이는 증거

    layer 0~13    reduction ≈ 64.8%
    layer 14      reduction ≈ 62.9%
    layer 15~31   reduction ≈ 46.7%

오프셋 차이가 약 14.7 layer 분량과 일치한다 — layer 14→15 의 급변이 계산된 오차와
정확히 대응한다.

## 함께 발견된 결함

1. **nibble layout** — `qs[j]` 의 low nibble 은 weight `j`, high 는 weight `j+16` 인데
   인접 쌍으로 읽었다. histogram 영향은 작으나 위치·cross-output 분석에 영향.
2. **proj_hist null** — projection 전체가 아니라 4,096개 표본 풀에서 복원추출했다.
   "histogram 정확히 보존" 이라는 사전등록 정의와 다르다.
3. **통계량 혼용** — median(1.34 %p)과 bootstrap(mean 대상, [3.39,6.20])을 섞어 썼다.
4. **self-test 산출물 누락** — scanner_config.json, model_sha256.txt,
   template_coverage.tsv, bootstrap_ci.tsv, selftest.log 가 없다.
   offset self-test 가 있었다면 이번 오류를 잡았어야 한다.
