# generality — SharedPack 일반성 스윕

`S_real {447, 1789, 7212} × B {16, 32} × threads {1, 4}`, N=8 wave on.
조합마다 `A,B,B,A` 를 **연달아** 돌려 같은 세션 안에서만 비율을 만든다.

harness: `prefill_bench/generality.sh`

## 파일

```
run.tsv       1차 자료. 완료된 회차만 한 줄씩 추가된다
raw/*.log     각 회차의 전체 로그
raw/*.bin     프리필 종료 로짓 f32 dump (.gitignore 대상)
```

## 0바이트 로그는 실패가 아니다

셸 리다이렉트로 파일을 먼저 만들고 실행이 끝난 뒤 채워진다.
따라서 **진행 중인 회차의 로그는 0바이트다.**

```
0바이트 + run.tsv 에 해당 tag 없음   → 진행 중
0바이트 + run.tsv 에 tag 있음        → 실제 실패. 조사 대상
```

`run.tsv` 에 줄이 있는데 `prefillMs` 가 `FAIL` 이면 timeout 또는 비정상 종료다.

## 진행률을 문서 본문에 적지 않는다

`run.tsv` 의 줄 수가 1차 자료다. 본문에 "27/48" 같이 적으면 곧 낡는다.

```
완료 회차 수:  tail -n +2 run.tsv | wc -l
전체 회차 수:  3 x 2 x 2 x 4 = 48  (+ 웜업 1, 폐기)
```

## 로짓 해시

같은 `S_real` 안에서 BASE 와 SP 의 md5 가 같아야 한다.
길이가 다르면 해시도 다르다 — 서로 비교하지 않는다.

```
S_real=447    c52e50e37e65
S_real=1789   acd54508ab4f
```
