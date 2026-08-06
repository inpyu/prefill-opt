# prefill-opt

SBC(Raspberry Pi 급) 커머디티 클러스터에서 **prefill / TTFT** 를 최적화하기 위한 연구용 포크.
[distributed-llama](https://github.com/b4rtaz/distributed-llama) 기반이며,
선행 프로젝트 [layer-skip-bypass](https://github.com/inpyu/layer-skip-bypass)(PiPP, decode 구간)와
달리 **prefill 구간**을 대상으로 한다.

전제 환경: GPU 없음 / CPU only / 1GbE 이더넷 / 프롬프트 0.5~4K 구간.

---

## 디렉터리

| 경로 | 내용 |
|---|---|
| `src/` | dllama 소스 (distributed-llama 포크) |
| `research/` | **연구 문서 — 여기부터 읽을 것** |
| `prefill_bench/` | prefill 실험 하네스 (스윕 · 파싱 · 마이크로벤치) |
| `scripts/` | 워커 노드 운영 및 분산 실행 스크립트 |
| `converter/` | HuggingFace / 토크나이저 변환 도구 (upstream) |
| `prompts_gen/` | 길이별 생성 프롬프트 (gitignore, `make_prompts.py` 로 재생성) |
| `bench_prefill/` | 실험 산출물 (gitignore) |

### research/ 읽는 순서

**[research/README.md](research/README.md) 부터 읽을 것** — 목표·문제·해결·선행연구를 한 문서에 정리했다.

| 문서 | 내용 |
|---|---|
| [README.md](research/README.md) | **프로젝트 개요 (여기부터)** |
| [00-RESEARCH-PLAN.md](research/00-RESEARCH-PLAN.md) | 문제 정의, 가설 H1~H5, 기여 구조, 타겟 학회 |
| [01-EXP1-cost-breakdown.md](research/01-EXP1-cost-breakdown.md) | prefill 비용 분해 실험 프로토콜 |
| [02-baseline-dotprod-fix.md](research/02-baseline-dotprod-fix.md) | 발견 #1 — `-mtune=native` 가 dotprod 를 무력화 |
| [03-attention-batching.md](research/03-attention-batching.md) | 발견 #2 — prefill attention 의 KV 재스트리밍 |
| [04-reading-path.md](research/04-reading-path.md) | 단계별 논문 학습 경로 |
| [05-kvcache-ttft-guide.md](research/05-kvcache-ttft-guide.md) | KV 캐시 생성 · TTFT 요소 · 분산 절차 가이드 |

---

## 빌드

```bash
make dllama
```

테스트 타깃: `make nn-cpu-test`, `nn-cpu-ops-test`, `nn-topology-test`, `nn-pipeline-test`

> **aarch64 주의**: `Makefile` 은 aarch64 에서 `-mcpu=native` 를 쓴다.
> `-march=native -mtune=native` 조합은 `__ARM_FEATURE_DOTPROD` 를 없애
> `llamafile_sgemm` 의 Q40×Q80 배치 경로를 통째로 비활성화하고,
> prefill 이 토큰별 matvec 폴백으로 떨어진다.
> → [research/02](research/02-baseline-dotprod-fix.md)

## 모델

모델·토크나이저(`.m` / `.t`)는 **커밋하지 않는다.** 심볼릭 링크로 관리한다.

```bash
ln -sf /path/to/dllama_model_llama3-8b_q40.m  dllama_model_llama3-8b_q40.m
ln -sf /path/to/dllama_tokenizer_llama3.t     dllama_tokenizer_llama3.t
```

## 기본 실행

```bash
# 워커
./dllama worker --port 9998 --nthreads 4

# root
./dllama inference \
  --model dllama_model_llama3-8b_q40.m \
  --tokenizer dllama_tokenizer_llama3.t \
  --buffer-float-type q80 --nthreads 4 --collective auto \
  --prompt "Hello world" --steps 64 \
  --workers 165.194.19.103:9998
```

## 실험 실행

```bash
cd prefill_bench
vim env.sh                                   # 노드 IP · 모델 경로 · 스윕 범위

python3 make_prompts.py --lengths 128,512,2048,8192 --out ../prompts_gen
bash redeploy_workers.sh                     # 워커에 현재 빌드 배포 (필수)
bash ../scripts/rpi_worker_run.sh            # worker-first: 워커를 root 보다 먼저 기동
bash run_breakdown.sh                        # S x N x BW 스윕
python3 parse_breakdown.py ../bench_prefill/<RUN_ID>
```

`bench_sgemm.cpp` 는 모델 로딩 없이 sgemm 배치 스케일링만 재는 마이크로벤치다.

### 계측 플래그

`--stage-timing 1` 을 주면 prefill 구간 분해가 출력된다.

| 항목 | 의미 |
|---|---|
| `syncWaitMs` | peer 대기 시간 (straggler) |
| `syncXferMs` | 실제 바이트 이동 시간 |
| `attnMs` | attention score/AV — O(S²) 항만 |
| `gemmMs` | projection + FFN + lm_head |
| `attnProjMs` / `ffnMs` / `normMs` / `lmHeadMs` | 세부 분해 |

### 운영 규칙

- **worker-first** — sub 노드 워커를 항상 root 보다 먼저 기동한다
- **재배포 필수** — 소스를 고쳤으면 `redeploy_workers.sh` 를 돌린다.
  구버전이 남은 워커는 straggler 가 되어 `wait_frac` 측정을 오염시킨다
- **동시 실행 금지** — 8B 모델을 두 프로세스가 동시에 올리면 OOM 이다
- run 사이 쿨다운(기본 60s)을 줄이지 말 것 — 열 조건 정렬용

---

## 현재 상태

수정 완료 (모두 무손실, 출력 byte-for-byte 동일 검증):

| | 내용 | 효과 |
|---|---|---|
| 발견 #1 | aarch64 dotprod 활성화 | GEMM 배치 32 에서 3.2~4.0× |
| 발견 #2 | prefill attention 배치화 (GQA 그룹화 + 쿼리 타일링) | attention 최대 8.5× |
| 누적 | llama3-8b_q40, Cortex-A76 4스레드, 단일 노드 | **496 → 176 ms/tok (2.8×)** |

둘 다 distributed-llama 구현 이슈이며 **논문 기여가 아니라 baseline 위생**이다.

## 알려진 사항

- `--pp-stage-skip*` 계열 플래그는 선행 프로젝트(PiPP, decode 구간 skip)의 것이다.
  **기본값 off 이며 prefill 측정에 영향을 주지 않는다.** 향후 prefill+decode 통합 시 재사용 가능.
- 토큰 임베딩이 F32 로 저장되어 모델 파일이 GGUF Q4_0 대비 약 1.8 GB 크다
  (`src/llm.cpp:196`). 연산 성능에는 영향이 없으나 노드당 메모리 여유를 잠식하며,
  각 노드가 전체 가중치를 들어야 하는 context-parallel 구성의 제약이 된다.
