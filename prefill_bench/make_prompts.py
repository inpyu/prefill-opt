#!/usr/bin/env python3
"""목표 토큰 길이별 프롬프트 생성.

토크나이저를 직접 붙이지 않고 단어 수로 근사한 뒤, dllama 로그의
`prefillTokens:` 실측값을 parse_breakdown.py 가 기록하므로 최종 분석은
근사가 아니라 실측 토큰 수 기준으로 이뤄진다.

  python3 make_prompts.py --lengths 128,512,2048,8192 --out ../prompts_gen
"""
import argparse
import pathlib
import random

# 자연어 산문. 반복 패턴은 어텐션 분포를 왜곡하므로 어휘를 섞는다.
CORPUS = """
Single board computers have become capable enough to run language models locally.
A cluster of such devices connected over commodity Ethernet can share the memory and
compute required for larger models, but the interconnect is orders of magnitude slower
than the fabrics used in datacenter deployments. Scheduling therefore matters more than
raw throughput. During the prefill phase the system must construct the key value cache
for every token at every layer before the first output token can be produced. This phase
is compute bound and its cost grows with prompt length. Communication patterns that are
acceptable when bandwidth is plentiful become dominant when it is not. Understanding where
the time actually goes is the first requirement for any principled optimization. Careful
measurement of computation, transfer, and synchronization wait separates designs that
scale from those that merely appear to. Thermal behaviour adds another dimension because
sustained load reduces clock frequency unevenly across nodes, producing stragglers that
collective operations cannot hide. The resulting variance accumulates over many layers.
""".split()


def build(target_tokens: int, seed: int) -> str:
    # 영어 산문 기준 대략 1 word ~ 1.3 tokens. 보수적으로 0.77 word/token.
    n_words = max(8, int(target_tokens * 0.77))
    rng = random.Random(seed)
    words = []
    while len(words) < n_words:
        start = rng.randrange(0, max(1, len(CORPUS) - 40))
        words.extend(CORPUS[start:start + 40])
    return " ".join(words[:n_words])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,512,2048,8192")
    ap.add_argument("--out", default="../prompts_gen")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for length in [int(x) for x in args.lengths.split(",")]:
        text = build(length, args.seed + length)
        path = out / f"prompt_{length}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"{path}  target={length} tok  words={len(text.split())}  chars={len(text)}")


if __name__ == "__main__":
    main()
