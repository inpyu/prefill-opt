#!/usr/bin/env bash
# 정확성 게이트. 어떤 분산 구성이든 단일 노드와 같은 결과를 내는지 확인한다.
#
#   bash verify.sh ref                      # 단일 노드 기준 출력 생성/갱신
#   bash verify.sh check <이름> [dllama 인자...]
#
#   bash verify.sh check tp2 --workers 165.194.19.113:9998
#   bash verify.sh check cp2 --cp-size 2 --workers 165.194.19.113:9998
#
# 왜 이 스크립트가 필요한가:
#   SP 축이 깨진 채로 성능만 측정해 왔다(SP2 43,620 / SP4 47,374ms). 정확성을 한 번도
#   확인하지 않아서, 그 수치가 무엇을 잰 것인지 알 수 없게 됐다(research/06 §4.10).
#   앞으로는 구성이 이 게이트를 통과한 뒤에만 성능을 잰다.
#
# 두 가지를 본다:
#   1) 생성 토큰 — 깨진 구성은 '!!!!' 같은 쓰레기를 낸다. 가장 먼저 걸린다.
#      단, 부동소수점 재결합으로 argmax 가 뒤집혀 뒤쪽 토큰이 갈릴 수 있으므로
#      "완전 일치"가 아니라 "앞 N개 일치 + 나머지도 정상 텍스트"로 판정한다.
#   2) 배치 perplexity — argmax 뒤집힘에 휘둘리지 않는 스칼라 지표.
#      기존 커널의 expf_neon 근사 바닥이 ~1e-3 이므로 0.5% 를 임계로 둔다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MODEL="${MODEL:-$REPO/dllama_model_llama3-8b_q40.m}"
TOKENIZER="${TOKENIZER:-$REPO/dllama_tokenizer_llama3.t}"
BIN="${BIN:-$REPO/dllama}"
REFDIR="${REFDIR:-$REPO/bench_prefill/verify_ref}"
# 생성 검사용 짧은 프롬프트(빠름)와 perplexity 용 긴 프롬프트(안정)를 분리한다.
#
# 처음에는 13토큰 하나로 둘 다 했더니 perplexity 가 불안정했다. TP4 는 447토큰에서
# 0.07% 로 검증된 구성인데(§4.9) 13토큰에서는 5.35% 가 나왔다 — 예측 12개를 평균낼
# 뿐이라 한 토큰의 로그확률만 흔들려도 지표가 크게 움직인다.
# 게이트가 정상 구성을 떨어뜨리면 게이트가 틀린 것이다.
PROMPT="${PROMPT:-The capital of France is Paris. The capital of Germany is}"
PPL_PROMPT_FILE="${PPL_PROMPT_FILE:-$REPO/prompts_gen/prompt_128.txt}"
STEPS="${STEPS:-24}"
MAXSEQ="${MAXSEQ:-256}"
PPLBATCH="${PPLBATCH:-32}"
PREFIX_MATCH="${PREFIX_MATCH:-8}"   # 앞 몇 개 토큰까지 일치를 요구할지
# perplexity 허용 편차 %.
#
# 정상으로 검증된 구성(TP2)으로 보정한 값이다. 임계를 0.5% 로 뒀더니 TP2 가
# 0.740% 로 탈락했다 — 게이트가 잘못된 것이지 TP2 가 틀린 게 아니다.
# §4.9 에서 TP4 를 447토큰으로 쟀을 때는 0.2% 였다. 이 게이트는 13토큰짜리 짧은
# 프롬프트를 쓰므로 평균 표본이 적어 편차가 크다.
# 기존 커널의 expf_neon 근사 바닥(~1e-3)에 더해 짧은 프롬프트 분산을 감안해 1.5% 로 둔다.
# 깨진 구성은 편차가 1,000,000% 규모라 이 임계로도 확실히 걸린다(SP2 실측).
PPL_TOL="${PPL_TOL:-1.0}"

mkdir -p "$REFDIR"

run_gen() {  # $1=out, 나머지=추가 인자
    local out="$1"; shift
    "$BIN" inference --model "$MODEL" --tokenizer "$TOKENIZER" \
        --nthreads 4 --buffer-float-type q80 --max-seq-len "$MAXSEQ" --steps "$STEPS" \
        --seed 42 --temperature 0 "$@" --prompt "$PROMPT" 2>&1 \
      | grep "^🔶 Pred" | sed 's/.*| //' > "$out"
}
run_ppl() {  # 나머지=추가 인자 -> perplexity 값 출력
    "$BIN" perplexity --model "$MODEL" --tokenizer "$TOKENIZER" \
        --nthreads 4 --buffer-float-type q80 --max-seq-len "$MAXSEQ" \
        --n-batches "$PPLBATCH" --ppl-batch "$PPLBATCH" "$@" \
        --prompt "$(cat "$PPL_PROMPT_FILE")" 2>&1 \
      | grep -E "^   perplexity:" | tail -1 | awk '{print $2}'
}

case "${1:?usage: verify.sh ref|check ...}" in
ref)
    echo "== 기준(단일 노드) 생성 =="
    run_gen "$REFDIR/tokens.txt"
    run_ppl > "$REFDIR/ppl.txt"
    echo "   토큰 $(wc -l < "$REFDIR/tokens.txt")개, perplexity $(cat "$REFDIR/ppl.txt")"
    echo "   본문: $(tr -d '\n' < "$REFDIR/tokens.txt" | head -c 100)"
    ;;
check)
    NAME="${2:?이름이 필요합니다}"; shift 2
    [ -f "$REFDIR/tokens.txt" ] || { echo "기준이 없습니다. 먼저 'verify.sh ref'"; exit 1; }
    echo "== $NAME =="
    run_gen "$REFDIR/$NAME.tokens.txt" "$@"
    PPL=$(run_ppl "$@")
    REFPPL=$(cat "$REFDIR/ppl.txt")

    fail=0
    # 1) 쓰레기 출력 검사: 서로 다른 토큰이 3종 미만이면 반복 쓰레기다
    uniq_n=$(sort -u "$REFDIR/$NAME.tokens.txt" | wc -l)
    if [ "$uniq_n" -lt 3 ]; then
        echo "   ✘ 쓰레기 출력 (고유 토큰 ${uniq_n}종): $(tr -d '\n' < "$REFDIR/$NAME.tokens.txt" | head -c 60)"
        fail=1
    fi
    # 2) 앞 N개 토큰 일치
    if ! diff -q <(head -"$PREFIX_MATCH" "$REFDIR/tokens.txt") \
                 <(head -"$PREFIX_MATCH" "$REFDIR/$NAME.tokens.txt") >/dev/null; then
        echo "   ✘ 앞 ${PREFIX_MATCH}토큰 불일치"
        echo "     기준: $(head -"$PREFIX_MATCH" "$REFDIR/tokens.txt" | tr -d '\n')"
        echo "     실제: $(head -"$PREFIX_MATCH" "$REFDIR/$NAME.tokens.txt" | tr -d '\n')"
        fail=1
    fi
    # 3) perplexity 편차
    if [ -n "$PPL" ]; then
        dev=$(awk -v a="$PPL" -v b="$REFPPL" 'BEGIN{printf "%.3f", (a>b?a-b:b-a)/b*100}')
        ok=$(awk -v d="$dev" -v t="$PPL_TOL" 'BEGIN{print (d<=t)?1:0}')
        [ "$ok" = "1" ] || { echo "   ✘ perplexity 편차 ${dev}% > ${PPL_TOL}% (기준 $REFPPL, 실제 $PPL)"; fail=1; }
        echo "   perplexity $PPL (기준 $REFPPL, 편차 ${dev}%)"
    else
        echo "   ✘ perplexity 측정 실패"; fail=1
    fi

    if [ "$fail" = "0" ]; then
        echo "   ✔ 통과 — 성능 측정을 진행해도 된다"
    else
        echo "   ✘ 실패 — 성능을 재지 말 것. 수치가 무엇을 잰 것인지 알 수 없다."
        exit 1
    fi
    ;;
*) echo "usage: verify.sh ref|check <name> [args...]"; exit 1;;
esac
