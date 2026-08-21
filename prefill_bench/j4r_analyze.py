"""J4-R anchor 보정 분석 — 기하평균 paired ratio.

    R(q) = sqrt(T_P0,전 · T_P0,후) / T_q

선형 보간이 아니라 기하평균을 쓴다. 세션 드리프트는 곱셈적이고, 문서의 paired
residual 정의(`e = log R_meas − log R̂`)도 로그 공간이므로 그쪽이 일관된다.

오차 규모: 단일 실행 CV 3.5% 일 때, 후보 1회와 양쪽 anchor 를 함께 쓴 paired ratio
의 오차는 sqrt(1 + 1/2)*3.5% ≈ 4.3%. 따라서 판정선 10% 는 CV 의 3배가 아니라
**보정된 ratio 오차의 약 2.3배**다. 양방향 반복이 있어 판정 자체는 여전히 보수적.
"""
import sys, math, statistics as st

rows=[]
for ln in open(sys.argv[1]):
    f=ln.rstrip("\n").split("\t")
    if len(f)>=5 and f[0].strip().isdigit() and int(f[0])>=1:   # seq 0 = warm-up 제외
        rows.append((int(f[0]), f[1], float(f[2]), float(f[4])))

anch={s:m for s,n,m,_ in rows if n=="P0"}
if len(anch)<2: sys.exit("anchor 부족")
print(f"anchor P0 {len(anch)}회: " + ", ".join(f"{m/1000:.1f}" for m in anch.values()))
av=list(anch.values())
print(f"  CV {st.pstdev(av)/st.mean(av)*100:.2f}%  range {(max(av)/min(av)-1)*100:.1f}%")

def paired(seq):
    """앞뒤 anchor 의 기하평균."""
    lo=[s for s in anch if s<seq]; hi=[s for s in anch if s>seq]
    if lo and hi: return math.sqrt(anch[max(lo)]*anch[min(hi)])
    return anch[max(lo)] if lo else anch[min(hi)]

by={}
for s,n,m,d in rows:
    if n=="P0": continue
    by.setdefault(n,[]).append((paired(s)/m, d, s))

PRED={"Pdp":1.058,"Pmid":1.021}
print(f"\n{'plan':<6}{'예측R̂':>8}{'R(회차별)':>22}{'기하평균':>10}{'drain(ms)':>22}")
res={}
for n in sorted(by):
    v=[x for x,_,_ in by[n]]; dr=[d for _,d,_ in by[n]]
    g=math.exp(st.mean([math.log(x) for x in v]))
    res[n]=(g,v)
    print(f"{n:<6}{PRED.get(n,float('nan')):>8.3f}"
          f"{'  '+', '.join(f'{x:.3f}' for x in v):>22}{g:>10.3f}"
          f"{'  '+', '.join(f'{d:.0f}' for d in dr):>22}")
dm=[d for s,n,m,d in rows if n=="P0"]
print(f"{'P0':<6}{1.000:>8.3f}{'  (anchor)':>22}{1.000:>10.3f}"
      f"{'  '+', '.join(f'{d:.0f}' for d in dm):>22}")

print("\n=== 사전등록 판정 ===")
for n,(g,v) in res.items():
    lo,hi=min(v),max(v)
    if lo>1.10:   verdict="축 활성, planner 방향 맞음"
    elif hi<0.90: verdict="축 활성, planner 방향 **틀림** → 이 길이에서 planner 분할 사용 금지"
    elif 0.95<=lo and hi<=1.05: verdict="이 조건에서 축 비활성"
    else:         verdict="판정 불가 → 사전 고정한 추가 paired block 2개까지만 수행"
    print(f"  {n}: 회차 {lo:.3f}~{hi:.3f}  →  {verdict}")
    e=math.log(g)-math.log(PRED.get(n,1.0))
    print(f"       e = logR − logR̂ = {e:+.4f}  ({(math.exp(e)-1)*100:+.1f}%)")
