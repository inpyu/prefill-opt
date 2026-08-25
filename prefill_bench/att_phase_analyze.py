#!/usr/bin/env python3
"""Phase 1 분석 — attention 내부 분해와 gate 판정.

research/DerivePP/05-attention-layer.md §5.4.

핵심: phase share 만으로 통과시키지 않는다. 측정된 share 와 현실적인
phase 별 가속률을 넣어 optimistic r_attention 을 직접 계산한다.

    1/r_attention = f_qk/r_qk + f_sm/r_sm + f_av/r_av + f_fin/r_fin + f_other

AV 가 35% 를 넘어도 attention 전체 1.5x 에 못 미칠 수 있다.
"""
import re, sys, glob, os, statistics as st

AD = 'artifacts/att_phase/raw'
PH = ['setup', 'qk', 'softmax', 'av', 'finalize']

# 현실적인 phase 별 가속률 상한. 근거 없이 낙관하지 않는다.
#   qk       현재 레이아웃에서 FMA 개선은 in-situ 에서 실패한 전례가 있다
#   softmax  exp 근사 변경 금지. scratch traversal 병합 정도
#   av       blocked-V + GQA register 공유의 목표치
#   finalize 정규화. 손댈 것이 거의 없다
R_OPT = {'setup': 1.00, 'qk': 1.20, 'softmax': 1.10, 'av': 1.70, 'finalize': 1.00}


def parse(path):
    """로그 하나에서 [ATT_PHASE] 블록을 읽는다."""
    txt = open(path, errors='ignore').read()
    i = txt.rfind('[ATT_PHASE]')
    if i < 0:
        return None
    blk = txt[i:i + 1200]
    out = {}
    for name in PH + ['other', 'thread_sum', 'thread_max']:
        m = re.search(r'^%s\s+([0-9.]+)' % re.escape(name), blk, re.M)
        if m:
            out[name] = float(m.group(1))
    m = re.search(r'회계\s+(\S+)\s+\(other ([-+0-9.]+)%\)', blk)
    if m:
        out['acct_ok'] = (m.group(1) == 'OK')
        out['other_pct'] = float(m.group(2))
    return out if 'thread_sum' in out else None


def overhead():
    print('=== A) 계측기 오버헤드 (사전등록: <=2%)')
    vals = {}
    for tag in ['ovh_1_off', 'ovh_2_on', 'ovh_3_off', 'ovh_4_on']:
        p = os.path.join(AD, tag + '.log')
        if not os.path.exists(p):
            print('  %s 없음' % tag); return
        t = open(p, errors='ignore').read()
        m = re.findall(r'^block_multihead_att\s+([0-9.]+)', t, re.M)
        vals[tag] = float(m[-1]) if m else None
    off = [vals['ovh_1_off'], vals['ovh_3_off']]
    on = [vals['ovh_2_on'], vals['ovh_4_on']]
    if None in off + on:
        print('  attMs 파싱 실패'); return
    go, gn = st.geometric_mean(off), st.geometric_mean(on)
    d = 100.0 * (gn - go) / go
    print('  off %.1f ms   on %.1f ms   차이 %+.2f%%  -> %s'
          % (go, gn, d, 'OK' if abs(d) <= 2.0 else '계측 입자도 축소 필요'))
    print()


def gate(f):
    """optimistic r_attention 을 계산한다."""
    inv = f.get('other_frac', 0.0)
    for k in PH:
        inv += f.get(k, 0.0) / R_OPT[k]
    return 1.0 / inv if inv > 0 else float('nan')


def main():
    overhead()
    print('=== B) pilot 분해와 gate')
    for S in ['s447', 's1789', 's7212']:
        runs = [parse(p) for p in sorted(glob.glob(os.path.join(AD, S + '_r*.log')))]
        runs = [r for r in runs if r]
        if not runs:
            print('%-7s 자료 없음' % S); continue
        tot = st.mean(r['thread_sum'] for r in runs)
        f = {}
        for k in PH:
            f[k] = st.mean(r.get(k, 0.0) for r in runs) / tot
        f['other_frac'] = st.mean(r.get('other', 0.0) for r in runs) / tot
        acct = all(r.get('acct_ok', False) for r in runs)
        tmax = st.mean(r['thread_max'] for r in runs)
        print('\n%s  (n=%d, thread_sum %.1f ms, thread_max %.1f ms, 회계 %s)'
              % (S, len(runs), tot, tmax, 'OK' if acct else '확인 필요'))
        for k in PH:
            print('  %-9s %6.2f%%' % (k, 100 * f[k]))
        print('  %-9s %6.2f%%' % ('other', 100 * f['other_frac']))
        r = gate(f)
        if r >= 1.5:
            verdict = 'RoleSplit 전체 진행'
        elif r >= 1.3:
            verdict = '20% E2E 어렵다. 보조 기여로만'
        else:
            verdict = 'V layout 중심 방향 중단'
        print('  optimistic r_attention = %.3fx  -> %s' % (r, verdict))
        print('  (가정 %s)' % ', '.join('%s=%.2f' % (k, v) for k, v in R_OPT.items()))


if __name__ == '__main__':
    main()
