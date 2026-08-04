#!/usr/bin/env python3
"""EXP-1 로그 → TSV.

run_breakdown.sh 가 남긴 bench_prefill/<RUN_ID>/logs/*.log 를 파싱해
prefill 구간의 시간/통신량을 한 행씩 뽑는다.

  python3 parse_breakdown.py ../bench_prefill/exp1_... -o .../breakdown.tsv

주의: 현재 dllama는 Sync를 wait/xfer로 나누지 않는다. sync_ms 는 둘의 합이며,
가설 H1(대기가 전송보다 큰가)을 판정하려면 nn-executor.cpp 계측 분리가 선행되어야 한다.
계측이 추가되면 SYNC_WAIT_RE / SYNC_XFER_RE 를 채우면 된다.
"""
import argparse
import pathlib
import re
import sys

# 🔷️ Eval  123 ms Sync   45 ms | Sent  100 kB Recv  100 kB | KV tx/rx 1/2 kB | Act tx/rx 3/4 kB | (128 tokens)
CHUNK_RE = re.compile(
    r"Eval\s*(\d+)\s*ms\s*Sync\s*(\d+)\s*ms"
    r".*?Sent\s*(\d+)\s*kB\s*Recv\s*(\d+)\s*kB"
    r".*?KV tx/rx\s*(\d+)\s*/\s*(\d+)\s*kB"
    r".*?Act tx/rx\s*(\d+)\s*/\s*(\d+)\s*kB"
    r".*?\((\d+)\s*(?:chunks,\s*(\d+)\s*)?tokens\)"
)

SCALARS = {
    "prefill_tokens": re.compile(r"^\s*prefillTokens:\s*(\d+)", re.M),
    "prefill_chunks": re.compile(r"^\s*prefillChunks:\s*(\d+)", re.M),
    "prefill_ms": re.compile(r"^\s*prefillMs:\s*([\d.]+)", re.M),
    "ttft_ms": re.compile(r"^\s*ttftMs:\s*([\d.]+)", re.M),
    "decode_ms": re.compile(r"^\s*decodeMs:\s*([\d.]+)", re.M),
    "total_ms": re.compile(r"^\s*totalMs:\s*([\d.]+)", re.M),
}

# 계측 분리 후 채울 자리 (01-EXP1 문서 §1 참조)
SYNC_WAIT_RE = re.compile(r"^\s*syncWaitMs:\s*([\d.]+)", re.M)
SYNC_XFER_RE = re.compile(r"^\s*syncXferMs:\s*([\d.]+)", re.M)
ATTN_RE = re.compile(r"^\s*attnMs:\s*([\d.]+)", re.M)
GEMM_RE = re.compile(r"^\s*gemmMs:\s*([\d.]+)", re.M)

OP_SCALARS = {
    "attn_proj_ms": re.compile(r"^\s*attnProjMs:\s*([\d.]+)", re.M),
    "ffn_ms": re.compile(r"^\s*ffnMs:\s*([\d.]+)", re.M),
    "norm_ms": re.compile(r"^\s*normMs:\s*([\d.]+)", re.M),
    "lm_head_ms": re.compile(r"^\s*lmHeadMs:\s*([\d.]+)", re.M),
    "other_op_ms": re.compile(r"^\s*otherMs:\s*([\d.]+)", re.M),
}

COLUMNS = [
    "run", "seqlen", "nodes", "bw_mbit", "rep",
    "prefill_tokens", "prefill_chunks",
    "prefill_ms", "ttft_ms", "decode_ms", "total_ms",
    # prefill 구간 chunk 로그 합산
    "eval_ms", "sync_ms",
    "sent_kb", "recv_kb", "kv_tx_kb", "kv_rx_kb", "act_tx_kb", "act_rx_kb",
    # 계측 분리 후 채워지는 값 (지금은 NA)
    "sync_wait_ms", "sync_xfer_ms", "attn_ms", "gemm_ms",
    "attn_proj_ms", "ffn_ms", "norm_ms", "lm_head_ms", "other_op_ms",
    # 파생
    "comm_frac", "wait_frac", "xfer_frac", "attn_frac", "attn_over_gemm", "status",
]

TAG_RE = re.compile(r"^s(\d+)_n(\d+)_bw(\d+)_r(\d+)$")


def scalar(pattern, text, cast=float):
    m = pattern.search(text)
    return cast(m.group(1)) if m else None


def parse_log(path: pathlib.Path) -> dict:
    text = path.read_text(errors="replace")
    row = {c: "NA" for c in COLUMNS}
    row["run"] = path.stem

    m = TAG_RE.match(path.stem)
    if m:
        row["seqlen"], row["nodes"], row["bw_mbit"], row["rep"] = m.groups()

    for key, pat in SCALARS.items():
        v = scalar(pat, text, float if key.endswith("_ms") else int)
        if v is not None:
            row[key] = v

    # inference 경로는 prefillTokens 대신 "Evaluation" 블록의 nTokens로 실측 토큰 수를 낸다.
    if row["prefill_tokens"] == "NA":
        m = re.search(r"^Evaluation\b.*?^\s*nTokens:\s*(\d+)", text, re.M | re.S)
        if m:
            row["prefill_tokens"] = int(m.group(1))

    # prefill 구간 chunk 로그만 합산 (decode 라인은 🔶 이므로 CHUNK_RE에 안 걸린다)
    acc = dict.fromkeys(
        ["eval_ms", "sync_ms", "sent_kb", "recv_kb",
         "kv_tx_kb", "kv_rx_kb", "act_tx_kb", "act_rx_kb"], 0)
    n_chunk = 0
    for line in text.splitlines():
        if "🔷" not in line:
            continue
        mm = CHUNK_RE.search(line)
        if not mm:
            continue
        g = mm.groups()
        acc["eval_ms"] += int(g[0])
        acc["sync_ms"] += int(g[1])
        acc["sent_kb"] += int(g[2])
        acc["recv_kb"] += int(g[3])
        acc["kv_tx_kb"] += int(g[4])
        acc["kv_rx_kb"] += int(g[5])
        acc["act_tx_kb"] += int(g[6])
        acc["act_rx_kb"] += int(g[7])
        n_chunk += 1
    if n_chunk:
        row.update(acc)

    extra = [("sync_wait_ms", SYNC_WAIT_RE), ("sync_xfer_ms", SYNC_XFER_RE),
             ("attn_ms", ATTN_RE), ("gemm_ms", GEMM_RE)]
    extra += list(OP_SCALARS.items())
    for key, pat in extra:
        v = scalar(pat, text)
        if v is not None:
            row[key] = v

    # 파생 지표
    # comm_frac: executor 타이머 기준 sync 비중 (eval vs sync)
    ev, sy = row["eval_ms"], row["sync_ms"]
    if isinstance(ev, int) and isinstance(sy, int) and (ev + sy) > 0:
        row["comm_frac"] = round(sy / (ev + sy), 4)
    # wait_frac: prefill 벽시계 대비 peer 대기 비중 — H1 판정에 직접 쓰는 값
    pm = row["prefill_ms"]
    wm = row["sync_wait_ms"]
    if isinstance(wm, float) and isinstance(pm, float) and pm > 0:
        row["wait_frac"] = round(wm / pm, 4)
    xm = row["sync_xfer_ms"]
    if isinstance(xm, float) and isinstance(pm, float) and pm > 0:
        row["xfer_frac"] = round(xm / pm, 4)
    # attn_frac / attn_over_gemm: H2 판정용. O(S^2) 항이 얼마나 지배적인가
    am, gm = row["attn_ms"], row["gemm_ms"]
    if isinstance(am, float) and isinstance(pm, float) and pm > 0:
        row["attn_frac"] = round(am / pm, 4)
    if isinstance(am, float) and isinstance(gm, float) and gm > 0:
        row["attn_over_gemm"] = round(am / gm, 4)

    row["status"] = "ok" if row["prefill_ms"] != "NA" else "incomplete"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    logs = sorted((run_dir / "logs").glob("*.log"))
    if not logs:
        sys.exit(f"로그 없음: {run_dir/'logs'}")

    rows = [parse_log(p) for p in logs]
    out = pathlib.Path(args.out) if args.out else run_dir / "breakdown.tsv"
    with out.open("w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"{out}  ({ok}/{len(rows)} ok)")

    # 요약: H1/H2 판정에 바로 쓰는 값
    # H1/H2 판정에 바로 쓰는 요약
    hdr = (f"\n{'run':<22}{'prefill_ms':>11}{'wait_frac':>10}{'xfer_frac':>10}"
           f"{'attn_frac':>10}{'attn/gemm':>10}")
    print(hdr)
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['run']:<22}{str(r['prefill_ms']):>11}{str(r['wait_frac']):>10}"
                  f"{str(r['xfer_frac']):>10}{str(r['attn_frac']):>10}"
                  f"{str(r['attn_over_gemm']):>10}")
    print("\n[H1] wait_frac > 0.20 → 성립(동기 구조가 병목). < 0.10 → 반증(연산 축으로).")
    print("[H2] attn/gemm < 0.30 → 성립(O(S^2)이 지배적이지 않음 → 분할 축 재선택이 메인).")


if __name__ == "__main__":
    main()
