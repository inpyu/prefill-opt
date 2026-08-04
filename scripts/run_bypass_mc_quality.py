#!/usr/bin/env python3
import csv
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_QA = [
    ("qa01", "Choose the correct answer. Which term describes splitting model layers across multiple devices? A) Tensor parallelism B) Pipeline parallelism C) Data augmentation D) Dropout. Respond with only one letter:", "B"),
    ("qa02", "Choose the correct answer. In autoregressive decoding, what is generated at each step? A) A model checkpoint B) A next token C) A network switch packet D) A tokenizer file. Respond with only one letter:", "B"),
    ("qa03", "Choose the correct answer. What does TTFT mean in LLM serving? A) Time to first token B) Total training FLOPs C) Tensor transfer format D) Token table file. Respond with only one letter:", "A"),
    ("qa04", "Choose the correct answer. Which resource does quantization primarily reduce? A) Model memory footprint B) Ethernet cable length C) CPU core count D) Number of users. Respond with only one letter:", "A"),
    ("qa05", "Choose the correct answer. In pipeline parallelism, a stage usually contains what? A) A subset of model layers B) A random tokenizer C) A power meter D) A shell prompt. Respond with only one letter:", "A"),
    ("qa06", "Choose the correct answer. What does a fallback path provide in a conditional optimization? A) A route back to baseline behavior B) A new dataset C) A larger vocabulary D) A different OS. Respond with only one letter:", "A"),
    ("qa07", "Choose the correct answer. Which metric becomes lower when per-token latency improves? A) TPOT B) Model size C) Prompt length D) Vocabulary size. Respond with only one letter:", "A"),
    ("qa08", "Choose the correct answer. What is the main role of Ethernet in a multi-node PP setup? A) Transferring activations between stages B) Training the tokenizer C) Cooling the CPU D) Storing model weights. Respond with only one letter:", "A"),
    ("qa09", "Choose the correct answer. What does greedy decoding with temperature zero select? A) The highest-logit token B) A random token C) The longest token D) The first vocabulary entry. Respond with only one letter:", "A"),
    ("qa10", "Choose the correct answer. Why is root receive wait important in PP decode? A) It reflects waiting for the pipeline result B) It measures disk writes C) It counts prompts D) It sets the learning rate. Respond with only one letter:", "A"),
]


def env(name, default):
    return os.environ.get(name, default)


def parse_metric(log_text, key):
    match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9.]+)", log_text, re.MULTILINE)
    return match.group(1) if match else "NA"


def extract_choice(text):
    normalized = re.sub(r"\s+", " ", text.strip().upper())
    if not normalized:
        return "", "empty"

    choice_list_re = r"\bA\b.{0,16}\bB\b.{0,16}\bC\b.{0,24}\bD\b"

    answer_patterns = [
        r"\b(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER)\s*(?:IS|:)?\s*[\(\[]?\s*([ABCD])\b",
        r"\bTHE\s+ANSWER\s+IS\s*[\(\[]?\s*([ABCD])\b",
        r"\bOPTION\s+([ABCD])\b",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1), "answer_pattern"

    for raw_line in text.splitlines():
        line = raw_line.strip().upper()
        if not line:
            continue
        if re.search(choice_list_re, line):
            continue
        if len(re.findall(r"\b[ABCD][\).:]", line)) >= 2:
            continue
        match = re.match(r"^[\(\[]?\s*([ABCD])\s*[\)\].:]?\s*$", line)
        if match:
            return match.group(1), "single_line"
        match = re.match(r"^[\(\[]?\s*([ABCD])\s*[\)\].:]", line)
        if match:
            return match.group(1), "line_prefix"

    prefix = normalized[:32]
    option_echo_prefix = normalized[:160]
    if len(re.findall(r"\b[ABCD][\).:]", option_echo_prefix)) >= 2:
        return "", "option_echo"
    if not re.search(choice_list_re, prefix):
        match = re.match(r"^[\(\[]?\s*([ABCD])\s*(?:[\)\].:]|\b)", prefix)
        if match:
            return match.group(1), "prefix"

    return "", "no_choice"


def accept_rate(prefix):
    files = sorted(prefix.parent.glob(prefix.name + ".node*.tsv"))
    total = 0
    accepted = 0
    for file in files:
        with file.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total += 1
                if row.get("was_skip") in ("1", "true", "TRUE"):
                    accepted += 1
    if total == 0:
        return "NA"
    return f"{100.0 * accepted / total:.2f}"


def collect_remote_skip_logs(root, workers, target_stage, skip_prefix):
    if not workers or target_stage <= 1:
        return
    decision_rank = target_stage - 1
    worker_index = decision_rank - 1
    if worker_index < 0 or worker_index >= len(workers):
        return
    host = workers[worker_index].split(":", 1)[0]
    remote_glob = f"/home/ubuntu/{skip_prefix}.node*.tsv"
    subprocess.run(
        ["bash", "-lc", f"scp -q ubuntu@{host}:{remote_glob} {skip_prefix.parent}/ 2>/dev/null || true"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_qa(path):
    if path is None:
        return DEFAULT_QA
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["id"], row["prompt"], row["answer"].strip().upper()))
    return rows


def write_default_qa(path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "prompt", "answer"])
        writer.writerows(DEFAULT_QA)


def run_case(root, workers, out_dir, repeat_id, qa_id, prompt, answer, mode, extra_args):
    case_id = f"r{repeat_id:02d}_{qa_id}_{mode}"
    clean_prefix = out_dir / "clean" / case_id
    skip_prefix = out_dir / "skip" / case_id
    log_path = out_dir / "raw" / f"{case_id}.log"

    cmd = [
        "./dllama", "inference",
        "--model", env("MODEL", "dllama_model_llama3-8b_q40.m"),
        "--tokenizer", env("TOKENIZER", "dllama_tokenizer_llama3_8B.t"),
        "--prompt", prompt,
        "--steps", env("STEPS", "96"),
        "--temperature", env("TEMPERATURE", "0"),
        "--nthreads", env("NTHREADS", "4"),
        "--collective", "auto",
        "--net-turbo", "0",
        "--pp-size", env("PP_SIZE", "8"),
        "--sp-size", "1",
        "--pp-token-only", "1",
        "--pipeline-float-type", "q40",
        "--buffer-float-type", "q80",
        "--pipeline-chunk-bytes", "16384",
        "--pipeline-delta", "1",
        "--pipeline-delta-min-bytes", "4096",
        "--stage-timing", env("STAGE_TIMING", "0"),
        "--wall-metrics", "1",
        "--decode-log-interval", "0",
        "--clean-output-prefix", str(clean_prefix),
    ] + extra_args

    if mode != "fullroute":
        cmd += ["--pp-stage-skip-log-file", str(skip_prefix)]
    cmd += ["--workers"] + workers

    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)

    log_text = log_path.read_text(errors="replace")
    status = "ok" if proc.returncode == 0 else f"cmd_fail({proc.returncode})"
    if re.search(r"Critical error|NET_TIMEOUT|Broken pipe|Pipeline recv error|Pipeline send error|tryReadSocket timeout|writeMany timeout|readMany timeout", log_text):
        status = "error"
    if mode != "fullroute":
        collect_remote_skip_logs(root, workers, int(env("TARGET_STAGE", "4")), skip_prefix)

    text_path = clean_prefix.with_suffix(".text")
    text = text_path.read_text(errors="replace") if text_path.exists() else ""
    pred, pred_method = extract_choice(text)
    correct = int(status == "ok" and pred == answer)
    row = {
        "repeat": str(repeat_id),
        "id": qa_id,
        "mode": mode,
        "answer": answer,
        "pred": pred,
        "correct": str(correct),
        "wall_tpot_ms": parse_metric(log_text, "wall_tpot_ms"),
        "root_wait_p95_ms": parse_metric(log_text, "root_recv_wait_p95_ms"),
        "accept_rate": accept_rate(skip_prefix),
        "pred_method": pred_method,
        "log": str(log_path),
        "text": re.sub(r"\s+", " ", text)[:180],
        "wall_decode_tps": "NA",
    }
    print(f"[r{repeat_id:02d} {qa_id} {mode}] status={status} answer={answer} pred={pred or 'NA'} method={pred_method} correct={correct} tpot={row['wall_tpot_ms']} p95={row['root_wait_p95_ms']} accept={row['accept_rate']}")
    return row


def run_mode_batch(root, workers, out_dir, repeat_id, qa, mode, extra_args):
    case_id = f"r{repeat_id:02d}_{mode}"
    prompts_path = out_dir / "prompts" / f"{case_id}.txt"
    clean_prefix = out_dir / "clean" / case_id
    skip_prefix = out_dir / "skip" / case_id
    log_path = out_dir / "raw" / f"{case_id}.log"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w") as f:
        for _, prompt, _ in qa:
            f.write(prompt.replace("\n", " ") + "\n")

    cmd = [
        "./dllama", "inference",
        "--model", env("MODEL", "dllama_model_llama3-8b_q40.m"),
        "--tokenizer", env("TOKENIZER", "dllama_tokenizer_llama3_8B.t"),
        "--prompts-file", str(prompts_path),
        "--steps", env("STEPS", "160"),
        "--temperature", env("TEMPERATURE", "0"),
        "--nthreads", env("NTHREADS", "4"),
        "--collective", "auto",
        "--net-turbo", "0",
        "--pp-size", env("PP_SIZE", "8"),
        "--sp-size", "1",
        "--pp-token-only", "1",
        "--decode-cb-max-active", "1",
        "--decode-cb-max-new-tokens", env("MAX_NEW_TOKENS", "2"),
        "--pipeline-float-type", "q40",
        "--buffer-float-type", "q80",
        "--pipeline-chunk-bytes", "16384",
        "--pipeline-delta", "1",
        "--pipeline-delta-min-bytes", "4096",
        "--stage-timing", env("STAGE_TIMING", "0"),
        "--wall-metrics", "1",
        "--decode-log-interval", "0",
        "--clean-output-prefix", str(clean_prefix),
    ] + extra_args
    if mode != "fullroute":
        cmd += ["--pp-stage-skip-log-file", str(skip_prefix)]
    cmd += ["--workers"] + workers

    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)

    log_text = log_path.read_text(errors="replace")
    status = "ok" if proc.returncode == 0 else f"cmd_fail({proc.returncode})"
    if re.search(r"Critical error|NET_TIMEOUT|Broken pipe|Pipeline recv error|Pipeline send error|tryReadSocket timeout|writeMany timeout|readMany timeout", log_text):
        status = "error"
    if mode != "fullroute":
        collect_remote_skip_logs(root, workers, int(env("TARGET_STAGE", "4")), skip_prefix)

    wall_decode_tps = parse_metric(log_text, "wall_decode_tps")
    accept = accept_rate(skip_prefix)
    cb_path = clean_prefix.with_suffix(".cb.tsv")
    generated = {}
    if cb_path.exists():
        with cb_path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                generated[int(row["request_index"])] = row

    rows = []
    for i, (qa_id, _, answer) in enumerate(qa):
        gen = generated.get(i, {})
        text = gen.get("text", "")
        pred, pred_method = extract_choice(text)
        correct = int(status == "ok" and pred == answer)
        rows.append({
            "repeat": str(repeat_id),
            "id": qa_id,
            "mode": mode,
            "answer": answer,
            "pred": pred,
            "correct": str(correct),
            "wall_tpot_ms": "NA",
            "root_wait_p95_ms": "NA",
            "accept_rate": accept,
            "pred_method": pred_method,
            "log": str(log_path),
            "text": re.sub(r"\s+", " ", text)[:180],
            "wall_decode_tps": wall_decode_tps,
        })
        print(f"[r{repeat_id:02d} {qa_id} {mode}] status={status} answer={answer} pred={pred or 'NA'} method={pred_method} correct={correct} accept={accept}")
    return rows


def med(values):
    nums = [float(v) for v in values if v not in ("", "NA")]
    return "NA" if not nums else f"{statistics.median(nums):.2f}"


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    workers_env = os.environ.get("WORKERS", "").split()
    if not workers_env:
        print("ERROR: WORKERS env is required.", file=sys.stderr)
        print('Example: WORKERS="165.194.19.94:9998 ..." ./scripts/run_bypass_mc_quality.py', file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(env("OUT_DIR", f"bench_logs/bypass_mc_quality_{ts}"))
    for sub in ("raw", "clean", "skip", "prompts"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    qa_path = os.environ.get("QA_TSV")
    if qa_path:
        qa = load_qa(qa_path)
    else:
        qa_path = out_dir / "qa_mc.tsv"
        write_default_qa(qa_path)
        qa = DEFAULT_QA
    repeats = int(env("REPEATS", "1"))

    target = env("TARGET_STAGE", "4")
    modes = [
        ("fullroute", ["--pp-stage-skip", "0", "--pp-stage-skip-shadow", "0"]),
        ("conservative", [
            "--pp-stage-skip-shadow", "0",
            "--pp-stage-skip", "1",
            "--pp-stage-skip-target", target,
            "--pp-stage-skip-theta", env("CONS_THETA", "0.85"),
            "--pp-stage-skip-alpha", env("CONS_ALPHA", "0.50"),
            "--pp-stage-skip-verifier", "delta",
            "--pp-stage-skip-max-consecutive", env("CONS_MAX_CONSEC", "2"),
            "--pp-stage-skip-max-reject-streak", "8",
            "--pp-stage-skip-log", "1",
        ]),
        ("aggressive", [
            "--pp-stage-skip-shadow", "0",
            "--pp-stage-skip", "1",
            "--pp-stage-skip-target", target,
            "--pp-stage-skip-theta", env("AGGR_THETA", "0.90"),
            "--pp-stage-skip-alpha", env("AGGR_ALPHA", "1.00"),
            "--pp-stage-skip-verifier", "delta",
            "--pp-stage-skip-max-consecutive", env("AGGR_MAX_CONSEC", "4"),
            "--pp-stage-skip-max-reject-streak", "8",
            "--pp-stage-skip-log", "1",
        ]),
    ]

    rows = []
    fast_prompts_file = env("FAST_PROMPTS_FILE", "1") != "0"
    for repeat_id in range(1, repeats + 1):
        if fast_prompts_file:
            for mode, args in modes:
                rows.extend(run_mode_batch(root, workers_env, out_dir, repeat_id, qa, mode, args))
        else:
            for qa_id, prompt, answer in qa:
                for mode, args in modes:
                    rows.append(run_case(root, workers_env, out_dir, repeat_id, qa_id, prompt, answer, mode, args))

    results = out_dir / "results.tsv"
    with results.open("w", newline="") as f:
        fieldnames = ["repeat", "id", "mode", "answer", "pred", "correct", "wall_tpot_ms", "root_wait_p95_ms", "accept_rate", "pred_method", "log", "text", "wall_decode_tps"]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = out_dir / "summary.tsv"
    with summary.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["mode", "n", "valid_pred", "correct", "accuracy", "median_tpot_ms", "median_root_wait_p95_ms", "avg_accept_rate"])
        for mode, _ in modes:
            mrows = [r for r in rows if r["mode"] == mode]
            n = len(mrows)
            valid_pred = sum(1 for r in mrows if r["pred"])
            correct = sum(int(r["correct"]) for r in mrows)
            acc_vals = [float(r["accept_rate"]) for r in mrows if r["accept_rate"] != "NA"]
            writer.writerow([
                mode,
                n,
                valid_pred,
                correct,
                f"{correct / n:.3f}" if n else "NA",
                med(r["wall_tpot_ms"] for r in mrows),
                med(r["root_wait_p95_ms"] for r in mrows),
                f"{statistics.mean(acc_vals):.2f}" if acc_vals else "NA",
            ])

    print()
    print(f"Output dir: {out_dir}")
    print(f"Results:    {results}")
    print(f"Summary:    {summary}")
    print(summary.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
