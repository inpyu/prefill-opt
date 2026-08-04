#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def choice_label(i):
    return "ABCD"[i]


def format_prompt(question, choices):
    parts = [f"{choice_label(i)}. {text}" for i, text in enumerate(choices[:4])]
    return (
        "Answer the multiple-choice question by writing only one letter: A, B, C, or D.\n"
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(parts)
        + "\nAnswer:"
    )


def load_arc_easy(split, limit, seed):
    from datasets import load_dataset

    data = load_dataset("ai2_arc", "ARC-Easy", split=split)
    idxs = list(range(len(data)))
    random.Random(seed).shuffle(idxs)
    rows = []
    for idx in idxs:
        item = data[idx]
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        if len(labels) < 4 or len(texts) < 4:
            continue
        label_to_text = {str(label).strip().upper(): text for label, text in zip(labels, texts)}
        if not all(label in label_to_text for label in "ABCD"):
            continue
        answer = str(item["answerKey"]).strip().upper()
        if answer not in "ABCD":
            continue
        choices = [label_to_text[label] for label in "ABCD"]
        rows.append((f"arc_easy_{len(rows)+1:03d}", format_prompt(item["question"], choices), answer))
        if len(rows) >= limit:
            break
    return rows


def load_hellaswag(split, limit, seed):
    from datasets import load_dataset

    data = load_dataset("hellaswag", split=split)
    idxs = list(range(len(data)))
    random.Random(seed).shuffle(idxs)
    rows = []
    for idx in idxs:
        item = data[idx]
        endings = item["endings"]
        if len(endings) != 4:
            continue
        answer = choice_label(int(item["label"]))
        question = f"{item['ctx']}"
        choices = [ending.strip() for ending in endings]
        rows.append((f"hellaswag_{len(rows)+1:03d}", format_prompt(question, choices), answer))
        if len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["arc_easy", "hellaswag"], default="arc_easy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.limit > 100:
        raise SystemExit("Use --limit <= 100 for this lightweight diagnostic")

    if args.task == "arc_easy":
        rows = load_arc_easy(args.split, args.limit, args.seed)
    else:
        rows = load_hellaswag(args.split, args.limit, args.seed)

    if not rows:
        raise SystemExit("No benchmark rows generated")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "prompt", "answer"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows: {out}")


if __name__ == "__main__":
    main()
