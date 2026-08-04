#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


REQUIRED_MEMORY_MB = {
    "1": 8282,
    "2": 5103,
    "4": 3655,
    "8": 2931,
}


def as_float(value):
    if value in ("", "NA", None):
        return None
    return float(value)


def median(values):
    values = [v for v in values if v is not None]
    if not values:
        return "NA"
    return f"{statistics.median(values):.2f}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pp_sweep.tsv>", file=sys.stderr)
        return 1

    src = Path(sys.argv[1])
    if not src.is_file():
        print(f"ERROR: input TSV not found: {src}", file=sys.stderr)
        return 1

    rows_by_pp = {}
    with src.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("wall_tpot_ms") in ("", "NA", None):
                continue
            rows_by_pp.setdefault(row["pp"], []).append(row)

    out = src.with_suffix("").with_name(src.with_suffix("").name + ".paper.tsv")
    with out.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "pp",
            "ok_runs",
            "tpot_median_ms",
            "root_wait_p50_median_ms",
            "root_wait_p95_median_ms",
            "required_memory_mb",
            "source_logs",
        ])
        for pp in sorted(rows_by_pp, key=lambda x: int(x)):
            rows = rows_by_pp[pp]
            writer.writerow([
                pp,
                len(rows),
                median(as_float(r.get("wall_tpot_ms")) for r in rows),
                median(as_float(r.get("root_wait_p50")) for r in rows),
                median(as_float(r.get("root_wait_p95")) for r in rows),
                REQUIRED_MEMORY_MB.get(pp, "NA"),
                ",".join(r.get("log", "") for r in rows),
            ])

    print(f"Wrote: {out}")
    print(out.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
