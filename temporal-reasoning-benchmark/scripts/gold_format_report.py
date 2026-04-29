import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("iso_ymd", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("mdy_dash", re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$")),
    ("mdy_slash", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    ("month_day", re.compile(r"^\d{1,2}-[A-Za-z]{3,9}$")),
    ("month_year_text", re.compile(r"^[A-Za-z]{3,9}[-\s]+\d{1,4}$")),
    ("hhmm", re.compile(r"^\d{1,2}:\d{2}$")),
    ("tz_compact_ampm_on", re.compile(r"^\d{1,2}(?::\d{2})?(AM|PM)on", re.IGNORECASE)),
    ("number", re.compile(r"^[+-]?\d+(?:\.\d+)?$")),
    ("label_abcd", re.compile(r"^[ABCD]$", re.IGNORECASE)),
)


def _classify_gold(gold: str) -> str:
    s = (gold or "").strip()
    if not s:
        return "empty"
    for name, pat in PATTERNS:
        if pat.match(s):
            return name
    if re.search(r"\d", s) and re.search(r"[A-Za-z]", s):
        return "mixed_alnum_other"
    return "other"


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Build gold format distribution report for split datasets.")
    ap.add_argument("--splits", nargs="+", required=True, help="One or more split jsonl files.")
    ap.add_argument("--out", default="outputs/tables/gold_format_report.csv")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in args.splits:
        p = Path(split).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing split: {p}")

        cnt = Counter()
        by_cat = defaultdict(Counter)
        total = 0
        for r in _read_jsonl(p):
            category = str(r.get("category", "unspecified"))
            fmt = _classify_gold(str(r.get("gold", "")))
            cnt[fmt] += 1
            by_cat[category][fmt] += 1
            total += 1

        for fmt, n in cnt.most_common():
            rows.append(
                {
                    "split_file": str(p),
                    "scope": "all",
                    "category": "all",
                    "format": fmt,
                    "count": n,
                    "ratio": f"{(n / total) if total else 0.0:.4f}",
                }
            )
        for cat, cat_cnt in sorted(by_cat.items()):
            cat_n = sum(cat_cnt.values())
            for fmt, n in cat_cnt.most_common():
                rows.append(
                    {
                        "split_file": str(p),
                        "scope": "category",
                        "category": cat,
                        "format": fmt,
                        "count": n,
                        "ratio": f"{(n / cat_n) if cat_n else 0.0:.4f}",
                    }
                )

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_file", "scope", "category", "format", "count", "ratio"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()

