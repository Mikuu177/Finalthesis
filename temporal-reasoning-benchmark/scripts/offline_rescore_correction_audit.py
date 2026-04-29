import argparse
import calendar
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _month_map() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, m in enumerate(calendar.month_name):
        if i > 0 and m:
            out[m.lower()] = i
    for i, m in enumerate(calendar.month_abbr):
        if i > 0 and m:
            out[m.lower()] = i
    return out


_MMAP = _month_map()
_PAT_TZ = re.compile(r"^(\d{1,2})(?::(\d{2}))?(AM|PM)on", re.IGNORECASE)
_PAT_YYYY_MM = re.compile(r"^(\d{4})-(\d{2})(?:-\d{2})?$")
_PAT_MONTH_YEAR = re.compile(r"^([A-Za-z]+)\s*(\d{3,4})$")


def _norm_cat(row: Dict) -> str:
    return str(row.get("true_category") or row.get("category") or "unspecified")


def _tz_gold_to_hhmm(gold: str) -> str:
    m = _PAT_TZ.match(gold or "")
    if not m:
        return ""
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ap = m.group(3).upper()
    if ap == "AM":
        hh = 0 if hh == 12 else hh
    else:
        hh = 12 if hh == 12 else hh + 12
    return f"{hh:02d}:{mm:02d}"


def _is_month_year_equiv(gold: str, pred_norm: str) -> bool:
    mg = _PAT_MONTH_YEAR.match((gold or "").replace(" ", ""))
    mp = _PAT_YYYY_MM.match(pred_norm or "")
    if not mg or not mp:
        return False
    m_txt = mg.group(1).lower()
    year = int(mg.group(2))
    month = _MMAP.get(m_txt)
    if month is None:
        return False
    return int(mp.group(1)) == year and int(mp.group(2)) == month


def _corrected_label(row: Dict) -> Tuple[bool, str]:
    if bool(row.get("correct")):
        return True, "base_correct"

    cat = _norm_cat(row)
    gold = str(row.get("gold", ""))
    pred_norm = str(row.get("pred_norm", ""))

    if cat == "Date Computation" and _is_month_year_equiv(gold, pred_norm):
        return True, "fix_month_year_equiv"

    if cat == "Time Zone Conversion":
        hhmm = _tz_gold_to_hhmm(gold)
        if hhmm and pred_norm == hhmm:
            return True, "fix_tz_hhmm_equiv"

    return False, "unchanged_wrong"


def _safe_acc(k: int, n: int) -> float:
    return (k / n) if n else 0.0


def _compute(rows: Iterable[Dict]) -> Tuple[Dict, Dict[str, Dict]]:
    rows = list(rows)
    n = len(rows)
    base_correct = 0
    corrected_correct = 0
    reason_cnt: Counter = Counter()
    by_cat: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "base_correct": 0, "corrected_correct": 0})

    for r in rows:
        cat = _norm_cat(r)
        by_cat[cat]["n"] += 1
        if bool(r.get("correct")):
            base_correct += 1
            by_cat[cat]["base_correct"] += 1

        cc, reason = _corrected_label(r)
        reason_cnt[reason] += 1
        if cc:
            corrected_correct += 1
            by_cat[cat]["corrected_correct"] += 1

    summary = {
        "n": n,
        "base_correct": base_correct,
        "base_accuracy": _safe_acc(base_correct, n),
        "corrected_correct": corrected_correct,
        "corrected_accuracy": _safe_acc(corrected_correct, n),
        "delta_accuracy": _safe_acc(corrected_correct, n) - _safe_acc(base_correct, n),
        "fix_month_year_equiv": int(reason_cnt.get("fix_month_year_equiv", 0)),
        "fix_tz_hhmm_equiv": int(reason_cnt.get("fix_tz_hhmm_equiv", 0)),
    }
    return summary, by_cat


def main():
    ap = argparse.ArgumentParser(description="Offline correction audit for QA scoring (no model re-run).")
    ap.add_argument("--run", action="append", required=True, help="Format: <name>=<run_dir>")
    ap.add_argument("--out", default="outputs/tables/offline_correction_audit_summary.csv")
    ap.add_argument("--out_category", default="outputs/tables/offline_correction_audit_categorywise.csv")
    args = ap.parse_args()

    entries: List[Tuple[str, Path]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run '{item}'. Expect name=path")
        name, path = item.split("=", 1)
        entries.append((name.strip(), Path(path).resolve()))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_cat = Path(args.out_category).resolve()
    out_cat.parent.mkdir(parents=True, exist_ok=True)

    sum_rows: List[Dict] = []
    cat_rows: List[Dict] = []

    for name, run_dir in entries:
        pred = run_dir / "predictions.jsonl"
        if not pred.exists():
            raise FileNotFoundError(f"Not found: {pred}")
        rows = _load_jsonl(pred)
        s, by_cat = _compute(rows)
        sum_rows.append(
            {
                "workflow": name,
                "run_dir": str(run_dir),
                "n": s["n"],
                "base_accuracy": f"{s['base_accuracy']:.4f}",
                "corrected_accuracy": f"{s['corrected_accuracy']:.4f}",
                "delta_accuracy": f"{s['delta_accuracy']:.4f}",
                "fix_month_year_equiv": s["fix_month_year_equiv"],
                "fix_tz_hhmm_equiv": s["fix_tz_hhmm_equiv"],
            }
        )

        for cat, d in sorted(by_cat.items()):
            cat_rows.append(
                {
                    "workflow": name,
                    "run_dir": str(run_dir),
                    "category": cat,
                    "n": d["n"],
                    "base_accuracy": f"{_safe_acc(d['base_correct'], d['n']):.4f}",
                    "corrected_accuracy": f"{_safe_acc(d['corrected_correct'], d['n']):.4f}",
                    "delta_accuracy": f"{_safe_acc(d['corrected_correct'], d['n']) - _safe_acc(d['base_correct'], d['n']):.4f}",
                }
            )

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "run_dir",
                "n",
                "base_accuracy",
                "corrected_accuracy",
                "delta_accuracy",
                "fix_month_year_equiv",
                "fix_tz_hhmm_equiv",
            ],
        )
        w.writeheader()
        w.writerows(sum_rows)

    with out_cat.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "run_dir",
                "category",
                "n",
                "base_accuracy",
                "corrected_accuracy",
                "delta_accuracy",
            ],
        )
        w.writeheader()
        w.writerows(cat_rows)

    print(f"[OK] wrote: {out}")
    print(f"[OK] wrote: {out_cat}")


if __name__ == "__main__":
    main()
