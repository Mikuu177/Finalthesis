import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_jsonl

GOLD_PAT = re.compile(r"(?i)(\d{1,2})\s*(AM|PM)(?:\s*on\b)?")


def _to_hhmm(text: str) -> Optional[str]:
    m = GOLD_PAT.search(text or "")
    if not m:
        return None
    hour = int(m.group(1))
    ampm = m.group(2).upper()
    if hour < 1 or hour > 12:
        return None
    if ampm == "AM":
        hh = 0 if hour == 12 else hour
    else:
        hh = 12 if hour == 12 else hour + 12
    return f"{hh:02d}:00"


def convert_file(path: Path) -> int:
    rows = read_jsonl(path)
    changed = 0
    for r in rows:
        g = str(r.get("gold", ""))
        hhmm = _to_hhmm(g)
        if hhmm:
            md = r.get("metadata", {}) or {}
            md["gold_original"] = g
            md["gold_normalized_mode"] = "tz_hhmm_from_am_pm"
            r["metadata"] = md
            r["gold"] = hhmm
            changed += 1
    write_jsonl(path, rows)
    return changed


def main():
    ap = argparse.ArgumentParser(description="Normalize Time Zone Conversion gold to HH:MM for current prompt/eval contract")
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    for fp in args.files:
        p = Path(fp).resolve()
        c = convert_file(p)
        print(f"[OK] {p} changed_rows={c}")


if __name__ == "__main__":
    main()
