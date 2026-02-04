import argparse
import json
from pathlib import Path

# Make repo importable when executed as a script
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.normalize import (
    extract_time_fragment,
    extract_date_fragment_to_iso,
    contains_tz_token,
)


def load_pred(run_dir: Path):
    pred_path = run_dir / 'predictions.jsonl'
    rows = []
    with pred_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='TZ smoke run directory containing predictions.jsonl')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_pred(run_dir)
    if not rows:
        print('[ERR] empty predictions')
        return

    gold_time = 0
    gold_date = 0
    gold_tz = 0
    gold_time_only = 0
    gold_datetime = 0

    match_after_time_extract = 0

    for r in rows:
        gold_raw = str(r.get('gold', ''))
        pred_raw = str(r.get('pred', ''))

        gt = extract_time_fragment(gold_raw)
        pt = extract_time_fragment(pred_raw)
        if gt:
            gold_time += 1
        if extract_date_fragment_to_iso(gold_raw):
            gold_date += 1
        if contains_tz_token(gold_raw):
            gold_tz += 1

        if gt and (not extract_date_fragment_to_iso(gold_raw)):
            gold_time_only += 1
        if gt and extract_date_fragment_to_iso(gold_raw):
            gold_datetime += 1

        if gt and pt and gt == pt:
            match_after_time_extract += 1

    n = len(rows)
    out = {
        'n': n,
        'gold_contains_time_rate': gold_time / n,
        'gold_contains_date_rate': gold_date / n,
        'gold_contains_tz_token_rate': gold_tz / n,
        'gold_is_time_only_rate': gold_time_only / n,
        'gold_is_datetime_rate': gold_datetime / n,
        'match_after_time_extract_rate': match_after_time_extract / n,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
