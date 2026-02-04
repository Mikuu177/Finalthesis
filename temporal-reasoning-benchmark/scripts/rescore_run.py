import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# Import scorer v1
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add src/ to path
from src.scorer import score_v1  # type: ignore


def tail(text: str, n: int = 10) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='Path to a run directory containing predictions.jsonl')
    ap.add_argument('--version', default='v1', help='Scoring version tag to write')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / 'predictions.jsonl'
    if not pred_path.exists():
        print(f"[ERR] Not found: {pred_path}")
        return 1

    # Accumulators
    overall = {'n': 0, 'parsed': 0, 'correct': 0}
    by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {'n':0,'parsed':0,'correct':0})

    failures = []

    # Rescore
    rescored_rows = []
    with pred_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cat = str(row.get('category', 'unspecified'))
            pred = str(row.get('pred', ''))
            gold = str(row.get('gold', ''))
            parsed = bool(row.get('parsed', False))

            correct, info = score_v1(pred, gold, cat)

            # stats
            overall['n'] += 1
            if parsed:
                overall['parsed'] += 1
            if correct:
                overall['correct'] += 1

            by_cat[cat]['n'] += 1
            if parsed:
                by_cat[cat]['parsed'] += 1
            if correct:
                by_cat[cat]['correct'] += 1

            # collect failures
            if not correct:
                failures.append({
                    'id': row.get('id'),
                    'category': cat,
                    'pred': pred,
                    'gold': gold,
                    'parsed': parsed,
                    'match': info.get('match'),
                    'raw_tail': tail(row.get('raw',''), 10),
                })

            # keep for potential extended outputs
            rescored_rows.append({**row, 'correct_v1': correct, 'match_v1': info.get('match')})

    # Write summary_v1.csv
    sum_path = run_dir / 'summary_v1.csv'
    with sum_path.open('w', encoding='utf-8', newline='') as f:
        f.write('model,category,n,correct,accuracy\n')
        # Derive model name from first row if present
        model_name = 'unknown'
        if rescored_rows:
            model_name = str(rescored_rows[0].get('model', 'unknown'))
        for cat, g in by_cat.items():
            n, c = g['n'], g['correct']
            acc = (c/n) if n else 0.0
            f.write(f"{model_name},{cat},{n},{c},{acc:.4f}\n")

    # Write quick_metrics_v1.txt
    qm_path = run_dir / 'quick_metrics_v1.txt'
    with qm_path.open('w', encoding='utf-8') as f:
        parse_rate = (overall['parsed']/overall['n']) if overall['n'] else 0.0
        acc = (overall['correct']/overall['n']) if overall['n'] else 0.0
        f.write('Overall\n')
        f.write(f"  samples = {overall['n']}\n")
        f.write(f"  parse_rate = {parse_rate:.4f}\n")
        f.write(f"  accuracy   = {acc:.4f}\n\n")
        f.write('By category\n')
        for cat in sorted(by_cat.keys()):
            g = by_cat[cat]
            n = g['n']
            pr = (g['parsed']/n) if n else 0.0
            a = (g['correct']/n) if n else 0.0
            f.write(f"  {cat:12s} n={n:3d} parse_rate={pr:.4f} acc={a:.4f}\n")

    # Write failures_v1.jsonl (all failures)
    fail_path = run_dir / 'failures_v1.jsonl'
    with fail_path.open('w', encoding='utf-8') as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Update run_metadata.json
    meta_path = run_dir / 'run_metadata.json'
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    meta['scoring_version'] = args.version
    meta['scoring_notes'] = 'v1: duration minutes normalization + relation canonical/entity-only mapping'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Rescore completed: {sum_path}, {qm_path}, {fail_path}")

if __name__ == '__main__':
    main()






