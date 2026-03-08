import json
import sys
from pathlib import Path
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python scripts/quick_metrics.py <run_dir>")
    sys.exit(1)

run_dir = Path(sys.argv[1])
pred_path = run_dir / 'predictions.jsonl'

if not pred_path.exists():
    print(f"Not found: {pred_path}")
    sys.exit(2)

n_total = 0
n_parsed = 0
n_correct = 0
by_cat = defaultdict(lambda: {'n':0,'parsed':0,'correct':0})

with pred_path.open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        n_total += 1
        parsed = bool(row.get('parsed', row.get('extract_ok', False)))
        if parsed:
            n_parsed += 1
        if row.get('correct'):
            n_correct += 1
        cat = row.get('category','unspecified')
        by_cat[cat]['n'] += 1
        if parsed:
            by_cat[cat]['parsed'] += 1
        if row.get('correct'):
            by_cat[cat]['correct'] += 1

print("Overall:")
print(f"  samples = {n_total}")
print(f"  parse_rate = { (n_parsed/n_total if n_total else 0):.4f }")
print(f"  accuracy   = { (n_correct/n_total if n_total else 0):.4f }")
print()
print("By category:")
for cat in sorted(by_cat.keys()):
    g = by_cat[cat]
    n = g['n']
    pr = g['parsed']/n if n else 0
    acc = g['correct']/n if n else 0
    print(f"  {cat:12s}  n={n:3d}  parse_rate={pr:.4f}  acc={acc:.4f}")





