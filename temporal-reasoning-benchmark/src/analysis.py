import argparse
from pathlib import Path
import csv
from collections import defaultdict


def read_summary(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def aggregate(run_dirs):
    # model -> category -> {n, correct}
    agg = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'correct': 0}))
    for run_dir in run_dirs:
        summary_path = Path(run_dir) / 'summary.csv'
        if not summary_path.exists():
            print(f"[WARN] skip, not found: {summary_path}")
            continue
        for row in read_summary(summary_path):
            model = row['model']
            cat = row['category']
            n = int(row['n'])
            c = int(row['correct'])
            agg[model][cat]['n'] += n
            agg[model][cat]['correct'] += c
    return agg


def write_table(agg, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # collect header
    categories = sorted({cat for model in agg for cat in agg[model].keys()})
    header = ['model'] + [f"acc@{c}" for c in categories] + ['overall_n', 'overall_correct', 'overall_acc']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for model in sorted(agg.keys()):
            row = [model]
            overall_n = 0
            overall_c = 0
            for cat in categories:
                n = agg[model][cat]['n']
                c = agg[model][cat]['correct']
                acc = (c / n) if n > 0 else 0.0
                row.append(f"{acc:.4f}")
                overall_n += n
                overall_c += c
            overall_acc = (overall_c / overall_n) if overall_n > 0 else 0.0
            row += [overall_n, overall_c, f"{overall_acc:.4f}"]
            w.writerow(row)
    print(f"Wrote table: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='Run directories containing summary.csv')
    ap.add_argument('--outputs', default='outputs/tables', help='Where to write combined tables')
    args = ap.parse_args()

    if not args.runs:
        # auto-discover in outputs/runs
        base = Path('outputs/runs')
        run_dirs = [p for p in base.glob('*') if p.is_dir()]
    else:
        run_dirs = [Path(p) for p in args.runs]

    if not run_dirs:
        print("No run dirs found.")
        return

    agg = aggregate(run_dirs)
    out_dir = Path(args.outputs)
    out_path = out_dir / 'combined_summary.csv'
    write_table(agg, out_path)


if __name__ == '__main__':
    main()






