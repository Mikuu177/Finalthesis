import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path):
    d = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            d[str(o['id'])] = o
    return d


def main():
    ap = argparse.ArgumentParser(description='Compute Oracle prompt upper bound from fixed vs cot predictions')
    ap.add_argument('--fixed_pred', required=True)
    ap.add_argument('--cot_pred', required=True)
    ap.add_argument('--out', default='outputs/tables/prompt_oracle_upper_bound.csv')
    ap.add_argument('--out_category', default='outputs/tables/prompt_oracle_upper_bound_categorywise.csv')
    args = ap.parse_args()

    fixed = _load(Path(args.fixed_pred).resolve())
    cot = _load(Path(args.cot_pred).resolve())
    ids = sorted(set(fixed.keys()) & set(cot.keys()))
    if not ids:
        raise RuntimeError('No common ids between fixed and cot predictions')

    # choose best prompt type per category by observed per-category accuracy
    cat_stat = defaultdict(lambda: {'fixed': [0, 0], 'cot': [0, 0]})
    for sid in ids:
        f = fixed[sid]
        c = cot[sid]
        cat = str(f.get('category') or f.get('true_category') or c.get('category') or c.get('true_category') or 'unknown')
        cat_stat[cat]['fixed'][0] += 1
        cat_stat[cat]['fixed'][1] += int(bool(f.get('correct')))
        cat_stat[cat]['cot'][0] += 1
        cat_stat[cat]['cot'][1] += int(bool(c.get('correct')))

    cat_best = {}
    for cat, v in cat_stat.items():
        fn, fk = v['fixed']
        cn, ck = v['cot']
        fa = fk / fn if fn else 0.0
        ca = ck / cn if cn else 0.0
        cat_best[cat] = 'fixed' if fa >= ca else 'cot'

    n = len(ids)
    oracle_correct = 0
    cat_oracle = defaultdict(lambda: [0, 0])
    for sid in ids:
        f = fixed[sid]
        c = cot[sid]
        cat = str(f.get('category') or f.get('true_category') or c.get('category') or c.get('true_category') or 'unknown')
        pick = cat_best[cat]
        row = f if pick == 'fixed' else c
        ok = int(bool(row.get('correct')))
        oracle_correct += ok
        cat_oracle[cat][0] += 1
        cat_oracle[cat][1] += ok

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['workflow', 'n', 'correct', 'accuracy', 'note'])
        w.writeheader()
        # overall fixed/cot
        fk = sum(int(bool(fixed[s]['correct'])) for s in ids)
        ck = sum(int(bool(cot[s]['correct'])) for s in ids)
        w.writerow({'workflow': 'Fixed Prompt', 'n': n, 'correct': fk, 'accuracy': f'{fk/n:.4f}', 'note': ''})
        w.writerow({'workflow': 'CoT Prompt', 'n': n, 'correct': ck, 'accuracy': f'{ck/n:.4f}', 'note': ''})
        w.writerow({'workflow': 'Oracle Prompt Upper Bound', 'n': n, 'correct': oracle_correct, 'accuracy': f'{oracle_correct/n:.4f}', 'note': 'select best prompt per category'})

    out_cat = Path(args.out_category).resolve()
    with out_cat.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['category', 'best_prompt', 'n', 'correct', 'accuracy', 'fixed_accuracy', 'cot_accuracy'])
        w.writeheader()
        for cat in sorted(cat_best.keys()):
            fn, fk = cat_stat[cat]['fixed']
            cn, ck = cat_stat[cat]['cot']
            on, ok = cat_oracle[cat]
            w.writerow({
                'category': cat,
                'best_prompt': cat_best[cat],
                'n': on,
                'correct': ok,
                'accuracy': f'{(ok/on if on else 0):.4f}',
                'fixed_accuracy': f'{(fk/fn if fn else 0):.4f}',
                'cot_accuracy': f'{(ck/cn if cn else 0):.4f}',
            })

    print(f'[OK] wrote: {out}')
    print(f'[OK] wrote: {out_cat}')


if __name__ == '__main__':
    main()
