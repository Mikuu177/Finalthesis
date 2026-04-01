import argparse
import csv
import json
from pathlib import Path


def _load(path: Path):
    d = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            d[str(o['id'])] = o
    return d


def main():
    ap = argparse.ArgumentParser(description='Offline fallback threshold sweep using router/fixed predictions')
    ap.add_argument('--router_pred', required=True)
    ap.add_argument('--fixed_pred', required=True)
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    ap.add_argument('--out', default='outputs/tables/fallback_threshold_sweep_offline.csv')
    args = ap.parse_args()

    router = _load(Path(args.router_pred).resolve())
    fixed = _load(Path(args.fixed_pred).resolve())
    ids = sorted(set(router.keys()) & set(fixed.keys()))
    if not ids:
        raise RuntimeError('No common ids between router and fixed predictions')

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['threshold', 'n', 'accuracy', 'fallback_rate', 'fallback_count', 'month_shift_accuracy', 'month_shift_fallback_count'],
        )
        w.writeheader()
        for th in args.thresholds:
            correct = 0
            fallback_count = 0
            month_n = 0
            month_correct = 0
            month_fb = 0
            for sid in ids:
                rr = router[sid]
                ff = fixed[sid]
                conf = rr.get('classifier_confidence')
                conf = float(conf) if conf is not None else None
                use_fb = (conf is None) or (conf < th)
                row = ff if use_fb else rr
                ok = int(bool(row.get('correct')))
                correct += ok
                if use_fb:
                    fallback_count += 1
                cat = rr.get('category') or rr.get('true_category')
                if cat == 'Month Shift':
                    month_n += 1
                    month_correct += ok
                    if use_fb:
                        month_fb += 1

            n = len(ids)
            w.writerow(
                {
                    'threshold': th,
                    'n': n,
                    'accuracy': f'{correct/n:.4f}',
                    'fallback_rate': f'{fallback_count/n:.4f}',
                    'fallback_count': fallback_count,
                    'month_shift_accuracy': f'{(month_correct/month_n if month_n else 0):.4f}',
                    'month_shift_fallback_count': month_fb,
                }
            )

    print(f'[OK] wrote: {out}')


if __name__ == '__main__':
    main()
