import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Create strict train/dev/test split for classifier-routing study')
    ap.add_argument('--dataset', default='data/raw/tram_arithmetic_mcq.jsonl')
    ap.add_argument('--out_dir', default='data/splits/classifier_router')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--dev_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--categories', nargs='+', required=True)
    args = ap.parse_args()

    if abs((args.train_ratio + args.dev_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError('train/dev/test ratios must sum to 1.0')

    ds_path = (PROJECT_ROOT / args.dataset).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(ds_path)
    allow = set(args.categories)
    rows = [r for r in rows if str(r.get('category', '')) in allow]
    if len(rows) < 200:
        raise RuntimeError(f'Not enough rows after filtering categories: {len(rows)}')

    # Import sklearn lazily
    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise RuntimeError('scikit-learn is required for split creation.') from e

    y = [str(r.get('category', 'unspecified')) for r in rows]

    train_rows, temp_rows, y_train, y_temp = train_test_split(
        rows,
        y,
        test_size=(args.dev_ratio + args.test_ratio),
        random_state=args.seed,
        stratify=y,
    )

    temp_test_ratio = args.test_ratio / (args.dev_ratio + args.test_ratio)
    dev_rows, test_rows, y_dev, y_test = train_test_split(
        temp_rows,
        y_temp,
        test_size=temp_test_ratio,
        random_state=args.seed + 1,
        stratify=y_temp,
    )

    _write_jsonl(out_dir / 'train.jsonl', train_rows)
    _write_jsonl(out_dir / 'dev.jsonl', dev_rows)
    _write_jsonl(out_dir / 'test.jsonl', test_rows)

    summary = {
        'dataset': str(ds_path),
        'out_dir': str(out_dir),
        'seed': args.seed,
        'ratios': {'train': args.train_ratio, 'dev': args.dev_ratio, 'test': args.test_ratio},
        'categories': sorted(list(allow)),
        'n': {'train': len(train_rows), 'dev': len(dev_rows), 'test': len(test_rows), 'total': len(rows)},
        'dist': {
            'train': dict(Counter(str(r.get('category', 'unspecified')) for r in train_rows)),
            'dev': dict(Counter(str(r.get('category', 'unspecified')) for r in dev_rows)),
            'test': dict(Counter(str(r.get('category', 'unspecified')) for r in test_rows)),
        },
    }
    (out_dir / 'split_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[OK] split created: {out_dir}")
    print(json.dumps(summary['n'], ensure_ascii=False))


if __name__ == '__main__':
    main()
