import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _text(row: Dict) -> str:
    q = str(row.get('question', '') or '')
    c = str(row.get('context', '') or '')
    return f"{q}\n{c}" if c else q


def main():
    ap = argparse.ArgumentParser(description='Evaluate saved task classifier on eval dataset')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    model_path = (PROJECT_ROOT / cfg.get('model_path', 'outputs/classifier/task_clf.joblib')).resolve()
    eval_path = (PROJECT_ROOT / cfg['eval_dataset_path']).resolve()
    out_dir = (PROJECT_ROOT / cfg.get('output_dir', 'outputs/classifier_eval')).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        from sklearn.metrics import accuracy_score, classification_report, f1_score
    except Exception as e:
        raise RuntimeError('Missing dependency for classifier eval. Please install scikit-learn and joblib.') from e

    clf = joblib.load(model_path)
    rows = _load_jsonl(eval_path)

    X = [_text(r) for r in rows]
    y_true = [str(r.get('category', 'unspecified')) for r in rows]
    y_pred = [str(x) for x in clf.predict(X)]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))

    report = {
        'model_path': str(model_path),
        'eval_dataset_path': str(eval_path),
        'n_eval': len(rows),
        'accuracy': acc,
        'macro_f1': macro_f1,
    }
    (out_dir / 'eval_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'eval_classification_report.txt').write_text(
        classification_report(y_true, y_pred, digits=4),
        encoding='utf-8',
    )

    with (out_dir / 'eval_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'true_category', 'pred_category', 'correct'])
        for r, yp in zip(rows, y_pred):
            yt = str(r.get('category', 'unspecified'))
            w.writerow([r.get('id'), yt, yp, yt == yp])

    print(f"[OK] eval accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    print(f"[OK] wrote: {out_dir / 'eval_report.json'}")


if __name__ == '__main__':
    main()
