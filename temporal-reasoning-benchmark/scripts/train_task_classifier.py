import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_text(row: Dict) -> str:
    q = str(row.get('question', '') or '')
    c = str(row.get('context', '') or '')
    if c:
        return f"{q}\n{c}"
    return q


def _safe_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        import joblib
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for classifier training. Please install scikit-learn and joblib."
        ) from e
    return {
        'TfidfVectorizer': TfidfVectorizer,
        'LogisticRegression': LogisticRegression,
        'accuracy_score': accuracy_score,
        'classification_report': classification_report,
        'confusion_matrix': confusion_matrix,
        'f1_score': f1_score,
        'train_test_split': train_test_split,
        'Pipeline': Pipeline,
        'joblib': joblib,
    }


def main():
    ap = argparse.ArgumentParser(description='Train small task classifier for temporal categories')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    dataset_path = (PROJECT_ROOT / cfg['dataset_path']).resolve() if cfg.get('dataset_path') else None
    train_dataset_path = (PROJECT_ROOT / cfg['train_dataset_path']).resolve() if cfg.get('train_dataset_path') else None
    eval_dataset_path = (PROJECT_ROOT / cfg['eval_dataset_path']).resolve() if cfg.get('eval_dataset_path') else None
    out_dir = (PROJECT_ROOT / cfg.get('output_dir', 'outputs/classifier')).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_categories = cfg.get('include_categories') or []
    min_samples_per_class = int(cfg.get('min_samples_per_class', 20))
    test_size = float(cfg.get('test_size', 0.2))
    random_state = int(cfg.get('seed', 42))

    if train_dataset_path:
        train_rows = _load_jsonl(train_dataset_path)
        eval_rows = _load_jsonl(eval_dataset_path) if eval_dataset_path else []

        if include_categories:
            allow = set(include_categories)
            train_rows = [r for r in train_rows if str(r.get('category', '')) in allow]
            if eval_rows:
                eval_rows = [r for r in eval_rows if str(r.get('category', '')) in allow]

        counts = Counter(str(r.get('category', 'unspecified')) for r in train_rows)
        keep = {k for k, v in counts.items() if v >= min_samples_per_class}
        train_rows = [r for r in train_rows if str(r.get('category', 'unspecified')) in keep]
        if eval_rows:
            eval_rows = [r for r in eval_rows if str(r.get('category', 'unspecified')) in keep]

        if len(train_rows) < 100:
            raise RuntimeError(f'Not enough train rows after filtering: {len(train_rows)}')

        X_train = [_build_text(r) for r in train_rows]
        y_train = [str(r.get('category', 'unspecified')) for r in train_rows]

        if eval_rows:
            X_test = [_build_text(r) for r in eval_rows]
            y_test = [str(r.get('category', 'unspecified')) for r in eval_rows]
        else:
            X_test, y_test = [], []
    else:
        if not dataset_path:
            raise ValueError('Either dataset_path or train_dataset_path must be provided.')
        rows = _load_jsonl(dataset_path)
        if include_categories:
            allow = set(include_categories)
            rows = [r for r in rows if str(r.get('category', '')) in allow]

        counts = Counter(str(r.get('category', 'unspecified')) for r in rows)
        keep = {k for k, v in counts.items() if v >= min_samples_per_class}
        rows = [r for r in rows if str(r.get('category', 'unspecified')) in keep]

        if len(rows) < 100:
            raise RuntimeError(f'Not enough rows after filtering: {len(rows)}')

        X = [_build_text(r) for r in rows]
        y = [str(r.get('category', 'unspecified')) for r in rows]

    libs = _safe_import_sklearn()
    train_test_split = libs['train_test_split']
    Pipeline = libs['Pipeline']
    TfidfVectorizer = libs['TfidfVectorizer']
    LogisticRegression = libs['LogisticRegression']
    accuracy_score = libs['accuracy_score']
    f1_score = libs['f1_score']
    classification_report = libs['classification_report']
    confusion_matrix = libs['confusion_matrix']
    joblib = libs['joblib']

    if not train_dataset_path:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ('lr', LogisticRegression(max_iter=1200, n_jobs=None, multi_class='auto')),
    ])
    clf.fit(X_train, y_train)

    if X_test:
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
        labels = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
    else:
        y_pred = []
        acc = None
        macro_f1 = None
        labels = sorted(set(y_train))
        cm = None

    model_path = out_dir / cfg.get('model_filename', 'task_clf.joblib')
    joblib.dump(clf, model_path)

    report = {
        'dataset_path': str(dataset_path) if dataset_path else None,
        'train_dataset_path': str(train_dataset_path) if train_dataset_path else None,
        'eval_dataset_path': str(eval_dataset_path) if eval_dataset_path else None,
        'n_total': (len(X_train) + len(X_test)),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'include_categories': sorted(set(y_train) | set(y_test)),
        'class_distribution_train': dict(Counter(y_train)),
        'class_distribution_test': dict(Counter(y_test)),
        'metrics': {
            'accuracy': acc,
            'macro_f1': macro_f1,
        },
        'model_path': str(model_path),
        'seed': random_state,
        'test_size': test_size,
    }

    (out_dir / 'classifier_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if X_test:
        (out_dir / 'classification_report.txt').write_text(
            classification_report(y_test, y_pred, digits=4),
            encoding='utf-8',
        )

        cm_rows = ['label,' + ','.join(labels)]
        for i, lab in enumerate(labels):
            cm_rows.append(lab + ',' + ','.join(str(int(x)) for x in cm[i]))
        (out_dir / 'confusion_matrix.csv').write_text('\n'.join(cm_rows) + '\n', encoding='utf-8')
    else:
        (out_dir / 'classification_report.txt').write_text('No eval split provided.\n', encoding='utf-8')
        (out_dir / 'confusion_matrix.csv').write_text('label\n', encoding='utf-8')

    print(f"[OK] model: {model_path}")
    if acc is not None:
        print(f"[OK] accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    else:
        print("[OK] trained without eval split")
    print(f"[OK] report: {out_dir / 'classifier_report.json'}")


if __name__ == '__main__':
    main()
