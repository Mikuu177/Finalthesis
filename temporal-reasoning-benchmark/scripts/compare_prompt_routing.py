import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _safe_float(v) -> Optional[float]:
    if v in (None, ''):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v) -> Optional[int]:
    if v in (None, ''):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _load_pred_metrics(pred_path: Path) -> Dict[str, Optional[float]]:
    if not pred_path.exists():
        return {
            'sample_count': 0,
            'total_calls': 0,
            'total_latency': 0.0,
            'correct_n': 0,
            'class_match_n': None,
            'class_acc': None,
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
        }

    sample_ids = set()
    total_calls = 0
    total_latency = 0.0
    correct_n = 0
    class_match_n = 0
    has_class = False

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    has_usage = False

    with pred_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get('id', ''))
            if sid:
                sample_ids.add(sid)

            calls = _safe_int(row.get('calls_used'))
            total_calls += calls if calls and calls > 0 else 1
            total_latency += float(row.get('latency') or 0.0)
            correct_n += int(bool(row.get('correct')))

            if 'category_match' in row:
                has_class = True
                class_match_n += int(bool(row.get('category_match')))

            usage = row.get('usage') or {}
            pt = _safe_int(usage.get('prompt_tokens'))
            ct = _safe_int(usage.get('completion_tokens'))
            tt = _safe_int(usage.get('total_tokens'))
            if pt is not None:
                has_usage = True
                prompt_tokens += pt
            if ct is not None:
                has_usage = True
                completion_tokens += ct
            if tt is not None:
                has_usage = True
                total_tokens += tt

    n = len(sample_ids)
    class_acc = (class_match_n / n) if (has_class and n > 0) else None

    return {
        'sample_count': n,
        'total_calls': total_calls,
        'total_latency': total_latency,
        'correct_n': correct_n,
        'class_match_n': class_match_n if has_class else None,
        'class_acc': class_acc,
        'prompt_tokens': prompt_tokens if has_usage else None,
        'completion_tokens': completion_tokens if has_usage else None,
        'total_tokens': total_tokens if has_usage else None,
    }


def _fmt(x: Optional[float], d: int = 4) -> str:
    if x is None:
        return ''
    return f"{x:.{d}f}"


def _weighted_metric(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    num = 0.0
    den = 0
    for r in rows:
        n = _safe_int(r.get('n')) or 0
        x = _safe_float(r.get(key))
        if n <= 0 or x is None:
            continue
        num += x * n
        den += n
    if den == 0:
        return None
    return num / den


def summarize_run(run_dir: Path) -> Dict[str, str]:
    meta_path = run_dir / 'run_metadata.json'
    pred_path = run_dir / 'predictions.jsonl'
    summary_path = run_dir / 'summary.csv'

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))

    metrics = _load_pred_metrics(pred_path)
    n = int(metrics['sample_count'] or 0)
    acc = (metrics['correct_n'] / n) if n else None
    calls_per_q = (metrics['total_calls'] / n) if n else None
    lat_per_q = (metrics['total_latency'] / n) if n else None
    calls_per_correct = (metrics['total_calls'] / metrics['correct_n']) if metrics['correct_n'] else None
    lat_per_correct = (metrics['total_latency'] / metrics['correct_n']) if metrics['correct_n'] else None

    parse_rate = None
    format_compliance = None
    if summary_path.exists():
        with summary_path.open('r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            parse_rate = _weighted_metric(rows, 'parse_rate')
            format_compliance = _weighted_metric(rows, 'contract_rate')

    return {
        'run_id': run_dir.name,
        'workflow_type': str(meta.get('workflow_type', 'unknown')),
        'router_type': str(meta.get('router_type', '')),
        'sample_count': str(n),
        'accuracy': _fmt(acc),
        'parse_rate': _fmt(parse_rate),
        'format_compliance': _fmt(format_compliance),
        'classifier_category_accuracy': _fmt(metrics.get('class_acc')),
        'fallback_rate': _fmt(_safe_float(meta.get('fallback_rate'))),
        'latency_sec_per_query': _fmt(lat_per_q),
        'calls_per_query': _fmt(calls_per_q),
        'calls_per_correct': _fmt(calls_per_correct),
        'latency_sec_per_correct': _fmt(lat_per_correct),
        'total_tokens': str(metrics.get('total_tokens') or ''),
        'prompt_tokens': str(metrics.get('prompt_tokens') or ''),
        'completion_tokens': str(metrics.get('completion_tokens') or ''),
    }


def main():
    ap = argparse.ArgumentParser(description='Compare fixed prompt baselines and classifier-routed prompt runs')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', default='outputs/tables/prompt_routing_comparison.csv')
    args = ap.parse_args()

    run_dirs = [Path(r).resolve() for r in args.runs]
    rows = [summarize_run(rd) for rd in run_dirs]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        'run_id', 'workflow_type', 'router_type', 'sample_count', 'accuracy', 'parse_rate', 'format_compliance',
        'classifier_category_accuracy', 'fallback_rate', 'latency_sec_per_query', 'calls_per_query', 'calls_per_correct',
        'latency_sec_per_correct', 'total_tokens', 'prompt_tokens', 'completion_tokens'
    ]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote: {out_path}")
    for r in rows:
        print(
            f"- {r['run_id']} | workflow={r['workflow_type']} | acc={r['accuracy']} | "
            f"cls_acc={r['classifier_category_accuracy'] or '-'} | calls/query={r['calls_per_query']}"
        )


if __name__ == '__main__':
    main()
