import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import append_jsonl, ensure_dir, load_dataset, make_run_dir, read_yaml, write_jsonl
from src.normalize import normalize_model_answer, is_date_answer, is_numeric_answer, is_time_answer
from src.prompt_builder import build_prompt, get_hint_policy_signature
from src.runner import load_models
from src.scorer import FINAL_PAT, extract_final_answer, score


def _load_classifier(path: Path):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError('Missing dependency: joblib is required for classifier loading.') from e
    if not path.exists():
        raise FileNotFoundError(f'Classifier model not found: {path}')
    return joblib.load(path)


def _pick_prompt(prompt_bank: Dict[str, Any], pred_category: str) -> Dict[str, Any]:
    by_cat = prompt_bank.get('by_category', {}) or {}
    default = prompt_bank.get('default', {}) or {}
    cfg = by_cat.get(pred_category, default)
    if not cfg:
        return {'prompt_mode': 'sp', 'n_shots': 0, 'prompt_version': 'v2.1'}
    return {
        'prompt_mode': cfg.get('prompt_mode', 'sp'),
        'n_shots': int(cfg.get('n_shots', 0)),
        'prompt_version': cfg.get('prompt_version', 'v2.1'),
    }


def _extract_ok(pred_norm: str) -> bool:
    return bool(is_time_answer(pred_norm) or is_date_answer(pred_norm) or is_numeric_answer(pred_norm))


def run(config_path: str, models_cfg_path: str, prompts_cfg_path: str):
    cfg = read_yaml(config_path)
    models_cfg = read_yaml(models_cfg_path)
    prompts_cfg = read_yaml(prompts_cfg_path)

    seed = int(cfg.get('seed', 42))
    dataset_path = cfg['dataset_path']
    subset_size = cfg.get('subset_size')
    sample_size = cfg.get('sample_size')
    filter_category = cfg.get('filter_category')
    if isinstance(filter_category, str):
        filter_category = [filter_category]

    params = cfg.get('params', {})
    outputs_dir = cfg.get('outputs_dir', 'outputs/runs')
    cache_dir = cfg.get('cache_dir', '.cache')
    fallback_cfg = cfg.get('fallback', {}) or {}
    fallback_enabled = bool(fallback_cfg.get('enabled', False))
    fallback_min_confidence = float(fallback_cfg.get('min_confidence', 0.0))
    fallback_on_missing_conf = bool(fallback_cfg.get('on_missing_confidence', False))
    fallback_prompt_cfg = fallback_cfg.get('prompt') or {}

    models = cfg.get('models', [])
    if len(models) != 1:
        raise ValueError('prompt router currently requires exactly one serving model in `models`.')
    serving_model = models[0]

    classifier_model_path = Path(cfg['classifier_model_path'])
    if not classifier_model_path.is_absolute():
        classifier_model_path = (PROJECT_ROOT / classifier_model_path).resolve()

    prompt_bank_path = Path(cfg['prompt_bank_path'])
    if not prompt_bank_path.is_absolute():
        prompt_bank_path = (PROJECT_ROOT / prompt_bank_path).resolve()

    classifier = _load_classifier(classifier_model_path)
    prompt_bank = yaml.safe_load(prompt_bank_path.read_text(encoding='utf-8'))

    adapters_all = load_models(models_cfg_path, models)
    adapter = adapters_all.get(serving_model)
    if adapter is None:
        raise RuntimeError(f'Model {serving_model} is not available in models config.')

    samples = load_dataset(dataset_path, subset_size=subset_size)
    if filter_category:
        allow = set(filter_category)
        samples = [s for s in samples if s.get('category') in allow]
    if sample_size and sample_size > 0 and len(samples) > sample_size:
        import random

        rnd = random.Random(seed)
        idxs = list(range(len(samples)))
        rnd.shuffle(idxs)
        samples = [samples[i] for i in idxs[:sample_size]]

    run_name_base = cfg.get('run_name') or 'prompt_router_eval'
    run_dir = make_run_dir(outputs_dir, f"{run_name_base}_pvv2.1")

    pred_path = Path(run_dir) / 'predictions.jsonl'
    summary_path = Path(run_dir) / 'summary.csv'
    route_summary_path = Path(run_dir) / 'route_summary.csv'

    write_jsonl(Path(run_dir) / 'sample_manifest.jsonl', [{'id': str(s.get('id')), 'category': s.get('category', 'unspecified')} for s in samples])

    snapshot = {
        'config': cfg,
        'models_cfg': models_cfg,
        'prompts_cfg': prompts_cfg,
        'workflow_type': 'classifier_prompt_router',
        'router_type': 'task_classifier_prompt_bank',
    }
    (Path(run_dir) / 'config_snapshot.yaml').write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True), encoding='utf-8')
    run_config_hash = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

    hint_sig = get_hint_policy_signature('v1')

    n_total = 0
    n_correct = 0
    n_contract = 0
    n_extract = 0
    n_class_correct = 0
    n_fallback = 0
    total_latency = 0.0
    total_calls = 0

    route_counts: Dict[str, int] = {}

    start = time.time()

    for s in samples:
        sid = str(s['id'])
        true_cat = str(s.get('category', 'unspecified'))
        q = str(s.get('question', '') or '')
        ctx = str(s.get('context', '') or '')
        clf_text = f"{q}\n{ctx}" if ctx else q

        pred_cat = str(classifier.predict([clf_text])[0])
        if pred_cat == true_cat:
            n_class_correct += 1

        clf_conf = None
        if hasattr(classifier, 'predict_proba'):
            try:
                proba = classifier.predict_proba([clf_text])[0]
                clf_conf = float(max(proba))
            except Exception:
                clf_conf = None

        psel = _pick_prompt(prompt_bank, pred_cat)
        used_fallback = False
        fallback_reason = ''
        if fallback_enabled:
            should_fallback = False
            if clf_conf is None and fallback_on_missing_conf:
                should_fallback = True
                fallback_reason = 'missing_confidence'
            elif clf_conf is not None and clf_conf < fallback_min_confidence:
                should_fallback = True
                fallback_reason = 'low_confidence'
            if should_fallback:
                used_fallback = True
                n_fallback += 1
                psel = {
                    'prompt_mode': fallback_prompt_cfg.get('prompt_mode', prompt_bank.get('default', {}).get('prompt_mode', 'sp')),
                    'n_shots': int(fallback_prompt_cfg.get('n_shots', prompt_bank.get('default', {}).get('n_shots', 0))),
                    'prompt_version': fallback_prompt_cfg.get('prompt_version', prompt_bank.get('default', {}).get('prompt_version', 'v2.1')),
                }

        prompt_sample = dict(s)
        prompt_sample['category'] = pred_cat
        prompt_sample['_prompt_version'] = psel['prompt_version']

        full_prompt, prompt_template, exemplar_ids = build_prompt(
            sample=prompt_sample,
            prompts_cfg_path=prompts_cfg_path,
            mode=psel['prompt_mode'],
            n_shots=psel['n_shots'],
            prompt_dir=cfg.get('prompt_dir', 'prompts'),
            seed=seed,
        )

        result = adapter.generate(
            [{"role": "user", "content": full_prompt}],
            params,
            cache_dir=cache_dir,
            cache_key_extra=f"{sid}::{pred_cat}::{psel['prompt_mode']}::{psel['n_shots']}",
        )

        raw = result.text or ''
        pred = extract_final_answer(raw)
        pred_norm = normalize_model_answer(pred, s)
        gold = str(s.get('gold', ''))
        corr, info = score(pred_norm, gold, relaxed=False)

        contract_ok = bool(FINAL_PAT.search(raw))
        extract_ok = _extract_ok(pred_norm)

        n_total += 1
        n_correct += int(bool(corr))
        n_contract += int(contract_ok)
        n_extract += int(extract_ok)
        total_latency += float(result.latency or 0.0)
        total_calls += 1

        rk = f"{pred_cat}|{psel['prompt_mode']}_{psel['n_shots']}"
        route_counts[rk] = route_counts.get(rk, 0) + 1

        row = {
            'id': sid,
            'model': serving_model,
            'workflow_type': 'classifier_prompt_router',
            'router_type': 'task_classifier_prompt_bank',
            'true_category': true_cat,
            'predicted_category': pred_cat,
            'category_match': pred_cat == true_cat,
            'classifier_confidence': clf_conf,
            'used_fallback': used_fallback,
            'fallback_reason': fallback_reason,
            'prompt_mode': psel['prompt_mode'],
            'n_shots': psel['n_shots'],
            'prompt_version': psel['prompt_version'],
            'prompt_template': prompt_template,
            'exemplar_ids': exemplar_ids or [],
            'run_config_hash': run_config_hash,
            'hint_policy_version': hint_sig['hint_policy_version'],
            'hint_policy_hash': hint_sig['hint_policy_hash'],
            'prompt': full_prompt,
            'raw': raw,
            'pred': pred,
            'pred_norm': pred_norm,
            'gold': gold,
            'correct': bool(corr),
            'contract_ok': contract_ok,
            'extract_ok': extract_ok,
            'match': info.get('match'),
            'usage': result.usage,
            'latency': result.latency,
            'calls_used': 1,
            'error': result.error,
        }
        append_jsonl(pred_path, row)

    elapsed = time.time() - start

    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'model', 'category', 'n', 'correct', 'accuracy', 'contract_rate', 'extract_rate', 'parse_rate',
            'classifier_category_accuracy', 'workflow_type', 'router_type'
        ])
        acc = (n_correct / n_total) if n_total else 0.0
        cr = (n_contract / n_total) if n_total else 0.0
        er = (n_extract / n_total) if n_total else 0.0
        carr = (n_class_correct / n_total) if n_total else 0.0
        w.writerow([
            serving_model,
            'all',
            n_total,
            n_correct,
            f"{acc:.4f}",
            f"{cr:.4f}",
            f"{er:.4f}",
            f"{er:.4f}",
            f"{carr:.4f}",
            'classifier_prompt_router',
            'task_classifier_prompt_bank',
        ])

    with open(route_summary_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['route_key', 'count'])
        for k, v in sorted(route_counts.items(), key=lambda x: x[0]):
            w.writerow([k, v])

    meta = {
        'run_id': Path(run_dir).name,
        'run_name': run_name_base,
        'workflow_type': 'classifier_prompt_router',
        'router_type': 'task_classifier_prompt_bank',
        'model': serving_model,
        'prompt_mode': 'dynamic',
        'n_shots': 'dynamic',
        'elapsed_sec': elapsed,
        'sample_count': n_total,
        'correct_count': n_correct,
        'accuracy': (n_correct / n_total) if n_total else 0.0,
        'parse_rate': (n_extract / n_total) if n_total else 0.0,
        'format_compliance': (n_contract / n_total) if n_total else 0.0,
        'classifier_category_accuracy': (n_class_correct / n_total) if n_total else 0.0,
        'fallback_count': n_fallback,
        'fallback_rate': (n_fallback / n_total) if n_total else 0.0,
        'fallback': {
            'enabled': fallback_enabled,
            'min_confidence': fallback_min_confidence,
            'on_missing_confidence': fallback_on_missing_conf,
            'prompt': {
                'prompt_mode': fallback_prompt_cfg.get('prompt_mode', prompt_bank.get('default', {}).get('prompt_mode', 'sp')),
                'n_shots': int(fallback_prompt_cfg.get('n_shots', prompt_bank.get('default', {}).get('n_shots', 0))),
                'prompt_version': fallback_prompt_cfg.get('prompt_version', prompt_bank.get('default', {}).get('prompt_version', 'v2.1')),
            },
        },
        'calls_total': total_calls,
        'latency_total_sec': total_latency,
        'run_config_hash': run_config_hash,
        'config_path': str(Path(config_path).resolve()),
    }
    (Path(run_dir) / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] accuracy={(n_correct / n_total if n_total else 0):.4f}, classifier_acc={(n_class_correct / n_total if n_total else 0):.4f}")


def main():
    ap = argparse.ArgumentParser(description='Classifier-driven prompt router runner')
    ap.add_argument('--config', required=True)
    ap.add_argument('--models_cfg', default='configs/models.yaml')
    ap.add_argument('--prompts_cfg', default='configs/prompts.yaml')
    args = ap.parse_args()

    run(args.config, args.models_cfg, args.prompts_cfg)


if __name__ == '__main__':
    main()
