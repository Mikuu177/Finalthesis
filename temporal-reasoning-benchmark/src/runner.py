import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import csv
from typing import Dict, Any, List

from .io_utils import read_yaml, load_dataset, ensure_dir, make_run_dir, append_jsonl, load_existing_predictions, write_jsonl
from .prompt_builder import build_prompt, get_hint_policy_signature
from .scorer import extract_final_answer, score, FINAL_PAT
from .adapters import build_adapter
from .normalize import (
    normalize_model_answer,
    is_label_answer,
    is_numeric_answer,
    is_time_answer,
    contains_time_fragment,
    is_date_answer,
    contains_date_fragment,
    contains_tz_token,
)


def load_models(models_cfg_path: str | Path, required_models: list[str]):
    cfg = read_yaml(models_cfg_path)
    models_cfg = cfg.get('models', {})
    adapters = {}
    for name in required_models:
        mcfg = models_cfg.get(name)
        if not mcfg:
            print(f"[WARN] Model '{name}' not defined in {models_cfg_path}, skip.")
            continue
        adapters[name] = build_adapter(name, mcfg)
    return adapters


def _filter_and_sample(samples: List[Dict[str, Any]], filter_category: List[str] | None, sample_size: int | None, seed: int):
    import random
    if filter_category:
        allow = set(filter_category)
        samples = [s for s in samples if s.get('category') in allow]
    if sample_size and sample_size > 0 and len(samples) > sample_size:
        rnd = random.Random(seed)
        idxs = list(range(len(samples)))
        rnd.shuffle(idxs)
        idxs = idxs[:sample_size]
        samples = [samples[i] for i in idxs]
    return samples


def _extract_rate_from_pred_norm(pred_norm: str) -> bool:
    return bool(is_time_answer(pred_norm) or is_date_answer(pred_norm) or is_numeric_answer(pred_norm))


def run(config_path: str, models_cfg_path: str, prompts_cfg_path: str):
    import json, hashlib, subprocess, time
    cfg = read_yaml(config_path)
    models_cfg = read_yaml(models_cfg_path)
    prompts_cfg = read_yaml(prompts_cfg_path)

    seed = int(cfg.get('seed', 42))
    dataset_path = cfg['dataset_path']
    subset_size = cfg.get('subset_size')

    filter_category = cfg.get('filter_category')
    if isinstance(filter_category, str):
        filter_category = [filter_category]

    sample_size = cfg.get('sample_size')

    prompt_mode = cfg.get('prompt_mode', 'sp')
    n_shots = int(cfg.get('n_shots', 0))
    prompt_dir = cfg.get('prompt_dir', 'prompts')
    prompt_version = cfg.get('prompt_version', 'v1')

    hint_sig = get_hint_policy_signature('v1')

    outputs_dir = cfg.get('outputs_dir', 'outputs/runs')
    cache_dir = cfg.get('cache_dir', '.cache')
    resume = bool(cfg.get('resume', True))

    params = cfg.get('params', {})

    models = cfg.get('models', [])
    adapters_all = load_models(models_cfg_path, models)

    samples = load_dataset(dataset_path, subset_size=subset_size)

    # propagate prompt_version to prompt builder (for v2.2 template override)
    for s in samples:
        s['_prompt_version'] = prompt_version

    samples = _filter_and_sample(samples, filter_category, sample_size, seed)

    run_name_base = cfg.get('run_name') or 'run'
    run_name = f"{run_name_base}_pv{prompt_version}"

    run_dir = make_run_dir(outputs_dir, run_name)
    pred_path = Path(run_dir) / 'predictions.jsonl'
    summary_path = Path(run_dir) / 'summary.csv'

    manifest_path = Path(run_dir) / 'sample_manifest.jsonl'
    write_jsonl(manifest_path, [{'id': str(s.get('id')), 'category': s.get('category', 'unspecified')} for s in samples])

    snapshot = {
        'config': cfg,
        'models_cfg': models_cfg,
        'prompts_cfg': prompts_cfg,
    }
    snap_path = Path(run_dir) / 'config_snapshot.yaml'
    snap_path.write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True), encoding='utf-8')

    run_id = run_dir.name
    snap_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    run_config_hash = hashlib.sha256(snap_json.encode('utf-8')).hexdigest()

    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = 'unknown'

    start_time = time.time()

    existing = load_existing_predictions(pred_path) if resume else {}

    per_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    per_contract: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_extract: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    per_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_numeric: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_time: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_contains_time: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    per_date: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_contains_date: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    per_tz_token: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    per_mapped_correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for model_name in models:
        adapter = adapters_all.get(model_name)
        if adapter is None:
            print(f"[WARN] Model '{model_name}' not found in {models_cfg_path}, skip.")
            continue

        for sample in samples:
            sid = str(sample['id'])
            key = f"{model_name}::{sid}"
            if resume and key in existing:
                continue

            full_prompt, prompt_template, exemplar_ids = build_prompt(
                sample=sample,
                prompts_cfg_path=prompts_cfg_path,
                mode=prompt_mode,
                n_shots=n_shots,
                prompt_dir=prompt_dir,
                seed=seed,
            )

            result = adapter.generate([{"role": "user", "content": full_prompt}], params, cache_dir=cache_dir, cache_key_extra=sid)

            raw_text = result.text or ''
            pred = extract_final_answer(raw_text)
            gold = str(sample.get('gold', ''))

            pred_norm = normalize_model_answer(pred, sample)
            corr, info = score(pred_norm, gold, relaxed=False)

            cat = sample.get('category', 'unspecified')

            contract_ok = bool(FINAL_PAT.search(raw_text))
            extract_ok = _extract_rate_from_pred_norm(pred_norm)

            per_counts[model_name][cat] += 1
            if corr:
                per_correct[model_name][cat] += 1
            if contract_ok:
                per_contract[model_name][cat] += 1
            if extract_ok:
                per_extract[model_name][cat] += 1

            if is_label_answer(pred):
                per_label[model_name][cat] += 1
            if is_numeric_answer(pred):
                per_numeric[model_name][cat] += 1
            if is_time_answer(pred_norm):
                per_time[model_name][cat] += 1
            if contains_time_fragment(raw_text):
                per_contains_time[model_name][cat] += 1

            if is_date_answer(pred_norm):
                per_date[model_name][cat] += 1
            if contains_date_fragment(raw_text):
                per_contains_date[model_name][cat] += 1

            if contains_tz_token(raw_text):
                per_tz_token[model_name][cat] += 1

            if is_label_answer(pred) and corr:
                per_mapped_correct[model_name][cat] += 1

            prompt_fingerprint = hashlib.sha256(full_prompt.encode('utf-8')).hexdigest()

            row = {
                'id': sid,
                'model': model_name,
                'category': cat,
                'prompt_version': prompt_version,
                'prompt_template': prompt_template,
                'exemplar_ids': exemplar_ids or [],
                'prompt_fingerprint': prompt_fingerprint,
                'run_config_hash': run_config_hash,
                'hint_policy_version': hint_sig['hint_policy_version'],
                'hint_policy_hash': hint_sig['hint_policy_hash'],
                'prompt': full_prompt,
                'raw': raw_text,
                'pred': pred,
                'pred_norm': pred_norm,
                'gold': gold,
                'correct': bool(corr),
                'contract_ok': contract_ok,
                'extract_ok': extract_ok,
                'match': info.get('match'),
                'usage': result.usage,
                'latency': result.latency,
                'error': result.error,
            }
            append_jsonl(pred_path, row)

    ensure_dir(run_dir)
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'category', 'n', 'correct', 'accuracy',
            'contract_rate', 'extract_rate', 'parse_rate',
            'time_rate', 'contains_time_rate',
            'date_rate', 'contains_date_rate',
            'numeric_rate', 'label_rate', 'text_rate',
            'mapped_correct_rate',
            'tz_token_rate',
            'prompt_version', 'hint_policy_version', 'hint_policy_hash'
        ])
        for model_name, cats in per_counts.items():
            for cat, n in cats.items():
                c = per_correct[model_name].get(cat, 0)
                acc = c / n if n else 0.0

                contract_rate = per_contract[model_name].get(cat, 0) / n if n else 0.0
                extract_rate = per_extract[model_name].get(cat, 0) / n if n else 0.0
                parse_rate = extract_rate

                tr = per_time[model_name].get(cat, 0) / n if n else 0.0
                ctr = per_contains_time[model_name].get(cat, 0) / n if n else 0.0

                dr = per_date[model_name].get(cat, 0) / n if n else 0.0
                cdr = per_contains_date[model_name].get(cat, 0) / n if n else 0.0

                nr = per_numeric[model_name].get(cat, 0) / n if n else 0.0
                lr = per_label[model_name].get(cat, 0) / n if n else 0.0

                tzr = per_tz_token[model_name].get(cat, 0) / n if n else 0.0

                max_shape = max(tr, ctr, dr, cdr, nr, lr)
                text_rate = 1.0 - max_shape

                mcr = per_mapped_correct[model_name].get(cat, 0) / n if n else 0.0

                writer.writerow([
                    model_name, cat, n, c, f"{acc:.4f}",
                    f"{contract_rate:.4f}", f"{extract_rate:.4f}", f"{parse_rate:.4f}",
                    f"{tr:.4f}", f"{ctr:.4f}",
                    f"{dr:.4f}", f"{cdr:.4f}",
                    f"{nr:.4f}", f"{lr:.4f}", f"{text_rate:.4f}",
                    f"{mcr:.4f}",
                    f"{tzr:.4f}",
                    prompt_version, hint_sig['hint_policy_version'], hint_sig['hint_policy_hash']
                ])

    end_time = time.time()
    meta = {
        'run_id': run_id,
        'run_config_hash': run_config_hash,
        'git_commit': git_commit,
        'models': models,
        'prompt_mode': prompt_mode,
        'n_shots': n_shots,
        'seed': seed,
        'dataset_path': dataset_path,
        'subset_size': subset_size,
        'filter_category': filter_category,
        'sample_size': sample_size,
        'prompt_version': prompt_version,
        'alias_name': run_name,
        'hint_policy_version': hint_sig['hint_policy_version'],
        'hint_policy_hash': hint_sig['hint_policy_hash'],
        'start_time': start_time,
        'end_time': end_time,
        'elapsed_sec': end_time - start_time,
    }
    (Path(run_dir) / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Done. Predictions: {pred_path}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to baseline.yaml')
    parser.add_argument('--models', required=False, default='configs/models.yaml', help='Path to models.yaml')
    parser.add_argument('--prompts', required=False, default='configs/prompts.yaml', help='Path to prompts.yaml')
    args = parser.parse_args()

    run(config_path=args.config, models_cfg_path=args.models, prompts_cfg_path=args.prompts)
