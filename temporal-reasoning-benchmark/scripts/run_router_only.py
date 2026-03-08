import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import append_jsonl, ensure_dir, load_dataset, make_run_dir, read_yaml, write_jsonl
from src.normalize import (
    contains_date_fragment,
    contains_time_fragment,
    is_date_answer,
    is_label_answer,
    is_numeric_answer,
    is_time_answer,
    normalize_model_answer,
)
from src.prompt_builder import build_prompt, get_hint_policy_signature
from src.runner import load_models
from src.scorer import FINAL_PAT, extract_final_answer, score


def _load_profiles(path: str | Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = raw.get("profiles", [])
    out = {}
    for p in profiles:
        model = p.get("model")
        if model:
            out[model] = p
    return out


def _filter_and_sample(
    samples: List[Dict[str, Any]],
    filter_category: Optional[List[str]],
    sample_size: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    import random

    if filter_category:
        allow = set(filter_category)
        samples = [s for s in samples if s.get("category") in allow]
    if sample_size and sample_size > 0 and len(samples) > sample_size:
        rnd = random.Random(seed)
        idxs = list(range(len(samples)))
        rnd.shuffle(idxs)
        idxs = idxs[:sample_size]
        samples = [samples[i] for i in idxs]
    return samples


def _pick_model(
    sample_category: str,
    candidate_models: List[str],
    profiles: Dict[str, Dict[str, Any]],
) -> Tuple[str, float, str]:
    """
    Return: (model_name, route_score, score_source)
    score_source: "category_accuracy" or "overall_accuracy"
    """
    best_model = None
    best_score = -1.0
    best_source = "overall_accuracy"

    for model in candidate_models:
        p = profiles.get(model)
        if not p:
            continue
        cat_acc = p.get("category_accuracy", {}).get(sample_category)
        overall = p.get("overall_accuracy", 0.0)

        if cat_acc is not None:
            score_value = float(cat_acc)
            source = "category_accuracy"
        else:
            score_value = float(overall or 0.0)
            source = "overall_accuracy"

        if score_value > best_score:
            best_score = score_value
            best_model = model
            best_source = source

    if best_model is None:
        raise RuntimeError("Router cannot pick a model: no candidate has profile.")
    return best_model, best_score, best_source


def _extract_rate_from_pred_norm(pred_norm: str) -> bool:
    return bool(is_time_answer(pred_norm) or is_date_answer(pred_norm) or is_numeric_answer(pred_norm))


def run(config_path: str, models_cfg_path: str, prompts_cfg_path: str):
    cfg = read_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    dataset_path = cfg["dataset_path"]
    subset_size = cfg.get("subset_size")
    filter_category = cfg.get("filter_category")
    if isinstance(filter_category, str):
        filter_category = [filter_category]
    sample_size = cfg.get("sample_size")

    prompt_mode = cfg.get("prompt_mode", "sp")
    n_shots = int(cfg.get("n_shots", 0))
    prompt_dir = cfg.get("prompt_dir", "prompts")
    prompt_version = cfg.get("prompt_version", "v1")

    outputs_dir = cfg.get("outputs_dir", "outputs/runs")
    cache_dir = cfg.get("cache_dir", ".cache")
    params = cfg.get("params", {})

    profile_path = cfg.get("profile_path", "outputs/profiles/profiles_v1.json")
    candidate_models = cfg.get("models", [])
    if not candidate_models:
        raise ValueError("`models` is required in router config.")

    profiles = _load_profiles(profile_path)
    adapters_all = load_models(models_cfg_path, candidate_models)

    usable_models = [m for m in candidate_models if m in adapters_all and m in profiles]
    if not usable_models:
        raise RuntimeError("No usable models after intersecting config/models/profile.")

    samples = load_dataset(dataset_path, subset_size=subset_size)
    for s in samples:
        s["_prompt_version"] = prompt_version
    samples = _filter_and_sample(samples, filter_category, sample_size, seed)

    run_name_base = cfg.get("run_name") or "router_only"
    run_name = f"{run_name_base}_pv{prompt_version}"
    run_dir = make_run_dir(outputs_dir, run_name)

    pred_path = Path(run_dir) / "predictions.jsonl"
    summary_path = Path(run_dir) / "summary.csv"
    route_summary_path = Path(run_dir) / "route_summary.csv"

    manifest_path = Path(run_dir) / "sample_manifest.jsonl"
    write_jsonl(manifest_path, [{"id": str(s.get("id")), "category": s.get("category", "unspecified")} for s in samples])

    snapshot = {
        "config": cfg,
        "models_cfg_path": models_cfg_path,
        "prompts_cfg_path": prompts_cfg_path,
        "router_type": "profile_category_accuracy",
        "usable_models": usable_models,
    }
    (Path(run_dir) / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )

    run_config_hash = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    hint_sig = get_hint_policy_signature("v1")

    per_counts = defaultdict(lambda: defaultdict(int))
    per_correct = defaultdict(lambda: defaultdict(int))
    per_contract = defaultdict(lambda: defaultdict(int))
    per_extract = defaultdict(lambda: defaultdict(int))

    per_label = defaultdict(lambda: defaultdict(int))
    per_numeric = defaultdict(lambda: defaultdict(int))
    per_time = defaultdict(lambda: defaultdict(int))
    per_contains_time = defaultdict(lambda: defaultdict(int))
    per_date = defaultdict(lambda: defaultdict(int))
    per_contains_date = defaultdict(lambda: defaultdict(int))

    route_counts = defaultdict(int)
    route_score_sum = defaultdict(float)
    route_source_counts = defaultdict(lambda: defaultdict(int))

    start_time = time.time()

    for sample in samples:
        sid = str(sample["id"])
        category = sample.get("category", "unspecified")

        routed_model, route_score, route_source = _pick_model(category, usable_models, profiles)
        adapter = adapters_all[routed_model]

        full_prompt, prompt_template, exemplar_ids = build_prompt(
            sample=sample,
            prompts_cfg_path=prompts_cfg_path,
            mode=prompt_mode,
            n_shots=n_shots,
            prompt_dir=prompt_dir,
            seed=seed,
        )

        result = adapter.generate(
            [{"role": "user", "content": full_prompt}],
            params,
            cache_dir=cache_dir,
            cache_key_extra=sid,
        )

        raw_text = result.text or ""
        pred = extract_final_answer(raw_text)
        gold = str(sample.get("gold", ""))
        pred_norm = normalize_model_answer(pred, sample)
        corr, info = score(pred_norm, gold, relaxed=False)

        contract_ok = bool(FINAL_PAT.search(raw_text))
        extract_ok = _extract_rate_from_pred_norm(pred_norm)

        route_counts[routed_model] += 1
        route_score_sum[routed_model] += route_score
        route_source_counts[routed_model][route_source] += 1

        per_counts[routed_model][category] += 1
        if corr:
            per_correct[routed_model][category] += 1
        if contract_ok:
            per_contract[routed_model][category] += 1
        if extract_ok:
            per_extract[routed_model][category] += 1

        if is_label_answer(pred):
            per_label[routed_model][category] += 1
        if is_numeric_answer(pred):
            per_numeric[routed_model][category] += 1
        if is_time_answer(pred_norm):
            per_time[routed_model][category] += 1
        if contains_time_fragment(raw_text):
            per_contains_time[routed_model][category] += 1
        if is_date_answer(pred_norm):
            per_date[routed_model][category] += 1
        if contains_date_fragment(raw_text):
            per_contains_date[routed_model][category] += 1

        row = {
            "id": sid,
            "model": routed_model,
            "routed_model": routed_model,
            "router_type": "profile_category_accuracy",
            "route_score": route_score,
            "route_source": route_source,
            "category": category,
            "prompt_version": prompt_version,
            "prompt_template": prompt_template,
            "exemplar_ids": exemplar_ids or [],
            "run_config_hash": run_config_hash,
            "hint_policy_version": hint_sig["hint_policy_version"],
            "hint_policy_hash": hint_sig["hint_policy_hash"],
            "prompt": full_prompt,
            "raw": raw_text,
            "pred": pred,
            "pred_norm": pred_norm,
            "gold": gold,
            "correct": bool(corr),
            "contract_ok": contract_ok,
            "extract_ok": extract_ok,
            "match": info.get("match"),
            "usage": result.usage,
            "latency": result.latency,
            "error": result.error,
        }
        append_jsonl(pred_path, row)

    elapsed = time.time() - start_time

    ensure_dir(run_dir)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "category",
                "n",
                "correct",
                "accuracy",
                "contract_rate",
                "extract_rate",
                "parse_rate",
                "time_rate",
                "contains_time_rate",
                "date_rate",
                "contains_date_rate",
                "numeric_rate",
                "label_rate",
                "text_rate",
                "prompt_version",
                "router_type",
            ]
        )
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

                max_shape = max(tr, ctr, dr, cdr, nr, lr)
                text_rate = 1.0 - max_shape

                writer.writerow(
                    [
                        model_name,
                        cat,
                        n,
                        c,
                        f"{acc:.4f}",
                        f"{contract_rate:.4f}",
                        f"{extract_rate:.4f}",
                        f"{parse_rate:.4f}",
                        f"{tr:.4f}",
                        f"{ctr:.4f}",
                        f"{dr:.4f}",
                        f"{cdr:.4f}",
                        f"{nr:.4f}",
                        f"{lr:.4f}",
                        f"{text_rate:.4f}",
                        prompt_version,
                        "profile_category_accuracy",
                    ]
                )

    with open(route_summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "routed_count", "routed_share", "avg_route_score", "category_source_count", "overall_source_count"])
        total = sum(route_counts.values())
        for model_name, count in sorted(route_counts.items()):
            share = (count / total) if total else 0.0
            avg_score = (route_score_sum[model_name] / count) if count else 0.0
            cat_source = route_source_counts[model_name].get("category_accuracy", 0)
            ov_source = route_source_counts[model_name].get("overall_accuracy", 0)
            writer.writerow([model_name, count, f"{share:.4f}", f"{avg_score:.4f}", cat_source, ov_source])

    meta = {
        "run_id": Path(run_dir).name,
        "run_config_hash": run_config_hash,
        "router_type": "profile_category_accuracy",
        "profile_path": str(Path(profile_path).resolve()),
        "models": usable_models,
        "prompt_mode": prompt_mode,
        "n_shots": n_shots,
        "seed": seed,
        "dataset_path": dataset_path,
        "subset_size": subset_size,
        "filter_category": filter_category,
        "sample_size": sample_size,
        "prompt_version": prompt_version,
        "alias_name": run_name,
        "hint_policy_version": hint_sig["hint_policy_version"],
        "hint_policy_hash": hint_sig["hint_policy_hash"],
        "elapsed_sec": elapsed,
    }
    (Path(run_dir) / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Predictions: {pred_path}")
    print(f"Summary: {summary_path}")
    print(f"Route summary: {route_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to router config yaml")
    parser.add_argument("--models", required=False, default="configs/models.yaml", help="Path to models.yaml")
    parser.add_argument("--prompts", required=False, default="configs/prompts.yaml", help="Path to prompts.yaml")
    args = parser.parse_args()

    run(config_path=args.config, models_cfg_path=args.models, prompts_cfg_path=args.prompts)
