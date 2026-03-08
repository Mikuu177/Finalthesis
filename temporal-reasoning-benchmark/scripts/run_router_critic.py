import argparse
import csv
import hashlib
import json
import re
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

VERDICT_PAT = re.compile(r"VERDICT\s*:\s*(ACCEPT|REJECT)", re.IGNORECASE)


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


def _pick_model(sample_category: str, candidate_models: List[str], profiles: Dict[str, Dict[str, Any]]) -> Tuple[str, float, str]:
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


def _build_critic_prompt(sample: Dict[str, Any], primary_text: str, primary_pred: str) -> str:
    q = sample.get("question", "")
    ctx = sample.get("context", "")
    category = sample.get("category", "unspecified")
    parts = [
        "You are a strict reviewer for temporal reasoning answers.",
        "Check whether the candidate answer is correct for the given task.",
        "Output exactly two required lines:",
        "VERDICT: ACCEPT or VERDICT: REJECT",
        "FINAL_ANSWER: <best final answer>",
        "No extra text after FINAL_ANSWER.",
        f"Category: {category}",
        f"Question: {q}",
    ]
    if ctx:
        parts.append(f"Context: {ctx}")
    parts.append(f"Candidate extracted answer: {primary_pred}")
    parts.append("Candidate full output:")
    parts.append(primary_text or "(empty)")
    return "\n".join(parts)


def _parse_critic(text: str) -> Tuple[str, str]:
    m = VERDICT_PAT.search(text or "")
    verdict = (m.group(1).upper() if m else "ACCEPT")
    answer = extract_final_answer(text or "")
    return verdict, answer


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
    critic_params = cfg.get("critic_params", params)
    critic_gate = cfg.get("critic_gate", {})
    gate_route_score_lt = float(critic_gate.get("route_score_lt", 0.95))
    gate_on_contract_fail = bool(critic_gate.get("on_contract_fail", True))
    gate_on_extract_fail = bool(critic_gate.get("on_extract_fail", True))
    gate_on_error = bool(critic_gate.get("on_error", True))

    profile_path = cfg.get("profile_path", "outputs/profiles/profiles_v1_clean.json")
    candidate_models = cfg.get("models", [])
    critic_model = cfg.get("critic_model")
    if not candidate_models:
        raise ValueError("`models` is required in router+critic config.")
    if not critic_model:
        raise ValueError("`critic_model` is required in router+critic config.")

    profiles = _load_profiles(profile_path)
    load_list = sorted(set(candidate_models + [critic_model]))
    adapters_all = load_models(models_cfg_path, load_list)
    usable_models = [m for m in candidate_models if m in adapters_all and m in profiles]
    if not usable_models:
        raise RuntimeError("No usable routed models after intersecting config/models/profile.")
    if critic_model not in adapters_all:
        raise RuntimeError(f"Critic model not available: {critic_model}")

    samples = load_dataset(dataset_path, subset_size=subset_size)
    for s in samples:
        s["_prompt_version"] = prompt_version
    samples = _filter_and_sample(samples, filter_category, sample_size, seed)

    run_name_base = cfg.get("run_name") or "router_critic"
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
        "workflow_type": "router_critic",
        "usable_models": usable_models,
        "critic_model": critic_model,
    }
    (Path(run_dir) / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    run_config_hash = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    hint_sig = get_hint_policy_signature("v1")

    n_total = 0
    n_correct = 0
    n_contract = 0
    n_extract = 0
    n_time = 0
    n_contains_time = 0
    n_date = 0
    n_contains_date = 0
    n_numeric = 0
    n_label = 0

    n_conflict = 0
    n_override = 0
    n_arb_success = 0
    n_critic_called = 0

    route_counts = defaultdict(int)
    route_score_sum = defaultdict(float)
    route_source_counts = defaultdict(lambda: defaultdict(int))

    start_time = time.time()
    critic_adapter = adapters_all[critic_model]

    for sample in samples:
        sid = str(sample["id"])
        category = sample.get("category", "unspecified")

        routed_model, route_score, route_source = _pick_model(category, usable_models, profiles)
        route_counts[routed_model] += 1
        route_score_sum[routed_model] += route_score
        route_source_counts[routed_model][route_source] += 1

        primary_adapter = adapters_all[routed_model]
        full_prompt, prompt_template, exemplar_ids = build_prompt(
            sample=sample,
            prompts_cfg_path=prompts_cfg_path,
            mode=prompt_mode,
            n_shots=n_shots,
            prompt_dir=prompt_dir,
            seed=seed,
        )
        primary = primary_adapter.generate(
            [{"role": "user", "content": full_prompt}],
            params,
            cache_dir=cache_dir,
            cache_key_extra=f"{sid}::primary::{routed_model}",
        )
        primary_raw = primary.text or ""
        primary_pred = extract_final_answer(primary_raw)
        primary_pred_norm = normalize_model_answer(primary_pred, sample)
        gold = str(sample.get("gold", ""))
        primary_corr, _ = score(primary_pred_norm, gold, relaxed=False)
        primary_contract_ok = bool(FINAL_PAT.search(primary_raw))
        primary_extract_ok = _extract_rate_from_pred_norm(primary_pred_norm)
        primary_has_error = bool(primary.error)

        gate_reasons: List[str] = []
        if route_score < gate_route_score_lt:
            gate_reasons.append("low_route_score")
        if gate_on_contract_fail and (not primary_contract_ok):
            gate_reasons.append("contract_fail")
        if gate_on_extract_fail and (not primary_extract_ok):
            gate_reasons.append("extract_fail")
        if gate_on_error and primary_has_error:
            gate_reasons.append("primary_error")
        critic_called = len(gate_reasons) > 0

        critic_prompt = ""
        critic_raw = ""
        verdict = "SKIP"
        critic_pred = ""
        critic_pred_norm = ""
        critic_corr = False
        critic = None
        if critic_called:
            n_critic_called += 1
            critic_prompt = _build_critic_prompt(sample, primary_raw, primary_pred_norm or primary_pred)
            critic = critic_adapter.generate(
                [{"role": "user", "content": critic_prompt}],
                critic_params,
                cache_dir=cache_dir,
                cache_key_extra=f"{sid}::critic::{critic_model}",
            )
            critic_raw = critic.text or ""
            verdict, critic_pred = _parse_critic(critic_raw)
            critic_pred_norm = normalize_model_answer(critic_pred, sample)
            critic_corr, _ = score(critic_pred_norm, gold, relaxed=False)

        conflict = critic_called and (verdict == "REJECT")
        use_critic = critic_called and conflict and bool(critic_pred_norm)

        if conflict:
            n_conflict += 1
        if use_critic:
            n_override += 1

        final_pred = critic_pred if use_critic else primary_pred
        final_pred_norm = normalize_model_answer(final_pred, sample)
        final_raw = critic_raw if use_critic else primary_raw
        final_corr, info = score(final_pred_norm, gold, relaxed=False)
        if use_critic and (not primary_corr) and final_corr:
            n_arb_success += 1

        n_total += 1
        if final_corr:
            n_correct += 1
        if FINAL_PAT.search(final_raw):
            n_contract += 1
        if _extract_rate_from_pred_norm(final_pred_norm):
            n_extract += 1
        if is_label_answer(final_pred):
            n_label += 1
        if is_numeric_answer(final_pred):
            n_numeric += 1
        if is_time_answer(final_pred_norm):
            n_time += 1
        if contains_time_fragment(final_raw):
            n_contains_time += 1
        if is_date_answer(final_pred_norm):
            n_date += 1
        if contains_date_fragment(final_raw):
            n_contains_date += 1

        row = {
            "id": sid,
            "model": "router_critic",
            "workflow_type": "router_critic",
            "router_type": "profile_category_accuracy",
            "routed_model": routed_model,
            "critic_model": critic_model,
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
            "primary_raw": primary_raw,
            "primary_pred": primary_pred,
            "primary_pred_norm": primary_pred_norm,
            "primary_correct": bool(primary_corr),
            "primary_contract_ok": primary_contract_ok,
            "primary_extract_ok": primary_extract_ok,
            "critic_called": bool(critic_called),
            "critic_gate_reasons": gate_reasons,
            "critic_prompt": critic_prompt,
            "critic_raw": critic_raw,
            "critic_verdict": verdict,
            "critic_pred": critic_pred,
            "critic_pred_norm": critic_pred_norm,
            "critic_correct": bool(critic_corr),
            "override_by_critic": bool(use_critic),
            "conflict": bool(conflict),
            "pred": final_pred,
            "pred_norm": final_pred_norm,
            "raw": final_raw,
            "gold": gold,
            "correct": bool(final_corr),
            "contract_ok": bool(FINAL_PAT.search(final_raw)),
            "extract_ok": _extract_rate_from_pred_norm(final_pred_norm),
            "match": info.get("match"),
            "primary_usage": primary.usage,
            "critic_usage": critic.usage if critic is not None else None,
            "calls_used": 1 + (1 if critic_called else 0),
            "usage": {
                "prompt_tokens": (primary.usage or {}).get("prompt_tokens", 0)
                + ((critic.usage or {}).get("prompt_tokens", 0) if critic is not None else 0),
                "completion_tokens": (primary.usage or {}).get("completion_tokens", 0)
                + ((critic.usage or {}).get("completion_tokens", 0) if critic is not None else 0),
                "total_tokens": (primary.usage or {}).get("total_tokens", 0)
                + ((critic.usage or {}).get("total_tokens", 0) if critic is not None else 0),
            },
            "latency": (primary.latency or 0.0) + ((critic.latency or 0.0) if critic is not None else 0.0),
            "error": "; ".join(
                [x for x in [primary.error, (critic.error if critic is not None else None)] if x]
            )
            or None,
        }
        append_jsonl(pred_path, row)

    elapsed = time.time() - start_time
    ensure_dir(run_dir)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
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
                "conflict_rate",
                "critic_override_rate",
                "arbitration_success_rate",
                "critic_call_rate",
                "prompt_version",
                "router_type",
                "workflow_type",
            ]
        )
        n = n_total
        acc = (n_correct / n) if n else 0.0
        contract_rate = (n_contract / n) if n else 0.0
        extract_rate = (n_extract / n) if n else 0.0
        parse_rate = extract_rate
        tr = (n_time / n) if n else 0.0
        ctr = (n_contains_time / n) if n else 0.0
        dr = (n_date / n) if n else 0.0
        cdr = (n_contains_date / n) if n else 0.0
        nr = (n_numeric / n) if n else 0.0
        lr = (n_label / n) if n else 0.0
        text_rate = 1.0 - max(tr, ctr, dr, cdr, nr, lr)
        conflict_rate = (n_conflict / n) if n else 0.0
        override_rate = (n_override / n) if n else 0.0
        arb_success_rate = (n_arb_success / n) if n else 0.0
        critic_call_rate = (n_critic_called / n) if n else 0.0
        cat_name = filter_category[0] if filter_category and len(filter_category) == 1 else "mixed"
        w.writerow(
            [
                "router_critic",
                cat_name,
                n,
                n_correct,
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
                f"{conflict_rate:.4f}",
                f"{override_rate:.4f}",
                f"{arb_success_rate:.4f}",
                f"{critic_call_rate:.4f}",
                prompt_version,
                "profile_category_accuracy",
                "router_critic",
            ]
        )

    with open(route_summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "routed_count", "routed_share", "avg_route_score", "category_source_count", "overall_source_count"])
        total = sum(route_counts.values())
        for model_name, count in sorted(route_counts.items()):
            share = (count / total) if total else 0.0
            avg_score = (route_score_sum[model_name] / count) if count else 0.0
            cat_source = route_source_counts[model_name].get("category_accuracy", 0)
            ov_source = route_source_counts[model_name].get("overall_accuracy", 0)
            w.writerow([model_name, count, f"{share:.4f}", f"{avg_score:.4f}", cat_source, ov_source])

    meta = {
        "run_id": Path(run_dir).name,
        "run_config_hash": run_config_hash,
        "router_type": "profile_category_accuracy",
        "workflow_type": "router_critic",
        "profile_path": str(Path(profile_path).resolve()),
        "models": usable_models,
        "critic_model": critic_model,
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
        "conflict_count": n_conflict,
        "override_count": n_override,
        "arbitration_success_count": n_arb_success,
        "critic_call_count": n_critic_called,
        "critic_gate": {
            "route_score_lt": gate_route_score_lt,
            "on_contract_fail": gate_on_contract_fail,
            "on_extract_fail": gate_on_extract_fail,
            "on_error": gate_on_error,
        },
        "elapsed_sec": elapsed,
    }
    (Path(run_dir) / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Predictions: {pred_path}")
    print(f"Summary: {summary_path}")
    print(f"Route summary: {route_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to router+critic config yaml")
    parser.add_argument("--models", required=False, default="configs/models.yaml", help="Path to models.yaml")
    parser.add_argument("--prompts", required=False, default="configs/prompts.yaml", help="Path to prompts.yaml")
    args = parser.parse_args()
    run(config_path=args.config, models_cfg_path=args.models, prompts_cfg_path=args.prompts)
