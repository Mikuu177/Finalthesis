import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: dict):
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=True), encoding="utf-8")


def _run_cmd(cmd: List[str], cwd: Path):
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _find_run_dir(outputs_runs: Path, alias_name: str, prompt_version: str) -> Path:
    suffix = f"_{alias_name}_pv{prompt_version}"
    cands = [p for p in outputs_runs.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not cands:
        raise RuntimeError(f"cannot find run dir with suffix: {suffix}")
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]


def _read_summary_accuracy(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        row = next(r)
        return float(row["accuracy"])


def _read_preds(path: Path) -> Dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("id", "")).strip()
        if sid:
            out[sid] = row
    return out


def _prepare_cfg(base_cfg: Path, repeat: int, cache_tag: str) -> Path:
    cfg = _read_yaml(base_cfg)
    base_name = str(cfg.get("run_name", "run"))
    cfg["run_name"] = f"{base_name}_temp0_r{repeat}"
    cfg["cache_dir"] = f".cache_repro_temp0/{cache_tag}_r{repeat}"
    params = dict(cfg.get("params", {}))
    params["temperature"] = 0.0
    params.pop("top_p", None)
    cfg["params"] = params
    if "critic_params" in cfg:
        cparams = dict(cfg.get("critic_params", {}))
        cparams["temperature"] = 0.0
        cparams.pop("top_p", None)
        cfg["critic_params"] = cparams
    tmp_dir = base_cfg.resolve().parents[1] / "outputs" / "tmp_cfg"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{base_cfg.stem}_temp0_r{repeat}.yaml"
    _write_yaml(out, cfg)
    return out


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0}
    return {"mean": statistics.mean(xs), "std": statistics.stdev(xs)}


def main():
    ap = argparse.ArgumentParser(description="Reproducibility check on Hour24 split with temperature=0.0")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outputs_runs = root / "outputs" / "runs"
    outputs_tables = root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)

    base_single = root / "configs" / "strong_single_deepseek_report_eval.yaml"
    base_router = root / "configs" / "router_only_hour24_report_eval.yaml"
    base_critic = root / "configs" / "router_critic_hour24_report_eval_v2.yaml"

    accs = {"single": [], "router": [], "critic_v2": []}
    consistency_rows = []

    for i in range(1, args.repeats + 1):
        cfg_single = _prepare_cfg(base_single, i, "single")
        cfg_router = _prepare_cfg(base_router, i, "router")
        cfg_critic = _prepare_cfg(base_critic, i, "critic_v2")

        _run_cmd(["python", "-m", "src.runner", "--config", str(cfg_single)], root)
        _run_cmd(["python", "scripts/run_router_only.py", "--config", str(cfg_router)], root)
        _run_cmd(["python", "scripts/run_router_critic.py", "--config", str(cfg_critic)], root)

        s_alias = _read_yaml(cfg_single)["run_name"]
        r_alias = _read_yaml(cfg_router)["run_name"]
        c_alias = _read_yaml(cfg_critic)["run_name"]
        pv = str(_read_yaml(cfg_single).get("prompt_version", "v1"))

        s_dir = _find_run_dir(outputs_runs, s_alias, pv)
        r_dir = _find_run_dir(outputs_runs, r_alias, pv)
        c_dir = _find_run_dir(outputs_runs, c_alias, pv)

        accs["single"].append(_read_summary_accuracy(s_dir / "summary.csv"))
        accs["router"].append(_read_summary_accuracy(r_dir / "summary.csv"))
        accs["critic_v2"].append(_read_summary_accuracy(c_dir / "summary.csv"))

        r_preds = _read_preds(r_dir / "predictions.jsonl")
        c_preds = _read_preds(c_dir / "predictions.jsonl")
        ids = sorted(set(r_preds.keys()) & set(c_preds.keys()))
        if not ids:
            raise RuntimeError("empty id overlap for router/critic")

        same_correct = 0
        same_pred = 0
        critic_zero = 0
        for sid in ids:
            rr = r_preds[sid]
            cc = c_preds[sid]
            if not cc.get("critic_called"):
                critic_zero += 1
            if bool(rr.get("correct")) == bool(cc.get("correct")):
                same_correct += 1
            if str(rr.get("pred_norm", "")) == str(cc.get("pred_norm", "")):
                same_pred += 1

        consistency_rows.append(
            {
                "repeat": i,
                "n_common": len(ids),
                "critic_called_count": len(ids) - critic_zero,
                "same_correct_count": same_correct,
                "same_correct_rate": same_correct / len(ids),
                "same_pred_norm_count": same_pred,
                "same_pred_norm_rate": same_pred / len(ids),
            }
        )

    summary = {
        "single_best": _mean_std(accs["single"]),
        "profile_router": _mean_std(accs["router"]),
        "profile_router_gatedcritic_v2": _mean_std(accs["critic_v2"]),
        "router_vs_critic_consistency_when_critic_zero": {
            "mean_same_correct_rate": statistics.mean([x["same_correct_rate"] for x in consistency_rows]),
            "mean_same_pred_norm_rate": statistics.mean([x["same_pred_norm_rate"] for x in consistency_rows]),
            "all_critic_called_counts": [x["critic_called_count"] for x in consistency_rows],
        },
        "raw_accuracies": accs,
    }

    (outputs_tables / "hour24_temp0_repeats_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (outputs_tables / "hour24_temp0_router_vs_critic_consistency.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "repeat",
                "n_common",
                "critic_called_count",
                "same_correct_count",
                "same_correct_rate",
                "same_pred_norm_count",
                "same_pred_norm_rate",
            ],
        )
        w.writeheader()
        for row in consistency_rows:
            w.writerow(row)

    print("[OK] wrote:", outputs_tables / "hour24_temp0_repeats_summary.json")
    print("[OK] wrote:", outputs_tables / "hour24_temp0_router_vs_critic_consistency.csv")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
