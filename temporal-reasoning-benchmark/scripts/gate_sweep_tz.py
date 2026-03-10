import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import List

import yaml


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: dict):
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=True), encoding="utf-8")


def _run(cmd: List[str], cwd: Path):
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_run(outputs_runs: Path, alias: str, pv: str) -> Path:
    suffix = f"_{alias}_pv{pv}"
    cands = [p for p in outputs_runs.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not cands:
        raise RuntimeError(f"run not found for alias={alias}")
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]


def _read_summary(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return next(r)


def main():
    ap = argparse.ArgumentParser(description="Gate sweep for TZ router+critic")
    ap.add_argument("--base_config", default="configs/router_critic_tz_report_eval_v2.yaml")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.30, 0.35, 0.36, 0.37, 0.50, 0.98])
    ap.add_argument("--out_csv", default="outputs/tables/tz_gate_sweep.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_path = (root / args.base_config).resolve()
    base_cfg = _read_yaml(base_path)
    pv = str(base_cfg.get("prompt_version", "v1"))
    outputs_runs = root / "outputs" / "runs"

    tmp_dir = root / "outputs" / "tmp_cfg"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for th in args.thresholds:
        cfg = dict(base_cfg)
        gate = dict(cfg.get("critic_gate", {}))
        gate["route_score_lt"] = float(th)
        cfg["critic_gate"] = gate
        cfg["run_name"] = f"{base_cfg.get('run_name','router_critic_tz')}_sweep_{str(th).replace('.', 'p')}"
        cfg["cache_dir"] = f".cache_tz_gate_sweep/{str(th).replace('.', 'p')}"
        tmp_cfg = tmp_dir / f"{base_path.stem}_sweep_{str(th).replace('.', 'p')}.yaml"
        _write_yaml(tmp_cfg, cfg)

        _run(["python", "scripts/run_router_critic.py", "--config", str(tmp_cfg)], root)
        run_dir = _latest_run(outputs_runs, cfg["run_name"], pv)
        summary = _read_summary(run_dir / "summary.csv")
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

        n = int(summary["n"])
        rows.append(
            {
                "threshold": th,
                "run_id": run_dir.name,
                "n": n,
                "accuracy": float(summary["accuracy"]),
                "critic_call_rate": float(summary.get("critic_call_rate", 0.0)),
                "conflict_rate": float(summary.get("conflict_rate", 0.0)),
                "override_rate": float(summary.get("critic_override_rate", 0.0)),
                "arbitration_success_rate": float(summary.get("arbitration_success_rate", 0.0)),
                "critic_call_count": int(meta.get("critic_call_count", 0)),
                "calls_per_query_est": 1.0 + (int(meta.get("critic_call_count", 0)) / n if n else 0.0),
                "elapsed_sec": float(meta.get("elapsed_sec", 0.0)),
            }
        )

    out_csv = (root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "run_id",
                "n",
                "accuracy",
                "critic_call_rate",
                "conflict_rate",
                "override_rate",
                "arbitration_success_rate",
                "critic_call_count",
                "calls_per_query_est",
                "elapsed_sec",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[OK] wrote:", out_csv)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
