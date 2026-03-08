import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _to_float(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _weighted_mean(pairs: List[Tuple[float, int]]) -> Optional[float]:
    valid = [(x, w) for x, w in pairs if x is not None and w is not None and w > 0]
    if not valid:
        return None
    return sum(x * w for x, w in valid) / sum(w for _, w in valid)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _scan_runs(runs_dir: Path, include_runs: set[str], min_n_per_row: int):
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if include_runs and run_id not in include_runs:
            continue
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            continue
        yield run_dir, summary_path, min_n_per_row


def build_profiles(project_root: Path, include_runs: List[str], min_n_per_row: int):
    runs_dir = project_root / "outputs" / "runs"
    include = set(include_runs)

    model_rows: Dict[str, List[dict]] = defaultdict(list)
    model_runs: Dict[str, set] = defaultdict(set)
    run_level_acc_parts: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
    used_runs: List[str] = []

    for run_dir, summary_path, min_n in _scan_runs(runs_dir, include, min_n_per_row):
        run_id = run_dir.name
        used_runs.append(run_id)
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = _to_int(row.get("n")) or 0
                if n < min_n:
                    continue
                model = (row.get("model") or "unknown").strip()
                cat = (row.get("category") or "unspecified").strip()
                acc = _to_float(row.get("accuracy"))
                pr = _to_float(row.get("parse_rate"))
                cr = _to_float(row.get("contract_rate"))
                model_rows[model].append(
                    {
                        "run_id": run_id,
                        "n": n,
                        "category": cat,
                        "accuracy": acc,
                        "parse_rate": pr,
                        "contract_rate": cr,
                    }
                )
                model_runs[model].add(run_id)
                if acc is not None:
                    old_c, old_n = run_level_acc_parts[model].get(run_id, (0.0, 0))
                    run_level_acc_parts[model][run_id] = (old_c + acc * n, old_n + n)

    profiles = []
    for model, rows in sorted(model_rows.items()):
        sample_count = sum(r["n"] for r in rows)
        run_count = len(model_runs[model])
        overall_acc = _weighted_mean([(r["accuracy"], r["n"]) for r in rows])
        overall_pr = _weighted_mean([(r["parse_rate"], r["n"]) for r in rows])
        overall_cr = _weighted_mean([(r["contract_rate"], r["n"]) for r in rows])
        cat_parts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        for r in rows:
            cat_parts[r["category"]].append((r["accuracy"], r["n"]))
        cat_acc = {k: _weighted_mean(v) for k, v in sorted(cat_parts.items())}
        run_acc_values = []
        for _, (corr_like, n) in sorted(run_level_acc_parts[model].items()):
            if n > 0:
                run_acc_values.append(corr_like / n)
        profiles.append(
            {
                "model": model,
                "sample_count": sample_count,
                "run_count": run_count,
                "overall_accuracy": overall_acc,
                "overall_parse_rate": overall_pr,
                "overall_format_compliance": overall_cr,
                "overall_latency_sec": None,
                "accuracy_std_across_runs": _std(run_acc_values),
                "category_accuracy": cat_acc,
                "source": {
                    "runs_dir": str(runs_dir),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "used_run_count": len(used_runs),
                    "used_runs": used_runs,
                    "min_n_per_row": min_n_per_row,
                },
            }
        )
    return profiles


def write_outputs(project_root: Path, profiles: List[dict], out_name: str):
    out_dir = project_root / "outputs" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{out_name}.json"
    csv_path = out_dir / f"{out_name}.csv"
    json_path.write_text(
        json.dumps({"version": "profile-v1-filtered", "profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "sample_count",
                "run_count",
                "overall_accuracy",
                "overall_parse_rate",
                "overall_format_compliance",
                "accuracy_std_across_runs",
            ],
        )
        w.writeheader()
        for p in profiles:
            w.writerow(
                {
                    "model": p["model"],
                    "sample_count": p["sample_count"],
                    "run_count": p["run_count"],
                    "overall_accuracy": p["overall_accuracy"],
                    "overall_parse_rate": p["overall_parse_rate"],
                    "overall_format_compliance": p["overall_format_compliance"],
                    "accuracy_std_across_runs": p["accuracy_std_across_runs"],
                }
            )
    return json_path, csv_path


def main():
    ap = argparse.ArgumentParser(description="Build filtered model profiles from selected runs")
    ap.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--include_runs", nargs="+", required=True)
    ap.add_argument("--out_name", default="profiles_v1_clean")
    ap.add_argument("--min_n_per_row", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    profiles = build_profiles(root, args.include_runs, args.min_n_per_row)
    j, c = write_outputs(root, profiles, args.out_name)
    print(f"[OK] profiles built: {len(profiles)} models")
    print(f"[OK] json: {j}")
    print(f"[OK] csv : {c}")


if __name__ == "__main__":
    main()
