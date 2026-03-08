import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v) -> Optional[int]:
    if v is None or v == "":
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
    num = sum(x * w for x, w in valid)
    den = sum(w for _, w in valid)
    if den == 0:
        return None
    return num / den


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _scan_runs(runs_dir: Path):
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            continue
        metadata_path = run_dir / "run_metadata.json"
        yield run_dir, summary_path, metadata_path


def build_profiles(project_root: Path):
    runs_dir = project_root / "outputs" / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")

    model_rows: Dict[str, List[dict]] = defaultdict(list)
    model_runs: Dict[str, set] = defaultdict(set)
    model_latency_parts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    # run-level accuracy（用于稳定性）: model -> run_id -> (correct_sum, n_sum)
    run_level_acc_parts: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)

    used_runs: List[str] = []

    for run_dir, summary_path, metadata_path in _scan_runs(runs_dir):
        run_id = run_dir.name
        used_runs.append(run_id)

        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        elapsed_sec = _to_float(metadata.get("elapsed_sec"))
        sample_size = _to_int(metadata.get("sample_size"))

        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = (row.get("model") or "unknown").strip()
                n = _to_int(row.get("n")) or 0
                if n <= 0:
                    continue

                row_payload = {
                    "run_id": run_id,
                    "model": model,
                    "category": (row.get("category") or "unspecified").strip(),
                    "n": n,
                    "accuracy": _to_float(row.get("accuracy")),
                    "parse_rate": _to_float(row.get("parse_rate")),
                    "contract_rate": _to_float(row.get("contract_rate")),
                    "prompt_mode": metadata.get("prompt_mode"),
                }
                model_rows[model].append(row_payload)
                model_runs[model].add(run_id)

                # run-level accuracy parts
                correct = _to_int(row.get("correct"))
                if correct is None:
                    acc = _to_float(row.get("accuracy"))
                    if acc is not None:
                        correct = int(round(acc * n))
                if correct is not None:
                    old = run_level_acc_parts[model].get(run_id, (0.0, 0))
                    run_level_acc_parts[model][run_id] = (old[0] + float(correct), old[1] + n)

                # latency uses run metadata once per model per run
                if elapsed_sec is not None and sample_size is not None and sample_size > 0:
                    # 避免同 run 多 category 重复叠加：只在模型首次出现时添加
                    if run_id not in {rid for rid, _ in model_latency_parts[model]}:
                        model_latency_parts[model].append((run_id, (elapsed_sec, sample_size)))

    profiles = []

    for model, rows in sorted(model_rows.items()):
        sample_count = sum(r["n"] for r in rows)
        run_count = len(model_runs[model])

        overall_accuracy = _weighted_mean([(r["accuracy"], r["n"]) for r in rows])
        overall_parse_rate = _weighted_mean([(r["parse_rate"], r["n"]) for r in rows])
        overall_format = _weighted_mean([(r["contract_rate"], r["n"]) for r in rows])

        # category metrics
        cat_acc_parts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        cat_parse_parts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        prompt_acc_parts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        for r in rows:
            cat_acc_parts[r["category"]].append((r["accuracy"], r["n"]))
            cat_parse_parts[r["category"]].append((r["parse_rate"], r["n"]))
            pm = r.get("prompt_mode")
            if pm:
                prompt_acc_parts[str(pm)].append((r["accuracy"], r["n"]))

        category_accuracy = {
            k: _weighted_mean(v) for k, v in sorted(cat_acc_parts.items())
        }
        category_parse_rate = {
            k: _weighted_mean(v) for k, v in sorted(cat_parse_parts.items())
        }
        prompt_mode_accuracy = {
            k: _weighted_mean(v) for k, v in sorted(prompt_acc_parts.items())
        }

        # run-level std
        run_acc_values = []
        for _, (corr_sum, n_sum) in sorted(run_level_acc_parts[model].items()):
            if n_sum > 0:
                run_acc_values.append(corr_sum / n_sum)
        accuracy_std = _std(run_acc_values)

        # latency per sample
        latency_pairs = [pair for _, pair in model_latency_parts[model]]
        overall_latency_sec = None
        if latency_pairs:
            total_elapsed = sum(x for x, _ in latency_pairs)
            total_samples = sum(n for _, n in latency_pairs)
            if total_samples > 0:
                overall_latency_sec = total_elapsed / total_samples

        profile = {
            "model": model,
            "sample_count": sample_count,
            "run_count": run_count,
            "overall_accuracy": overall_accuracy,
            "overall_parse_rate": overall_parse_rate,
            "overall_format_compliance": overall_format,
            "overall_latency_sec": overall_latency_sec,
            "accuracy_std_across_runs": accuracy_std,
            "category_accuracy": category_accuracy,
            "category_parse_rate": category_parse_rate,
            "prompt_mode_accuracy": prompt_mode_accuracy,
            "source": {
                "runs_dir": str(runs_dir),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "used_run_count": len(used_runs),
                "used_runs": used_runs,
            },
        }
        profiles.append(profile)

    return profiles


def write_outputs(project_root: Path, profiles: List[dict]):
    out_dir = project_root / "outputs" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "profiles_v1.json"
    json_path.write_text(
        json.dumps({"version": "profile-v1", "profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "profiles_v1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "sample_count",
                "run_count",
                "overall_accuracy",
                "overall_parse_rate",
                "overall_format_compliance",
                "overall_latency_sec",
                "accuracy_std_across_runs",
            ],
        )
        writer.writeheader()
        for p in profiles:
            writer.writerow(
                {
                    "model": p["model"],
                    "sample_count": p["sample_count"],
                    "run_count": p["run_count"],
                    "overall_accuracy": p["overall_accuracy"],
                    "overall_parse_rate": p["overall_parse_rate"],
                    "overall_format_compliance": p["overall_format_compliance"],
                    "overall_latency_sec": p["overall_latency_sec"],
                    "accuracy_std_across_runs": p["accuracy_std_across_runs"],
                }
            )

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Build model profiles from benchmark run outputs")
    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="temporal-reasoning-benchmark project root",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    profiles = build_profiles(project_root)
    json_path, csv_path = write_outputs(project_root, profiles)

    print(f"[OK] profiles built: {len(profiles)} models")
    print(f"[OK] json: {json_path}")
    print(f"[OK] csv : {csv_path}")


if __name__ == "__main__":
    main()
