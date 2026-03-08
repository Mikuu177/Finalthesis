import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_summary(summary_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _weighted_metric(rows: List[Dict[str, str]], key: str) -> Tuple[Optional[float], int]:
    num = 0.0
    den = 0
    for row in rows:
        n = _safe_int(row.get("n")) or 0
        x = _safe_float(row.get(key))
        if n <= 0 or x is None:
            continue
        num += x * n
        den += n
    if den == 0:
        return None, 0
    return num / den, den


def _collect_prediction_metrics(pred_path: Path) -> Dict[str, Optional[float]]:
    if not pred_path.exists():
        return {
            "sample_count": 0,
            "total_calls": 0,
            "calls_per_query": None,
            "avg_latency_sec_per_call": None,
            "avg_latency_sec_per_query": None,
            "total_prompt_tokens": None,
            "total_completion_tokens": None,
            "total_tokens": None,
            "cost_calls_per_correct": None,
            "cost_latency_sec_per_correct": None,
        }

    total_calls = 0
    sample_ids = set()
    total_latency = 0.0
    latency_n = 0

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    has_usage = False

    correct_n = 0

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            calls_used = _safe_int(row.get("calls_used"))
            total_calls += calls_used if calls_used is not None and calls_used > 0 else 1
            sid = str(row.get("id", ""))
            if sid:
                sample_ids.add(sid)

            if row.get("correct"):
                correct_n += 1

            lat = _safe_float(row.get("latency"))
            if lat is not None:
                total_latency += lat
                latency_n += 1

            usage = row.get("usage") or {}
            pt = _safe_int(usage.get("prompt_tokens"))
            ct = _safe_int(usage.get("completion_tokens"))
            tt = _safe_int(usage.get("total_tokens"))
            if pt is not None:
                prompt_tokens += pt
                has_usage = True
            if ct is not None:
                completion_tokens += ct
                has_usage = True
            if tt is not None:
                total_tokens += tt
                has_usage = True

    sample_count = len(sample_ids)
    calls_per_query = (total_calls / sample_count) if sample_count > 0 else None
    avg_latency_call = (total_latency / latency_n) if latency_n > 0 else None
    avg_latency_query = (total_latency / sample_count) if sample_count > 0 else None
    calls_per_correct = (total_calls / correct_n) if correct_n > 0 else None
    latency_per_correct = (total_latency / correct_n) if correct_n > 0 else None

    return {
        "sample_count": sample_count,
        "total_calls": total_calls,
        "calls_per_query": calls_per_query,
        "avg_latency_sec_per_call": avg_latency_call,
        "avg_latency_sec_per_query": avg_latency_query,
        "total_prompt_tokens": prompt_tokens if has_usage else None,
        "total_completion_tokens": completion_tokens if has_usage else None,
        "total_tokens": total_tokens if has_usage else None,
        "cost_calls_per_correct": calls_per_correct,
        "cost_latency_sec_per_correct": latency_per_correct,
    }


def _fmt(x: Optional[float], d: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{d}f}"


def _infer_workflow_type(meta: Dict[str, object]) -> str:
    wt = str(meta.get("workflow_type", "") or "")
    if wt:
        return wt
    if meta.get("router_type"):
        return "profile_router_only"
    return "strong_single_or_fixed"


def summarize_run(run_dir: Path) -> Dict[str, str]:
    summary_path = run_dir / "summary.csv"
    pred_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "run_metadata.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    summary_rows = _read_summary(summary_path)
    acc, n_from_summary = _weighted_metric(summary_rows, "accuracy")
    parse_rate, _ = _weighted_metric(summary_rows, "parse_rate")
    contract_rate, _ = _weighted_metric(summary_rows, "contract_rate")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pred_metrics = _collect_prediction_metrics(pred_path)
    elapsed = _safe_float(meta.get("elapsed_sec"))
    workflow_type = _infer_workflow_type(meta)

    sample_count = int(pred_metrics["sample_count"] or n_from_summary or 0)
    total_calls = int(pred_metrics["total_calls"] or 0)
    calls_per_query = pred_metrics["calls_per_query"]
    latency_per_query = pred_metrics["avg_latency_sec_per_query"]
    calls_per_correct = pred_metrics["cost_calls_per_correct"]
    latency_per_correct = pred_metrics["cost_latency_sec_per_correct"]

    return {
        "run_id": run_dir.name,
        "workflow_type": workflow_type,
        "router_type": str(meta.get("router_type", "")),
        "prompt_mode": str(meta.get("prompt_mode", "")),
        "n_shots": str(meta.get("n_shots", "")),
        "sample_count": str(sample_count),
        "accuracy": _fmt(acc),
        "parse_rate": _fmt(parse_rate),
        "format_compliance": _fmt(contract_rate),
        "latency_sec_total": _fmt(elapsed),
        "latency_sec_per_query": _fmt(latency_per_query),
        "calls_total": str(total_calls),
        "calls_per_query": _fmt(calls_per_query),
        "calls_per_correct": _fmt(calls_per_correct),
        "latency_sec_per_correct": _fmt(latency_per_correct),
        "total_tokens": str(pred_metrics["total_tokens"] or ""),
        "prompt_tokens": str(pred_metrics["total_prompt_tokens"] or ""),
        "completion_tokens": str(pred_metrics["total_completion_tokens"] or ""),
        "notes": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready comparison table for workflows")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories (each must include summary.csv)",
    )
    parser.add_argument(
        "--out",
        default="outputs/tables/workflow_comparison.csv",
        help="Output csv path",
    )
    args = parser.parse_args()

    run_dirs = [Path(x).resolve() for x in args.runs]
    rows = [summarize_run(p) for p in run_dirs]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "workflow_type",
        "router_type",
        "prompt_mode",
        "n_shots",
        "sample_count",
        "accuracy",
        "parse_rate",
        "format_compliance",
        "latency_sec_total",
        "latency_sec_per_query",
        "calls_total",
        "calls_per_query",
        "calls_per_correct",
        "latency_sec_per_correct",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[OK] wrote comparison table: {out_path}")
    for row in rows:
        print(
            f"- {row['run_id']} | workflow={row['workflow_type']} | acc={row['accuracy']} "
            f"| calls/query={row['calls_per_query']} | latency/query={row['latency_sec_per_query']}"
        )


if __name__ == "__main__":
    main()
