import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl


def _load_by_id(path: Path):
    out = {}
    for row in read_jsonl(path):
        sid = str(row.get("id", "")).strip()
        if sid:
            out[sid] = row
    return out


def _quantiles(xs):
    if not xs:
        return {"min": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    ys = sorted(xs)
    n = len(ys)

    def q(p):
        i = int(round((n - 1) * p))
        return ys[max(0, min(n - 1, i))]

    return {
        "min": ys[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": ys[-1],
        "mean": mean(ys),
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze TZ router+critic mechanism and route-score calibration")
    ap.add_argument("--router_pred", required=True)
    ap.add_argument("--critic_pred", required=True)
    ap.add_argument("--out_dir", default="outputs/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    r = _load_by_id(Path(args.router_pred).resolve())
    c = _load_by_id(Path(args.critic_pred).resolve())
    ids = sorted(set(r.keys()) & set(c.keys()))
    if not ids:
        raise RuntimeError("No common ids between router and critic predictions.")

    # Four-way breakdown requested by commander.
    breakdown = {
        "primary_correct_critic_to_wrong": 0,
        "primary_wrong_critic_to_correct": 0,
        "primary_wrong_critic_still_wrong": 0,
        "primary_correct_critic_still_correct": 0,
    }
    detailed_rows = []
    route_scores_all = []
    route_scores_primary_correct = []
    route_scores_primary_wrong = []

    for sid in ids:
        row = c[sid]
        primary_correct = bool(row.get("primary_correct"))
        final_correct = bool(row.get("correct"))
        critic_called = bool(row.get("critic_called"))
        override = bool(row.get("override_by_critic"))
        route_score = float(row.get("route_score", 0.0))
        route_scores_all.append(route_score)
        if primary_correct:
            route_scores_primary_correct.append(route_score)
        else:
            route_scores_primary_wrong.append(route_score)

        if primary_correct and (not final_correct):
            bucket = "primary_correct_critic_to_wrong"
        elif (not primary_correct) and final_correct:
            bucket = "primary_wrong_critic_to_correct"
        elif (not primary_correct) and (not final_correct):
            bucket = "primary_wrong_critic_still_wrong"
        else:
            bucket = "primary_correct_critic_still_correct"
        breakdown[bucket] += 1

        detailed_rows.append(
            {
                "id": sid,
                "routed_model": row.get("routed_model"),
                "route_score": f"{route_score:.4f}",
                "gold": row.get("gold"),
                "primary_pred_norm": row.get("primary_pred_norm"),
                "primary_correct": primary_correct,
                "critic_called": critic_called,
                "critic_verdict": row.get("critic_verdict"),
                "critic_pred_norm": row.get("critic_pred_norm"),
                "override_by_critic": override,
                "final_pred_norm": row.get("pred_norm"),
                "final_correct": final_correct,
                "bucket": bucket,
            }
        )

    with (out_dir / "tz_primary_critic_final_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "routed_model",
                "route_score",
                "gold",
                "primary_pred_norm",
                "primary_correct",
                "critic_called",
                "critic_verdict",
                "critic_pred_norm",
                "override_by_critic",
                "final_pred_norm",
                "final_correct",
                "bucket",
            ],
        )
        w.writeheader()
        for row in detailed_rows:
            w.writerow(row)

    with (out_dir / "tz_primary_critic_final_breakdown_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n": len(ids),
                "breakdown": breakdown,
                "critic_called_count": sum(1 for x in detailed_rows if x["critic_called"]),
                "override_count": sum(1 for x in detailed_rows if x["override_by_critic"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    route_summary = {
        "all": _quantiles(route_scores_all),
        "primary_correct": _quantiles(route_scores_primary_correct),
        "primary_wrong": _quantiles(route_scores_primary_wrong),
    }
    (out_dir / "tz_route_score_distribution.json").write_text(
        json.dumps(route_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[OK] wrote:", out_dir / "tz_primary_critic_final_breakdown.csv")
    print("[OK] wrote:", out_dir / "tz_primary_critic_final_breakdown_summary.json")
    print("[OK] wrote:", out_dir / "tz_route_score_distribution.json")
    print(json.dumps({"n": len(ids), "breakdown": breakdown}, ensure_ascii=False))


if __name__ == "__main__":
    main()
