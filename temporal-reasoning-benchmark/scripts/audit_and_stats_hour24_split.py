import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / den
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z2 / (4 * n * n))) / den
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def _load_by_id(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    out = {}
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid:
            out[sid] = r
    return out


def _bool(v) -> bool:
    return bool(v)


def main():
    ap = argparse.ArgumentParser(description="Audit router-only vs router+critic and compute split stats tables")
    ap.add_argument("--single_pred", required=True)
    ap.add_argument("--router_pred", required=True)
    ap.add_argument("--critic_pred", required=True)
    ap.add_argument("--out_dir", default="outputs/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    single = _load_by_id(Path(args.single_pred).resolve())
    router = _load_by_id(Path(args.router_pred).resolve())
    critic = _load_by_id(Path(args.critic_pred).resolve())

    common_ids = sorted(set(single.keys()) & set(router.keys()) & set(critic.keys()))
    if not common_ids:
        raise RuntimeError("No common sample ids across three runs.")

    # Audit Router-only vs Critic-v2
    audit_rows: List[dict] = []
    mismatch_count = 0
    same_route_count = 0
    critic_called_count = 0
    for sid in common_ids:
        r = router[sid]
        c = critic[sid]
        r_route = str(r.get("routed_model", ""))
        c_route = str(c.get("routed_model", ""))
        same_route = r_route == c_route
        if same_route:
            same_route_count += 1

        c_called = _bool(c.get("critic_called"))
        if c_called:
            critic_called_count += 1

        row = {
            "id": sid,
            "router_routed_model": r_route,
            "critic_routed_model": c_route,
            "same_routed_model": same_route,
            "router_pred_norm": str(r.get("pred_norm", "")),
            "critic_primary_pred_norm": str(c.get("primary_pred_norm", "")),
            "critic_final_pred_norm": str(c.get("pred_norm", "")),
            "router_correct": _bool(r.get("correct")),
            "critic_correct": _bool(c.get("correct")),
            "same_correct": _bool(r.get("correct")) == _bool(c.get("correct")),
            "critic_called": c_called,
            "critic_gate_reasons": json.dumps(c.get("critic_gate_reasons", []), ensure_ascii=False),
            "router_latency": r.get("latency", 0.0),
            "critic_latency": c.get("latency", 0.0),
            "router_error": r.get("error"),
            "critic_error": c.get("error"),
        }
        if (not row["same_routed_model"]) or (not row["same_correct"]) or (row["router_pred_norm"] != row["critic_final_pred_norm"]):
            mismatch_count += 1
            audit_rows.append(row)

    audit_csv = out_dir / "audit_router_vs_critic_v2_diffs.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "router_routed_model",
                "critic_routed_model",
                "same_routed_model",
                "router_pred_norm",
                "critic_primary_pred_norm",
                "critic_final_pred_norm",
                "router_correct",
                "critic_correct",
                "same_correct",
                "critic_called",
                "critic_gate_reasons",
                "router_latency",
                "critic_latency",
                "router_error",
                "critic_error",
            ],
        )
        w.writeheader()
        for row in audit_rows:
            w.writerow(row)

    audit_summary = {
        "n_common": len(common_ids),
        "mismatch_count": mismatch_count,
        "same_routed_model_count": same_route_count,
        "critic_called_count": critic_called_count,
        "note": "mismatch means route mismatch OR final prediction/correctness mismatch",
    }
    (out_dir / "audit_router_vs_critic_v2_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stats tables
    def _acc(rows: Dict[str, dict]) -> Tuple[int, int]:
        n = 0
        k = 0
        for sid in common_ids:
            n += 1
            if _bool(rows[sid].get("correct")):
                k += 1
        return k, n

    k_s, n = _acc(single)
    k_r, _ = _acc(router)
    k_c, _ = _acc(critic)

    stats_rows = []
    for name, k in [("Single-Best", k_s), ("Profile-Router", k_r), ("Profile-Router+GatedCriticV2", k_c)]:
        lo, hi = _wilson_ci(k, n)
        stats_rows.append(
            {
                "workflow": name,
                "correct": k,
                "n": n,
                "accuracy": f"{k/n:.4f}",
                "wilson_ci_low": f"{lo:.4f}",
                "wilson_ci_high": f"{hi:.4f}",
            }
        )

    stats_csv = out_dir / "hour24_split_wilson_ci.csv"
    with stats_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["workflow", "correct", "n", "accuracy", "wilson_ci_low", "wilson_ci_high"])
        w.writeheader()
        for row in stats_rows:
            w.writerow(row)

    # paired contingency + McNemar
    def _pair_counts(a: Dict[str, dict], b: Dict[str, dict]) -> Dict[str, int]:
        both_correct = 0
        a_only = 0
        b_only = 0
        both_wrong = 0
        for sid in common_ids:
            ac = _bool(a[sid].get("correct"))
            bc = _bool(b[sid].get("correct"))
            if ac and bc:
                both_correct += 1
            elif ac and (not bc):
                a_only += 1
            elif (not ac) and bc:
                b_only += 1
            else:
                both_wrong += 1
        return {
            "both_correct": both_correct,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "both_wrong": both_wrong,
        }

    pairs = [
        ("Single-Best", "Profile-Router", single, router),
        ("Profile-Router", "Profile-Router+GatedCriticV2", router, critic),
    ]

    pair_csv = out_dir / "hour24_split_paired_contingency.csv"
    mcnemar_csv = out_dir / "hour24_split_mcnemar.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as f1, mcnemar_csv.open("w", encoding="utf-8", newline="") as f2:
        w1 = csv.DictWriter(
            f1,
            fieldnames=["system_a", "system_b", "both_correct", "a_only_correct", "b_only_correct", "both_wrong"],
        )
        w2 = csv.DictWriter(
            f2,
            fieldnames=["system_a", "system_b", "b", "c", "mcnemar_exact_pvalue"],
        )
        w1.writeheader()
        w2.writeheader()
        for a_name, b_name, a_rows, b_rows in pairs:
            ct = _pair_counts(a_rows, b_rows)
            w1.writerow({"system_a": a_name, "system_b": b_name, **ct})
            # McNemar uses discordant counts b/c
            # b: a correct, b wrong ; c: a wrong, b correct
            b = ct["a_only_correct"]
            c = ct["b_only_correct"]
            p = _mcnemar_exact_p(b, c)
            w2.writerow(
                {
                    "system_a": a_name,
                    "system_b": b_name,
                    "b": b,
                    "c": c,
                    "mcnemar_exact_pvalue": f"{p:.6f}",
                }
            )

    print(f"[OK] audit summary: {out_dir / 'audit_router_vs_critic_v2_summary.json'}")
    print(f"[OK] audit diffs  : {audit_csv}")
    print(f"[OK] wilson table : {stats_csv}")
    print(f"[OK] paired table : {pair_csv}")
    print(f"[OK] mcnemar table: {mcnemar_csv}")


if __name__ == "__main__":
    main()
