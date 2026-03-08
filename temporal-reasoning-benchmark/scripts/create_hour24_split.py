import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_jsonl


def _load_ids_from_manifest(path: Path) -> List[str]:
    rows = read_jsonl(path)
    ids = []
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            ids.append(rid)
    return ids


def _load_dataset_map(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    out: Dict[str, dict] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="Create fixed profile/report split for Hour Adjustment (24h)")
    ap.add_argument("--dataset", required=True, help="Full dataset jsonl")
    ap.add_argument("--source_manifest", required=True, help="Manifest jsonl containing the fixed 100 ids")
    ap.add_argument("--out_dir", default="data/splits/hour24")
    ap.add_argument("--build_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_path = Path(args.dataset).resolve()
    manifest_path = Path(args.source_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_ids = _load_ids_from_manifest(manifest_path)
    source_ids = sorted(set(source_ids))
    if not source_ids:
        raise RuntimeError("source_manifest has no ids")

    ds = _load_dataset_map(dataset_path)
    missing = [x for x in source_ids if x not in ds]
    if missing:
        raise RuntimeError(f"{len(missing)} ids in source_manifest not found in dataset")

    rnd = random.Random(args.seed)
    shuffled = source_ids[:]
    rnd.shuffle(shuffled)

    build_n = int(round(len(shuffled) * args.build_ratio))
    build_n = max(1, min(len(shuffled) - 1, build_n))

    build_ids = sorted(shuffled[:build_n])
    report_ids = sorted(shuffled[build_n:])

    profile_rows = [ds[x] for x in build_ids]
    report_rows = [ds[x] for x in report_ids]

    (out_dir / "profile_build_ids.json").write_text(
        json.dumps({"ids": build_ids}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report_eval_ids.json").write_text(
        json.dumps({"ids": report_ids}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(out_dir / "profile_build.jsonl", profile_rows)
    write_jsonl(out_dir / "report_eval.jsonl", report_rows)

    summary = {
        "dataset": str(dataset_path),
        "source_manifest": str(manifest_path),
        "source_n": len(source_ids),
        "profile_build_n": len(build_ids),
        "report_eval_n": len(report_ids),
        "build_ratio": args.build_ratio,
        "seed": args.seed,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] split created: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
