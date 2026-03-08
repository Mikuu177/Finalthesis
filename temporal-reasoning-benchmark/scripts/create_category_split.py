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


def _load_dataset(path: Path, category: str) -> List[dict]:
    rows = read_jsonl(path)
    return [r for r in rows if str(r.get("category", "")) == category]


def _row_map(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid:
            out[sid] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="Create fixed split assets for a category")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--sample_n", type=int, default=100)
    ap.add_argument("--build_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ds_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_rows = _load_dataset(ds_path, args.category)
    if len(cat_rows) < args.sample_n:
        raise RuntimeError(f"category '{args.category}' has only {len(cat_rows)} rows, < sample_n={args.sample_n}")

    row_by_id = _row_map(cat_rows)
    ids = sorted(row_by_id.keys())
    rnd = random.Random(args.seed)
    rnd.shuffle(ids)
    picked = sorted(ids[: args.sample_n])

    build_n = int(round(args.sample_n * args.build_ratio))
    build_n = max(1, min(args.sample_n - 1, build_n))
    rnd2 = random.Random(args.seed + 1)
    work = picked[:]
    rnd2.shuffle(work)
    build_ids = sorted(work[:build_n])
    report_ids = sorted(work[build_n:])

    build_rows = [row_by_id[x] for x in build_ids]
    report_rows = [row_by_id[x] for x in report_ids]

    (out_dir / "profile_build_ids.json").write_text(json.dumps({"ids": build_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report_eval_ids.json").write_text(json.dumps({"ids": report_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "profile_build.jsonl", build_rows)
    write_jsonl(out_dir / "report_eval.jsonl", report_rows)
    summary = {
        "dataset": str(ds_path),
        "category": args.category,
        "sample_n": args.sample_n,
        "profile_build_n": len(build_rows),
        "report_eval_n": len(report_rows),
        "build_ratio": args.build_ratio,
        "seed": args.seed,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] split created: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
