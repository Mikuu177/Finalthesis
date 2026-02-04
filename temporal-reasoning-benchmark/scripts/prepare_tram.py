import argparse
import csv
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# -------------------------
# Repo root discovery
# -------------------------

def find_repo_root(start: Path) -> Path:
    """Heuristic: walk up to find repo root (expects requirements.txt/configs/src)."""
    markers = ['requirements.txt', 'configs', 'src']
    cur = start.resolve()
    for _ in range(10):
        if all((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


# -------------------------
# Path resolution + auto locate
# -------------------------

def _try_resolve_input(input_str: Optional[str], repo_root: Path, patterns: List[str]) -> Tuple[Optional[Path], Dict[str, Any]]:
    attempts: List[str] = []

    # 1) user provided
    if input_str:
        p = Path(input_str).expanduser()
        attempts.append(f"raw_input={p}")
        if not p.is_absolute():
            p = (repo_root / p)
            attempts.append(f"repo_root_join={p}")
        try:
            p = p.resolve()
        except Exception:
            p = p.absolute()
        attempts.append(f"resolved={p}")
        if p.exists():
            return p, {'mode': 'user_input', 'attempts': attempts}

    # 2) auto locate via glob
    candidates: List[Path] = []
    for pat in patterns:
        found = list(repo_root.glob(pat))
        attempts.append(f"glob({pat}) -> {len(found)}")
        candidates.extend(found)

    # uniq
    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            rc = c.absolute()
        if str(rc) in seen:
            continue
        seen.add(str(rc))
        uniq.append(rc)

    if len(uniq) == 1:
        return uniq[0], {'mode': 'auto_locate', 'attempts': attempts, 'candidates': [str(uniq[0])]}
    if len(uniq) == 0:
        return None, {'mode': 'auto_locate_none', 'attempts': attempts, 'candidates': []}
    return None, {'mode': 'auto_locate_multi', 'attempts': attempts, 'candidates': [str(x) for x in uniq]}


def actionable_error(header: str, repo_root: Path, cwd: Path, details: Dict[str, Any], next_cmd: str) -> str:
    parts = [header]
    parts.append("\n[Context]")
    parts.append(f"  repo_root = {repo_root}")
    parts.append(f"  cwd      = {cwd}")
    parts.append("\n[Attempts]")
    for a in details.get('attempts', []):
        parts.append(f"  - {a}")
    if 'candidates' in details:
        parts.append("\n[Candidates]")
        for c in details['candidates']:
            parts.append(f"  - {c}")
    parts.append("\n[Next step]")
    parts.append(f"  {next_cmd}")
    return "\n".join(parts)


# -------------------------
# CSV reading with encoding fallback
# -------------------------

def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]], str]:
    encodings = ['utf-8-sig', 'utf-8', 'latin-1']
    last_err = None
    for enc in encodings:
        try:
            with path.open('r', encoding=enc, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                cols = reader.fieldnames or []
                return cols, rows, enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV with encodings {encodings}: {last_err}")


# -------------------------
# Column inference
# -------------------------

def infer_qa_columns(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    lower = {c.lower(): c for c in cols}
    q_candidates = ['question', 'query', 'prompt', 'input', 'problem']
    a_candidates = ['answer', 'gold', 'label', 'target', 'output']

    q_col = None
    a_col = None

    for k in q_candidates:
        if k in lower:
            q_col = lower[k]
            break

    for k in a_candidates:
        if k in lower:
            a_col = lower[k]
            break

    return q_col, a_col


def stable_row_id(row: Dict[str, Any], idx: int, id_col: Optional[str], question_fallback: str) -> str:
    if id_col and row.get(id_col) not in (None, ''):
        return str(row[id_col])
    h = hashlib.sha256((str(idx) + '|' + question_fallback).encode('utf-8')).hexdigest()[:12]
    return f"row_{idx}_{h}"


_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")


def normalize_arithmetic_answer(ans: Any) -> str:
    if ans is None:
        return ''
    s = str(ans).strip()
    s = ''.join(s.split())
    try:
        s2 = s.replace(',', '')
        if _NUM_RE.match(s2):
            v = float(s2)
            if abs(v - int(v)) < 1e-9:
                return str(int(v))
            out = (f"{v:.12f}").rstrip('0').rstrip('.')
            return out
    except Exception:
        pass
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_audit(audit_path: Path, audit: Dict[str, Any]):
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_jsonl(out_path: Path) -> Tuple[bool, List[str]]:
    required = ['id', 'task', 'question', 'gold', 'metadata', 'category']
    errors = []
    if not out_path.exists():
        return False, [f"Output not found: {out_path}"]
    with out_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"Line {i}: invalid json: {e}")
                continue
            for k in required:
                if k not in obj:
                    errors.append(f"Line {i}: missing field '{k}'")
            for k in ['id', 'task', 'question', 'gold', 'category']:
                if obj.get(k) in (None, ''):
                    errors.append(f"Line {i}: empty {k}")
            if not isinstance(obj.get('metadata'), dict):
                errors.append(f"Line {i}: metadata not a dict")
    return len(errors) == 0, errors


def is_mcq_label(x: str) -> bool:
    return (x or '').strip().upper() in ('A', 'B', 'C', 'D')


def mcq_options_from_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    mapping = {}
    candidates = {
        'A': ['Option A', 'option a', 'option_a', 'A'],
        'B': ['Option B', 'option b', 'option_b', 'B'],
        'C': ['Option C', 'option c', 'option_c', 'C'],
        'D': ['Option D', 'option d', 'option_d', 'D'],
    }
    for lab, keys in candidates.items():
        val = None
        for k in keys:
            if k in row and row.get(k) not in (None, ''):
                val = str(row.get(k))
                break
        if val is None:
            return None
        mapping[lab] = val
    return mapping


def main():
    ap = argparse.ArgumentParser(description='Prepare TRAM datasets into unified JSONL schema for temporal-reasoning-benchmark.')
    ap.add_argument('--input', default=None, help='Input path (relative/absolute). If missing/wrong, auto-locate under repo_root.')
    ap.add_argument('--repo_root', default=None, help='Optional repo root. Defaults to auto-discovery from script location.')
    ap.add_argument('--output', default='data/raw/tram_arithmetic_mcq.jsonl', help='Output JSONL path (relative to repo_root unless absolute).')
    ap.add_argument('--task', default='arithmetic', help='Task name to stamp into output (default: arithmetic).')
    ap.add_argument('--question_col', default=None, help='Override question column name.')
    ap.add_argument('--answer_col', default=None, help='Override answer/gold column name.')
    ap.add_argument('--id_col', default=None, help='Override id column name (if present).')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of rows (0 means all).')
    ap.add_argument('--dry_run', action='store_true', help='Probe mode: print inferred columns and first 3 samples, do not write output.')
    ap.add_argument('--validate_out', action='store_true', help='Validate output JSONL after writing.')
    ap.add_argument('--add_meta_alias', action='store_true', help='Also write a meta field equal to metadata (compat alias).')

    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else find_repo_root(script_dir)

    patterns = [
        '**/TRAM-Benchmark/datasets/arithmetic/arithmetic_saq.csv',
        '**/arithmetic_saq.csv',
        '**/TRAM-Benchmark/datasets/arithmetic/arithmetic_mcq.csv',
        '**/arithmetic_mcq.csv',
    ]

    in_path, details = _try_resolve_input(args.input, repo_root, patterns)

    if in_path is None:
        if details.get('mode') == 'auto_locate_multi':
            msg = actionable_error(
                header='[ERR] Multiple candidate files found. Please specify one with --input.',
                repo_root=repo_root,
                cwd=cwd,
                details=details,
                next_cmd='Example: python scripts/prepare_tram.py --input "<one_of_the_candidates>" --dry_run'
            )
            raise RuntimeError(msg)
        msg = actionable_error(
            header='[ERR] Input file not found (no candidates).',
            repo_root=repo_root,
            cwd=cwd,
            details=details,
            next_cmd='PowerShell: Get-ChildItem -Recurse -Filter arithmetic_mcq.csv | Select-Object FullName'
        )
        raise RuntimeError(msg)

    if details.get('mode') == 'auto_locate':
        print(f"Auto-located at: {in_path}")

    cols, rows, enc = read_csv_rows(in_path)
    if enc == 'latin-1':
        print('[WARN] CSV decoded with latin-1 fallback. Please verify text correctness.')

    q_col, a_col = infer_qa_columns(cols)
    if args.question_col:
        q_col = args.question_col
    if args.answer_col:
        a_col = args.answer_col

    if args.dry_run:
        print('=== DRY RUN (probe only) ===')
        print(f"file: {in_path}")
        print(f"encoding: {enc}")
        print('columns:')
        for c in cols:
            print(f"  - {c}")
        print(f"inferred question_col: {q_col}")
        print(f"inferred answer_col:   {a_col}")
        print('--- first 3 rows (raw preview) ---')
        for i, r in enumerate(rows[:3]):
            preview = {
                'idx': i,
                'question_candidate': r.get(q_col) if q_col else None,
                'answer_candidate': r.get(a_col) if a_col else None,
                'category_candidate': r.get('Category', 'unspecified'),
            }
            print(json.dumps(preview, ensure_ascii=False))
        return

    if not q_col or not a_col:
        msg = actionable_error(
            header='[ERR] Failed to infer question/answer columns. Use --question_col / --answer_col to override.',
            repo_root=repo_root,
            cwd=cwd,
            details={
                **details,
                'attempts': details.get('attempts', []) + [f"columns={cols}", f"inferred_q={q_col}", f"inferred_a={a_col}"]
            },
            next_cmd='Example: python scripts/prepare_tram.py --input <file.csv> --question_col Question --answer_col Answer --dry_run'
        )
        raise RuntimeError(msg)

    outp = Path(args.output).expanduser()
    if not outp.is_absolute():
        outp = (repo_root / outp)
    outp = outp.resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    limit = args.limit if args.limit and args.limit > 0 else total

    id_col = args.id_col
    if not id_col:
        for c in cols:
            if c.lower() in ('id', 'qid', 'uid', 'index'):
                id_col = c
                break

    source_file_sha = sha256_file(in_path)

    out_lines: List[str] = []
    category_dist: Dict[str, int] = {}
    gold_mode_dist: Dict[str, int] = {}

    for idx, r in enumerate(rows[:limit]):
        question = str(r.get(q_col) or '')
        answer_raw = r.get(a_col)
        answer_label = str(answer_raw).strip() if answer_raw is not None else ''
        answer_norm = normalize_arithmetic_answer(answer_raw)

        mcq_options = None
        gold_mode = 'raw'
        gold_value = answer_norm

        if is_mcq_label(answer_label):
            mcq_options = mcq_options_from_row(r)
            if mcq_options:
                opt_text = mcq_options[answer_label.upper()]
                gold_value = normalize_arithmetic_answer(opt_text)
                gold_mode = 'mcq_semantic'
            else:
                gold_mode = 'mcq_label'

        category = str(r.get('Category') or 'unspecified').strip() or 'unspecified'

        rid = stable_row_id(r, idx, id_col, question)

        metadata = {
            'source_file': str(in_path),
            'source_file_sha256': source_file_sha,
            'encoding': enc,
            'row_index': idx,
            'tram_fields': r,
            'answer_raw': answer_raw,
            'answer_norm': answer_norm,
            'gold_mode': gold_mode,
        }
        if mcq_options:
            metadata['mcq_answer_label'] = answer_label.upper()
            metadata['mcq_options'] = mcq_options

        obj = {
            'id': rid,
            'task': args.task,
            'category': category,
            'question': question,
            'gold': gold_value,
            'metadata': metadata,
        }
        if args.add_meta_alias:
            obj['meta'] = metadata

        category_dist[category] = category_dist.get(category, 0) + 1
        gold_mode_dist[gold_mode] = gold_mode_dist.get(gold_mode, 0) + 1

        out_lines.append(json.dumps(obj, ensure_ascii=False))

    with outp.open('w', encoding='utf-8', newline='') as f:
        for line in out_lines:
            f.write(line + '\n')

    jsonl_sha = sha256_file(outp)

    audit = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'input_file': str(in_path),
        'input_encoding': enc,
        'output_file': str(outp),
        'task': args.task,
        'rows_total_in_file': total,
        'rows_written': len(out_lines),
        'question_col': q_col,
        'answer_col': a_col,
        'id_col': id_col,
        'repo_root': str(repo_root),
        'cwd': str(cwd),
        'source_file_sha256': source_file_sha,
        'jsonl_sha256': jsonl_sha,
        'converter_git_commit': 'unknown',
        'category_distribution': category_dist,
        'gold_mode_distribution': gold_mode_dist,
    }
    audit_path = outp.with_suffix(outp.suffix + '.audit.json')
    write_audit(audit_path, audit)

    print(f"Wrote JSONL: {outp}")
    print(f"Wrote audit: {audit_path}")

    if args.validate_out:
        ok, errs = validate_jsonl(outp)
        if ok:
            print('[OK] validate_out passed')
        else:
            print('[ERR] validate_out failed')
            for e in errs[:50]:
                print('  - ' + e)
            raise SystemExit(2)


if __name__ == '__main__':
    main()
