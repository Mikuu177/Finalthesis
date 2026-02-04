import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Any, Iterable, Optional
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from repository root if present
load_dotenv()

# --------------------
# Basic FS utilities
# --------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def append_jsonl(path: str | Path, row: Dict[str, Any]):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


# --------------------
# Dataset loader
# --------------------

def load_dataset(dataset_path: str | Path, subset_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load unified dataset JSONL.

    Expected schema (minimum):
      {id, question, gold, category, metadata(optional)}

    Returns at most subset_size rows if provided.
    """
    data = read_jsonl(dataset_path)
    if subset_size is not None and subset_size > 0:
        data = data[: min(subset_size, len(data))]

    normalized = []
    for i, row in enumerate(data):
        rid = row.get('id', i)
        question = row.get('question', row.get('input', ''))
        context = row.get('context', '')
        gold = row.get('gold') if 'gold' in row else row.get('answer')
        category = row.get('category', 'unspecified')
        metadata = row.get('metadata') or row.get('meta') or {}

        normalized.append({
            'id': rid,
            'question': question,
            'context': context,
            'gold': gold,
            'category': category,
            'metadata': metadata,
        })
    return normalized


# --------------------
# Resume support
# --------------------

def load_existing_predictions(pred_path: str | Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not Path(pred_path).exists():
        return existing
    for row in read_jsonl(pred_path):
        k = f"{row.get('model')}::{row.get('id')}"
        existing[k] = row
    return existing


# --------------------
# Simple cache (hash -> file per model)
# --------------------

def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def cache_get(cache_dir: str | Path, model_name: str, prompt_fingerprint: str) -> Optional[Dict[str, Any]]:
    ensure_dir(cache_dir)
    model_dir = ensure_dir(Path(cache_dir) / model_name)
    fp = _hash_key(prompt_fingerprint)
    fpath = model_dir / f"{fp}.json"
    if fpath.exists():
        try:
            return json.loads(fpath.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def cache_put(cache_dir: str | Path, model_name: str, prompt_fingerprint: str, value: Dict[str, Any]):
    ensure_dir(cache_dir)
    model_dir = ensure_dir(Path(cache_dir) / model_name)
    fp = _hash_key(prompt_fingerprint)
    fpath = model_dir / f"{fp}.json"
    fpath.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


# --------------------
# Timestamped run dir
# --------------------

def make_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    ts = time.strftime('%Y%m%d-%H%M%S')
    name = f"{ts}" + (f"_{run_name}" if run_name else '')
    run_dir = ensure_dir(Path(base_dir) / name)
    return run_dir
