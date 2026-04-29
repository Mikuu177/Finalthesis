import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Bootstrapping
# -----------------------------
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRB_ROOT = PROJECT_ROOT / "temporal-reasoning-benchmark"
RUNTIME_ROOT = PROJECT_ROOT / "backend" / "runtime"
TRAIN_RUNTIME_ROOT = RUNTIME_ROOT / "train_jobs"
TEST_RUNTIME_ROOT = RUNTIME_ROOT / "test_runs"
TRAIN_CACHE_INDEX_PATH = RUNTIME_ROOT / "train_cache_index.json"
ANALYSIS_PAYLOAD_PATH = RUNTIME_ROOT / "analysis_payload_strict.json"

if str(TRB_ROOT) not in sys.path:
    sys.path.insert(0, str(TRB_ROOT))

try:
    import joblib
except Exception:
    joblib = None

try:
    from src.prompt_builder import build_prompt
    from src.normalize import normalize_model_answer
    from src.scorer import extract_final_answer, score
    _trb_import_error: Optional[str] = None
except Exception as e:
    build_prompt = None
    normalize_model_answer = None
    extract_final_answer = None
    score = None
    _trb_import_error = str(e)


# -----------------------------
# App
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Environment / Clients
# -----------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

DOUBAO_BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115")

ROUTER_THRESHOLD_DEFAULT = float(os.getenv("ROUTER_FALLBACK_THRESHOLD", "0.95"))
ROUTER_SERVING_MODEL = os.getenv("ROUTER_SERVING_MODEL", "deepseek-chat")

CLASSIFIER_MODEL_PATH = Path(
    os.getenv(
        "CLASSIFIER_MODEL_PATH",
        str(TRB_ROOT / "outputs/classifier_strict/task_clf.joblib"),
    )
).resolve()
PROMPT_BANK_PATH = Path(
    os.getenv("PROMPT_BANK_PATH", str(TRB_ROOT / "configs/prompt_bank.yaml"))
).resolve()
PROMPTS_CFG_PATH = Path(
    os.getenv("PROMPTS_CFG_PATH", str(TRB_ROOT / "configs/prompts.yaml"))
).resolve()
PROMPT_DIR = Path(os.getenv("PROMPT_DIR", str(TRB_ROOT / "prompts"))).resolve()

CLASSIFIER_SPLIT_TRAIN = Path(
    os.getenv(
        "CLASSIFIER_SPLIT_TRAIN",
        str(TRB_ROOT / "data/splits/classifier_router/train.jsonl"),
    )
).resolve()
CLASSIFIER_SPLIT_DEV = Path(
    os.getenv(
        "CLASSIFIER_SPLIT_DEV",
        str(TRB_ROOT / "data/splits/classifier_router/dev.jsonl"),
    )
).resolve()
CLASSIFIER_SPLIT_TEST = Path(
    os.getenv(
        "CLASSIFIER_SPLIT_TEST",
        str(TRB_ROOT / "data/splits/classifier_router/test.jsonl"),
    )
).resolve()

DEFAULT_TEMPORAL_CATEGORIES = [
    "Date Computation",
    "Hour Adjustment (24h)",
    "Time Zone Conversion",
    "Year Shift",
    "Month Shift",
]

STRICT_ANALYSIS_RUNS = [
    {
        "workflow": "Fixed Prompt",
        "run_id": "20260325-220652_baseline_fixed_prompt_strict_pvv2.1",
        "path": TRB_ROOT
        / "outputs/runs/20260325-220652_baseline_fixed_prompt_strict_pvv2.1/predictions.jsonl",
    },
    {
        "workflow": "CoT Prompt",
        "run_id": "20260325-220652_baseline_cot_prompt_strict_pvv2.1",
        "path": TRB_ROOT
        / "outputs/runs/20260325-220652_baseline_cot_prompt_strict_pvv2.1/predictions.jsonl",
    },
    {
        "workflow": "Classifier Router",
        "run_id": "20260325-220653_classifier_prompt_router_strict_pvv2.1",
        "path": TRB_ROOT
        / "outputs/runs/20260325-220653_classifier_prompt_router_strict_pvv2.1/predictions.jsonl",
    },
    {
        "workflow": "Classifier Router + Fallback",
        "run_id": "20260325-220653_classifier_prompt_router_strict_fallback_pvv2.1",
        "path": TRB_ROOT
        / "outputs/runs/20260325-220653_classifier_prompt_router_strict_fallback_pvv2.1/predictions.jsonl",
    },
]

ANALYSIS_RULESET_VERSION = "v1.1"

_MONTH_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

PROBLEM_TYPE_INFO = {
    "Date Computation": {
        "definition": "Compute date shifts across day/month/year constraints.",
        "why_hard": "Needs multi-step calendar reasoning and boundary handling.",
    },
    "Hour Adjustment (24h)": {
        "definition": "Perform hour/minute addition-subtraction under 24-hour format.",
        "why_hard": "Carry/borrow and day-wrap errors are common.",
    },
    "Time Zone Conversion": {
        "definition": "Convert time between source and target time zones.",
        "why_hard": "Direction mistakes and date rollover are frequent.",
    },
    "Year Shift": {
        "definition": "Shift years while preserving valid date semantics.",
        "why_hard": "Leap-year and boundary assumptions can break answers.",
    },
    "Month Shift": {
        "definition": "Shift calendar month(s) from a reference date/question.",
        "why_hard": "Month length mismatch and interpretation ambiguity.",
        "risk_hint": "Fallback is often needed when classifier confidence is low.",
    },
}
PROBLEM_TYPE_INFO["Date Computation"]["risk_hint"] = "Boundary crossing and leap-day handling can fail."
PROBLEM_TYPE_INFO["Hour Adjustment (24h)"]["risk_hint"] = "Borrow/carry mistakes often cause near-miss answers."
PROBLEM_TYPE_INFO["Time Zone Conversion"]["risk_hint"] = "Timezone direction and date rollover are common error sources."
PROBLEM_TYPE_INFO["Year Shift"]["risk_hint"] = "Leap-year assumptions can silently break date validity."


def _usable_key(k: str) -> bool:
    return bool(k and "YOUR_" not in k)


# three-model clients
deepseek_client = (
    openai.AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    if _usable_key(DEEPSEEK_API_KEY)
    else None
)

gpt5_client = (
    openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if _usable_key(OPENAI_API_KEY)
    else None
)

doubao_client = (
    openai.AsyncOpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
    if _usable_key(DOUBAO_API_KEY)
    else None
)


# -----------------------------
# Runtime stores
# -----------------------------
_classifier = None
_prompt_bank = None
_active_classifier_model_path: Path = CLASSIFIER_MODEL_PATH

train_jobs: Dict[str, Dict[str, Any]] = {}
test_runs: Dict[str, Dict[str, Any]] = {}


def _ensure_runtime_dirs() -> None:
    for p in [RUNTIME_ROOT, TRAIN_RUNTIME_ROOT, TEST_RUNTIME_ROOT]:
        p.mkdir(parents=True, exist_ok=True)
    if not TRAIN_CACHE_INDEX_PATH.exists():
        TRAIN_CACHE_INDEX_PATH.write_text("{}", encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prompt_bank() -> Dict[str, Any]:
    global _prompt_bank
    if _prompt_bank is None:
        if not PROMPT_BANK_PATH.exists():
            raise FileNotFoundError(f"PROMPT_BANK_PATH not found: {PROMPT_BANK_PATH}")
        _prompt_bank = yaml.safe_load(PROMPT_BANK_PATH.read_text(encoding="utf-8")) or {}
    return _prompt_bank


def _infer_eval_predictions_path(
    eval_predictions_path: Optional[str],
    eval_report_path: Optional[str],
    model_path: Optional[str],
) -> Optional[str]:
    if eval_predictions_path:
        p = Path(eval_predictions_path)
        if p.exists():
            return str(p)

    candidates: List[Path] = []
    if eval_report_path:
        er = Path(eval_report_path)
        candidates.append(er.with_name("eval_predictions.csv"))
    if model_path:
        mp = Path(model_path)
        candidates.append(mp.parent / "eval" / "eval_predictions.csv")

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _load_classifier_model():
    global _classifier
    if _classifier is None:
        if joblib is None:
            raise RuntimeError("Missing dependency: joblib/scikit-learn not available.")
        if not _active_classifier_model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {_active_classifier_model_path}"
            )
        _classifier = joblib.load(_active_classifier_model_path)
    return _classifier


def _set_active_classifier_model(path: Path) -> None:
    global _active_classifier_model_path, _classifier
    _active_classifier_model_path = path.resolve()
    _classifier = None


def _pick_prompt_cfg(prompt_bank: Dict[str, Any], predicted_category: str) -> Dict[str, Any]:
    default_cfg = prompt_bank.get("default") or {}
    cat_cfg = (prompt_bank.get("by_category") or {}).get(predicted_category) or default_cfg
    return {
        "prompt_mode": cat_cfg.get("prompt_mode", "sp"),
        "n_shots": int(cat_cfg.get("n_shots", 0)),
        "prompt_version": cat_cfg.get("prompt_version", "v2.1"),
    }


def _fallback_cfg(prompt_bank: Dict[str, Any]) -> Dict[str, Any]:
    default_cfg = prompt_bank.get("default") or {}
    return {
        "prompt_mode": default_cfg.get("prompt_mode", "sp"),
        "n_shots": int(default_cfg.get("n_shots", 0)),
        "prompt_version": default_cfg.get("prompt_version", "v2.1"),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _active_model_version() -> str:
    p = _active_classifier_model_path
    if not p.exists():
        return "missing"
    st = p.stat()
    return f"{p.name}@{int(st.st_mtime)}"


def _build_train_cache_key(payload: Dict[str, Any]) -> str:
    train_hash = _sha256_file(CLASSIFIER_SPLIT_TRAIN) if CLASSIFIER_SPLIT_TRAIN.exists() else "missing"
    test_hash = _sha256_file(CLASSIFIER_SPLIT_TEST) if CLASSIFIER_SPLIT_TEST.exists() else "missing"
    key_payload = {
        "train_split": str(CLASSIFIER_SPLIT_TRAIN),
        "test_split": str(CLASSIFIER_SPLIT_TEST),
        "train_hash": train_hash,
        "test_hash": test_hash,
        "categories": sorted(payload.get("categories") or DEFAULT_TEMPORAL_CATEGORIES),
        "min_samples_per_class": int(payload.get("min_samples_per_class", 20)),
        "seed": int(payload.get("seed", 42)),
        "model_type": "tfidf_logreg",
    }
    raw = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_train_cache_index() -> Dict[str, Any]:
    _ensure_runtime_dirs()
    data = _read_json(TRAIN_CACHE_INDEX_PATH, {})
    return data if isinstance(data, dict) else {}


def _save_train_cache_index(index_obj: Dict[str, Any]) -> None:
    _write_json(TRAIN_CACHE_INDEX_PATH, index_obj)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _paginate_rows(rows: List[Dict[str, Any]], page: int, page_size: int) -> Dict[str, Any]:
    p = max(1, int(page))
    ps = max(1, min(int(page_size), 200))
    total = len(rows)
    start = (p - 1) * ps
    end = start + ps
    return {
        "total": total,
        "page": p,
        "page_size": ps,
        "rows": rows[start:end],
    }


def _split_path_by_name(split_name: str) -> Path:
    sn = (split_name or "train").strip().lower()
    if sn == "train":
        return CLASSIFIER_SPLIT_TRAIN
    if sn == "dev":
        return CLASSIFIER_SPLIT_DEV
    if sn == "test":
        return CLASSIFIER_SPLIT_TEST
    raise ValueError("split must be one of: train, dev, test")


def _count_by_category(path: Path) -> Dict[str, int]:
    rows = _read_jsonl(path)
    out: Dict[str, int] = {}
    for r in rows:
        c = str(r.get("category", "unspecified"))
        out[c] = out.get(c, 0) + 1
    return out


def _extract_answer_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    m = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _judge_correct(raw: str, gold: str, sample: Dict[str, Any]) -> bool:
    if normalize_model_answer and extract_final_answer and score:
        try:
            pred = extract_final_answer(raw or "")
            pred_norm = normalize_model_answer(pred, sample)
            corr, _ = score(pred_norm, str(gold or ""), relaxed=False)
            return bool(corr)
        except Exception:
            pass

    pred = _extract_answer_text(raw).lower().replace(" ", "")
    g = str(gold or "").lower().replace(" ", "")
    if not pred or not g:
        return False
    return pred == g or g in pred


def _has_parse(raw: str) -> bool:
    return bool(_extract_answer_text(raw))


def _parse_hhmm(text: str) -> Optional[str]:
    s = (text or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"


def _parse_tz_compact_to_hhmm(text: str) -> Optional[str]:
    s = (text or "").strip()
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(AM|PM)on", s, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ap = m.group(3).upper()
    if hour < 1 or hour > 12 or minute < 0 or minute > 59:
        return None
    if ap == "AM":
        hh = 0 if hour == 12 else hour
    else:
        hh = 12 if hour == 12 else hour + 12
    return f"{hh:02d}:{minute:02d}"


def _normalize_year(text: str) -> Optional[int]:
    s = (text or "").strip()
    m = re.search(r"(-?\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _norm_month_name(text: str) -> Optional[int]:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return _MONTH_NUM.get(s)


def _two_digit_year_to_full(y: int) -> int:
    if y >= 100:
        return y
    return 1900 + y


def _year_candidates(y: int) -> List[int]:
    if y >= 100:
        return [y]
    return [1900 + y, 2000 + y]


def _normalize_year_month(text: str) -> Optional[str]:
    s = (text or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        yy = int(m.group(1))
        mm = int(m.group(2))
        if 1 <= mm <= 12:
            return f"{yy:04d}-{mm:02d}"

    m = re.match(r"^([A-Za-z]{3,9})[-\s]+(\d{1,4})$", s)
    if m:
        mm = _norm_month_name(m.group(1))
        if mm is None:
            return None
        yy = _two_digit_year_to_full(int(m.group(2)))
        return f"{yy:04d}-{mm:02d}"

    m = re.match(r"^([A-Za-z]{3,9})(\d{1,4})$", s)
    if m:
        mm = _norm_month_name(m.group(1))
        if mm is None:
            return None
        yy = _two_digit_year_to_full(int(m.group(2)))
        return f"{yy:04d}-{mm:02d}"

    m = re.match(r"^(\d{1,2})-([A-Za-z]{3,9})$", s)
    if m:
        yy = _two_digit_year_to_full(int(m.group(1)))
        mm = _norm_month_name(m.group(2))
        if mm is None:
            return None
        return f"{yy:04d}-{mm:02d}"
    return None


def _normalize_year_month_from_any(text: str) -> Optional[str]:
    ym = _normalize_year_month(text)
    if ym:
        return ym
    d = _normalize_full_date(text)
    if d:
        return d[:7]
    return None


def _normalize_year_month_candidates(text: str) -> List[str]:
    s = (text or "").strip()
    out: List[str] = []

    m = re.match(r"^([A-Za-z]{3,9})[-\s]+(\d{1,4})$", s)
    if m:
        mm = _norm_month_name(m.group(1))
        yy = int(m.group(2))
        if mm is not None:
            for y in _year_candidates(yy):
                out.append(f"{y:04d}-{mm:02d}")
        return sorted(set(out))

    m = re.match(r"^(\d{1,2})-([A-Za-z]{3,9})$", s)
    if m:
        yy = int(m.group(1))
        mm = _norm_month_name(m.group(2))
        if mm is not None:
            for y in _year_candidates(yy):
                out.append(f"{y:04d}-{mm:02d}")
        return sorted(set(out))

    m = re.match(r"^([A-Za-z]{3,9})(\d{1,4})$", s)
    if m:
        mm = _norm_month_name(m.group(1))
        yy = int(m.group(2))
        if mm is not None:
            for y in _year_candidates(yy):
                out.append(f"{y:04d}-{mm:02d}")
        return sorted(set(out))

    norm = _normalize_year_month_from_any(text)
    return [norm] if norm else []


def _normalize_full_date(text: str) -> Optional[str]:
    s = (text or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"

    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None


def _normalize_by_category(category: str, text: str) -> Optional[str]:
    cat = (category or "").strip()
    if cat == "Time Zone Conversion":
        hh = _parse_hhmm(text)
        if hh:
            return hh
        return _parse_tz_compact_to_hhmm(text)

    if cat == "Hour Adjustment (24h)":
        return _parse_hhmm(text)

    if cat == "Year Shift":
        y = _normalize_year(text)
        return str(y) if y is not None else None

    if cat == "Month Shift":
        mm = _norm_month_name(text)
        if mm is not None:
            return str(mm)
        ym = _normalize_year_month_from_any(text)
        if ym and re.match(r"^\d{4}-(\d{2})$", ym):
            return str(int(ym.split("-")[1]))
        return None

    if cat == "Date Computation":
        d = _normalize_full_date(text)
        if d:
            return d
        ym = _normalize_year_month(text)
        if ym:
            return ym
        return None

    return None


def _build_gold_index_from_split() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not CLASSIFIER_SPLIT_TEST.exists():
        return idx
    for row in _read_jsonl(CLASSIFIER_SPLIT_TEST):
        rid = str(row.get("id", ""))
        if not rid:
            continue
        idx[rid] = row
    return idx


def _correction_result(row: Dict[str, Any], split_ref: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    category = str(row.get("true_category") or row.get("category") or (split_ref or {}).get("category") or "unspecified")
    gold_raw = str((split_ref or {}).get("gold", row.get("gold", "")))
    pred_norm = str(row.get("pred_norm", ""))
    base_correct = bool(row.get("correct"))

    if base_correct:
        return {
            "corrected_correct": True,
            "reason": "base_correct",
            "category": category,
            "gold_raw": gold_raw,
            "gold_norm_by_cat": _normalize_by_category(category, gold_raw),
            "pred_norm": pred_norm,
            "pred_norm_by_cat": _normalize_by_category(category, pred_norm),
        }

    gold_c = _normalize_by_category(category, gold_raw)
    pred_c = _normalize_by_category(category, pred_norm)
    corrected_correct = bool(gold_c and pred_c and gold_c == pred_c)
    if category == "Date Computation" and not corrected_correct:
        pred_ym = _normalize_year_month_from_any(pred_norm)
        if pred_ym:
            gold_ym_candidates = _normalize_year_month_candidates(gold_raw)
            if pred_ym in gold_ym_candidates:
                corrected_correct = True
                gold_c = "|".join(gold_ym_candidates)
                pred_c = pred_ym
    reason = "unchanged_wrong"
    if corrected_correct:
        if category == "Date Computation":
            reason = "fix_date_equiv"
        elif category == "Time Zone Conversion":
            reason = "fix_tz_hhmm_equiv"
        elif category == "Hour Adjustment (24h)":
            reason = "fix_hhmm_padding_equiv"
        elif category == "Year Shift":
            reason = "fix_year_numeric_equiv"
        elif category == "Month Shift":
            reason = "fix_month_name_equiv"

    return {
        "corrected_correct": corrected_correct,
        "reason": reason,
        "category": category,
        "gold_raw": gold_raw,
        "gold_norm_by_cat": gold_c,
        "pred_norm": pred_norm,
        "pred_norm_by_cat": pred_c,
    }


def _read_workflow_metrics_from_table() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    path = TRB_ROOT / "outputs/tables/prompt_routing_comparison_strict.csv"
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = str(r.get("run_id", ""))
            if not rid:
                continue
            out[rid] = {
                "parse_rate": float(r.get("parse_rate", 0) or 0),
                "format_compliance": float(r.get("format_compliance", 0) or 0),
                "latency_sec_per_query": float(r.get("latency_sec_per_query", 0) or 0),
                "calls_per_query": float(r.get("calls_per_query", 0) or 0),
            }
    return out


def _read_oracle_metric() -> Dict[str, Any]:
    path = TRB_ROOT / "outputs/tables/prompt_oracle_upper_bound_strict.csv"
    if not path.exists():
        path = TRB_ROOT / "outputs/tables/third_round_summary.csv"
    if not path.exists():
        return {
            "name": "Oracle Prompt Upper Bound",
            "accuracy": None,
            "definition": "Upper bound under category-best prompt selection",
            "interpretation": "Not a deployable online policy",
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wf = str(row.get("workflow", ""))
            if "Oracle" not in wf:
                continue
            acc = row.get("accuracy")
            return {
                "name": "Oracle Prompt Upper Bound",
                "accuracy": float(acc) if acc not in ("", None) else None,
                "definition": "Upper bound under category-best prompt selection",
                "interpretation": "Not a deployable online policy",
            }
    return {
        "name": "Oracle Prompt Upper Bound",
        "accuracy": None,
        "definition": "Upper bound under category-best prompt selection",
        "interpretation": "Not a deployable online policy",
    }


def _make_insight_tag(category_row: Dict[str, Any]) -> str:
    values = [
        category_row.get("fixed_corrected", 0.0),
        category_row.get("cot_corrected", 0.0),
        category_row.get("router_corrected", 0.0),
        category_row.get("router_fallback_corrected", 0.0),
    ]
    if all(abs(v - 1.0) < 1e-9 for v in values):
        return "Ceiling category"
    if max(values) <= 0.0:
        return "Shared bottleneck"
    if category_row.get("router_fallback_corrected", 0.0) >= category_row.get("fixed_corrected", 0.0):
        return "Recovered by fallback"
    return "Prompt gap remains"


def _compute_analysis_payload() -> Dict[str, Any]:
    split_idx = _build_gold_index_from_split()
    metrics_idx = _read_workflow_metrics_from_table()
    now = datetime.now().isoformat()

    workflow_summary: List[Dict[str, Any]] = []
    category_accumulator: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "category": "",
            "fixed_n": 0,
            "fixed_base_correct": 0,
            "fixed_corr_correct": 0,
            "cot_n": 0,
            "cot_base_correct": 0,
            "cot_corr_correct": 0,
            "router_n": 0,
            "router_base_correct": 0,
            "router_corr_correct": 0,
            "router_fallback_n": 0,
            "router_fallback_base_correct": 0,
            "router_fallback_corr_correct": 0,
        }
    )
    error_rows: List[Dict[str, Any]] = []
    correction_breakdown: Dict[str, int] = defaultdict(int)

    wf_key_map = {
        "Fixed Prompt": "fixed",
        "CoT Prompt": "cot",
        "Classifier Router": "router",
        "Classifier Router + Fallback": "router_fallback",
    }

    for spec in STRICT_ANALYSIS_RUNS:
        wf = spec["workflow"]
        run_id = spec["run_id"]
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing strict run predictions: {path}")
        rows = _read_jsonl(path)
        n = len(rows)
        base_correct = 0
        corrected_correct = 0
        wk = wf_key_map[wf]
        for r in rows:
            rid = str(r.get("id", ""))
            split_ref = split_idx.get(rid)
            c = _correction_result(r, split_ref)
            corr_ok = bool(c["corrected_correct"])
            base_ok = bool(r.get("correct"))
            base_correct += int(base_ok)
            corrected_correct += int(corr_ok)
            correction_breakdown[c["reason"]] += 1

            cat = c["category"]
            d = category_accumulator[cat]
            d["category"] = cat
            d[f"{wk}_n"] += 1
            d[f"{wk}_base_correct"] += int(base_ok)
            d[f"{wk}_corr_correct"] += int(corr_ok)

            if not base_ok:
                error_rows.append(
                    {
                        "sample_id": rid,
                        "category": cat,
                        "workflow": wf,
                        "gold": c["gold_raw"],
                        "pred_norm": c["pred_norm"],
                        "corrected_match": corr_ok,
                        "error_type": c["reason"] if corr_ok else "unchanged_wrong",
                    }
                )

        m = metrics_idx.get(run_id, {})
        base_acc = (base_correct / n) if n else 0.0
        corr_acc = (corrected_correct / n) if n else 0.0
        workflow_summary.append(
            {
                "workflow": wf,
                "run_id": run_id,
                "sample_count": n,
                "base_accuracy": round(base_acc, 4),
                "corrected_accuracy": round(corr_acc, 4),
                "delta": round(corr_acc - base_acc, 4),
                "parse_rate": round(float(m.get("parse_rate", 0.0)), 4),
                "format_compliance": round(float(m.get("format_compliance", 0.0)), 4),
                "latency_sec_per_query": round(float(m.get("latency_sec_per_query", 0.0)), 4),
                "calls_per_query": round(float(m.get("calls_per_query", 0.0)), 4),
            }
        )

    category_summary: List[Dict[str, Any]] = []
    for cat in sorted(category_accumulator.keys()):
        d = category_accumulator[cat]
        row = {
            "category": cat,
            "fixed_base": round((d["fixed_base_correct"] / d["fixed_n"]) if d["fixed_n"] else 0.0, 4),
            "fixed_corrected": round((d["fixed_corr_correct"] / d["fixed_n"]) if d["fixed_n"] else 0.0, 4),
            "cot_base": round((d["cot_base_correct"] / d["cot_n"]) if d["cot_n"] else 0.0, 4),
            "cot_corrected": round((d["cot_corr_correct"] / d["cot_n"]) if d["cot_n"] else 0.0, 4),
            "router_base": round((d["router_base_correct"] / d["router_n"]) if d["router_n"] else 0.0, 4),
            "router_corrected": round((d["router_corr_correct"] / d["router_n"]) if d["router_n"] else 0.0, 4),
            "router_fallback_base": round(
                (d["router_fallback_base_correct"] / d["router_fallback_n"]) if d["router_fallback_n"] else 0.0,
                4,
            ),
            "router_fallback_corrected": round(
                (d["router_fallback_corr_correct"] / d["router_fallback_n"]) if d["router_fallback_n"] else 0.0,
                4,
            ),
        }
        row["fixed_delta"] = round(row["fixed_corrected"] - row["fixed_base"], 4)
        row["cot_delta"] = round(row["cot_corrected"] - row["cot_base"], 4)
        row["router_delta"] = round(row["router_corrected"] - row["router_base"], 4)
        row["router_fallback_delta"] = round(row["router_fallback_corrected"] - row["router_fallback_base"], 4)
        row["insight_tag"] = _make_insight_tag(row)
        category_summary.append(row)

    payload = {
        "source": {
            "strict_runs": [x["run_id"] for x in STRICT_ANALYSIS_RUNS],
            "comparison_table": str(TRB_ROOT / "outputs/tables/prompt_routing_comparison_strict.csv"),
            "split_test": str(CLASSIFIER_SPLIT_TEST),
        },
        "ruleset_version": ANALYSIS_RULESET_VERSION,
        "generated_at": now,
        "workflow_summary": workflow_summary,
        "category_summary": category_summary,
        "error_rows": error_rows,
        "correction_breakdown": dict(sorted(correction_breakdown.items())),
        "oracle": _read_oracle_metric(),
    }
    _write_json(ANALYSIS_PAYLOAD_PATH, payload)
    return payload


def _load_analysis_payload() -> Dict[str, Any]:
    if ANALYSIS_PAYLOAD_PATH.exists():
        cached = _read_json(ANALYSIS_PAYLOAD_PATH, {})
        if cached and cached.get("ruleset_version") == ANALYSIS_RULESET_VERSION:
            return cached
    return _compute_analysis_payload()


# -----------------------------
# Schemas
# -----------------------------
class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class QueryWithRouterRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    threshold: Optional[float] = None


class ClassifyQuestionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = 5


class TrainStartRequest(BaseModel):
    categories: Optional[List[str]] = None
    threshold: Optional[float] = None
    min_samples_per_class: int = 20
    seed: int = 42


class TestRunRequest(BaseModel):
    categories: Optional[List[str]] = None
    limit_per_category: int = 3
    threshold: Optional[float] = None
    suite_name: str = "temporal_core_suite"


class ProblemExamplesRequest(BaseModel):
    category: str
    limit: int = 3


class ModelResponse(BaseModel):
    model: str
    response: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class RouterResponse(BaseModel):
    model: str
    response: str
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    threshold: float
    fallback_triggered: bool
    selected_prompt_mode: Optional[str] = None
    selected_prompt_version: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ClassifyQuestionResponse(BaseModel):
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: List[Dict[str, Any]]
    model_path: Optional[str] = None
    error: Optional[str] = None


# -----------------------------
# Model query helpers
# -----------------------------
async def _query_chat(client, model_name: str, prompt: str, label: str) -> ModelResponse:
    if client is None:
        return ModelResponse(model=label, response="", latency_ms=None, error="API key not configured")

    t0 = time.perf_counter()
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        dt = (time.perf_counter() - t0) * 1000
        text = completion.choices[0].message.content if completion.choices else ""
        return ModelResponse(model=label, response=text or "", latency_ms=round(dt, 2), error=None)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return ModelResponse(model=label, response="", latency_ms=round(dt, 2), error=str(e))


async def query_deepseek(prompt: str) -> ModelResponse:
    return await _query_chat(deepseek_client, "deepseek-chat", prompt, "DeepSeek")


async def query_gpt5mini(prompt: str) -> ModelResponse:
    return await _query_chat(gpt5_client, OPENAI_MODEL, prompt, "GPT-5-mini")


async def query_doubao(prompt: str) -> ModelResponse:
    return await _query_chat(doubao_client, DOUBAO_MODEL, prompt, "Doubao")


async def _query_with_router_core(prompt: str, threshold: float) -> RouterResponse:
    if deepseek_client is None:
        return RouterResponse(
            model="Classifier Router + Fallback",
            response="",
            predicted_category=None,
            confidence=None,
            threshold=threshold,
            fallback_triggered=False,
            selected_prompt_mode=None,
            selected_prompt_version=None,
            latency_ms=None,
            error="DeepSeek API key not configured for router serving model",
        )

    if build_prompt is None:
        detail = f": {_trb_import_error}" if _trb_import_error else ""
        return RouterResponse(
            model="Classifier Router + Fallback",
            response="",
            predicted_category=None,
            confidence=None,
            threshold=threshold,
            fallback_triggered=False,
            selected_prompt_mode=None,
            selected_prompt_version=None,
            latency_ms=None,
            error=f"Missing temporal-reasoning-benchmark dependencies (prompt builder not importable){detail}",
        )

    try:
        clf = _load_classifier_model()
        prompt_bank = _load_prompt_bank()

        text = prompt.strip()
        pred_cat = str(clf.predict([text])[0])

        conf = None
        if hasattr(clf, "predict_proba"):
            try:
                probs = clf.predict_proba([text])[0]
                conf = float(max(probs))
            except Exception:
                conf = None

        selected = _pick_prompt_cfg(prompt_bank, pred_cat)
        fallback_triggered = False

        if conf is None or conf < threshold:
            selected = _fallback_cfg(prompt_bank)
            fallback_triggered = True

        sample = {
            "id": "live_query",
            "category": pred_cat,
            "question": text,
            "context": "",
            "_prompt_version": selected["prompt_version"],
        }

        full_prompt, _, _ = build_prompt(
            sample=sample,
            prompts_cfg_path=str(PROMPTS_CFG_PATH),
            mode=selected["prompt_mode"],
            n_shots=selected["n_shots"],
            prompt_dir=str(PROMPT_DIR),
            seed=42,
        )

        t0 = time.perf_counter()
        completion = await deepseek_client.chat.completions.create(
            model=ROUTER_SERVING_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        answer = completion.choices[0].message.content if completion.choices else ""

        return RouterResponse(
            model="Classifier Router + Fallback",
            response=answer or "",
            predicted_category=pred_cat,
            confidence=conf,
            threshold=threshold,
            fallback_triggered=fallback_triggered,
            selected_prompt_mode=selected["prompt_mode"],
            selected_prompt_version=selected["prompt_version"],
            latency_ms=latency_ms,
            error=None,
        )
    except Exception as e:
        return RouterResponse(
            model="Classifier Router + Fallback",
            response="",
            predicted_category=None,
            confidence=None,
            threshold=threshold,
            fallback_triggered=False,
            selected_prompt_mode=None,
            selected_prompt_version=None,
            latency_ms=None,
            error=str(e),
        )


def _classify_prompt(prompt: str, top_k: int = 5) -> ClassifyQuestionResponse:
    try:
        clf = _load_classifier_model()
        text = prompt.strip()
        pred_cat = str(clf.predict([text])[0])

        probs_out: List[Dict[str, Any]] = []
        conf = None
        if hasattr(clf, "predict_proba"):
            try:
                probs = clf.predict_proba([text])[0]
                labels = getattr(clf, "classes_", [])
                pairs = list(zip(labels, probs))
                pairs.sort(key=lambda x: float(x[1]), reverse=True)
                for cat, p in pairs[: max(1, top_k)]:
                    probs_out.append({"category": str(cat), "probability": float(p)})
                if pairs:
                    conf = float(pairs[0][1])
            except Exception:
                probs_out = []
                conf = None

        return ClassifyQuestionResponse(
            predicted_category=pred_cat,
            confidence=conf,
            probabilities=probs_out,
            model_path=str(_active_classifier_model_path),
            error=None,
        )
    except Exception as e:
        return ClassifyQuestionResponse(
            predicted_category=None,
            confidence=None,
            probabilities=[],
            model_path=str(_active_classifier_model_path),
            error=str(e),
        )


# -----------------------------
# Problem types / examples
# -----------------------------
def _load_problem_examples(limit_per_category: int, categories: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    rows = _read_jsonl(CLASSIFIER_SPLIT_TEST)
    allow = set(categories or DEFAULT_TEMPORAL_CATEGORIES)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        cat = str(row.get("category", ""))
        if cat not in allow:
            continue
        if len(grouped[cat]) >= limit_per_category:
            continue
        grouped[cat].append(
            {
                "id": str(row.get("id")),
                "question": row.get("question", ""),
                "gold": str(row.get("gold", "")),
                "source_split": "classifier_router/test.jsonl",
                "notes": "TRAM split sample",
                "category": cat,
            }
        )

    for cat in allow:
        grouped.setdefault(cat, [])

    return grouped


# -----------------------------
# Training job execution
# -----------------------------
async def _run_train_job(job_id: str, job_dir: Path, payload: Dict[str, Any], cache_key: str) -> None:
    job = train_jobs[job_id]
    t0 = time.time()
    py = sys.executable

    out_dir = TRB_ROOT / "outputs" / "workbench_train" / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = {
        "train_dataset_path": str(CLASSIFIER_SPLIT_TRAIN.relative_to(TRB_ROOT)),
        "eval_dataset_path": str(CLASSIFIER_SPLIT_TEST.relative_to(TRB_ROOT)),
        "output_dir": str(out_dir.relative_to(TRB_ROOT)),
        "model_filename": "task_clf.joblib",
        "include_categories": sorted(payload.get("categories") or DEFAULT_TEMPORAL_CATEGORIES),
        "min_samples_per_class": int(payload.get("min_samples_per_class", 20)),
        "seed": int(payload.get("seed", 42)),
    }

    eval_dir = out_dir / "eval"
    eval_cfg = {
        "model_path": str((out_dir / "task_clf.joblib").relative_to(TRB_ROOT)),
        "eval_dataset_path": str(CLASSIFIER_SPLIT_TEST.relative_to(TRB_ROOT)),
        "output_dir": str(eval_dir.relative_to(TRB_ROOT)),
    }

    train_cfg_path = job_dir / "train_config.yaml"
    eval_cfg_path = job_dir / "eval_config.yaml"
    _write_json(job_dir / "train_payload.json", payload)
    train_cfg_path.write_text(yaml.safe_dump(train_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    job["state"] = "running"
    job["stage"] = "preparing"
    job["progress"] = 0.1
    job["log"] = ["Training job started."]

    try:
        job["stage"] = "training_classifier"
        train_cmd = [py, str(TRB_ROOT / "scripts/train_task_classifier.py"), "--config", str(train_cfg_path)]
        proc1 = await asyncio.create_subprocess_exec(
            *train_cmd,
            cwd=str(TRB_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out1, _ = await proc1.communicate()
        text1 = out1.decode("utf-8", errors="ignore")
        (job_dir / "train.log").write_text(text1, encoding="utf-8")
        job["log"].append("train_task_classifier.py finished")

        if proc1.returncode != 0:
            raise RuntimeError(f"train_task_classifier failed (code {proc1.returncode})")

        job["progress"] = 0.65

        job["stage"] = "evaluating_on_holdout"
        eval_cmd = [py, str(TRB_ROOT / "scripts/eval_task_classifier.py"), "--config", str(eval_cfg_path)]
        proc2 = await asyncio.create_subprocess_exec(
            *eval_cmd,
            cwd=str(TRB_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out2, _ = await proc2.communicate()
        text2 = out2.decode("utf-8", errors="ignore")
        (job_dir / "eval.log").write_text(text2, encoding="utf-8")
        job["log"].append("eval_task_classifier.py finished")

        if proc2.returncode != 0:
            raise RuntimeError(f"eval_task_classifier failed (code {proc2.returncode})")

        model_path = out_dir / "task_clf.joblib"
        train_report_path = out_dir / "classifier_report.json"
        eval_report_path = eval_dir / "eval_report.json"
        eval_predictions_path = eval_dir / "eval_predictions.csv"
        conf_matrix_path = out_dir / "confusion_matrix.csv"

        if not model_path.exists() or not train_report_path.exists():
            raise RuntimeError("Training artifacts not found after scripts finished.")

        train_report = _read_json(train_report_path, {})
        eval_report = _read_json(eval_report_path, {})

        job["stage"] = "registering_artifacts"
        _set_active_classifier_model(model_path)

        idx = _load_train_cache_index()
        idx[cache_key] = {
            "cache_key": cache_key,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "report_path": str(train_report_path),
            "eval_report_path": str(eval_report_path),
            "eval_predictions_path": str(eval_predictions_path),
            "confusion_matrix_path": str(conf_matrix_path),
            "metrics": {
                "accuracy": (eval_report.get("accuracy") if eval_report else (train_report.get("metrics") or {}).get("accuracy")),
                "macro_f1": (eval_report.get("macro_f1") if eval_report else (train_report.get("metrics") or {}).get("macro_f1")),
                "n_train": train_report.get("n_train"),
                "n_test": train_report.get("n_test"),
            },
            "config": {
                "categories": train_cfg.get("include_categories"),
                "min_samples_per_class": train_cfg.get("min_samples_per_class"),
                "seed": train_cfg.get("seed"),
            },
        }
        _save_train_cache_index(idx)

        duration = round(time.time() - t0, 3)
        cls_report_file = out_dir / "classification_report.txt"

        job.update(
            {
                "state": "completed",
                "stage": "completed",
                "progress": 1.0,
                "cache_hit": False,
                "cache_reason": "cache_miss_new_training_executed",
                "model_path": str(model_path),
                "report_path": str(train_report_path),
                "eval_report_path": str(eval_report_path),
                "eval_predictions_path": str(eval_predictions_path),
                "confusion_matrix_path": str(conf_matrix_path),
                "metrics": idx[cache_key]["metrics"],
                "active_model_version": _active_model_version(),
                "classification_report_preview": cls_report_file.read_text(encoding="utf-8")[:1200]
                if cls_report_file.exists()
                else "",
                "duration_sec": duration,
                "error": None,
                "updated_at": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        duration = round(time.time() - t0, 3)
        train_log = job_dir / "train.log"
        eval_log = job_dir / "eval.log"
        log_tail_parts = []
        if train_log.exists():
            log_tail_parts.append(train_log.read_text(encoding="utf-8")[-1200:])
        if eval_log.exists():
            log_tail_parts.append(eval_log.read_text(encoding="utf-8")[-1200:])
        job.update(
            {
                "state": "failed",
                "stage": "failed",
                "progress": 1.0,
                "duration_sec": duration,
                "error": str(e),
                "cache_reason": "training_pipeline_failed",
                "active_model_version": _active_model_version(),
                "classification_report_preview": "",
                "log_tail": "\n---\n".join(log_tail_parts),
                "updated_at": datetime.now().isoformat(),
            }
        )


# -----------------------------
# Test suite execution
# -----------------------------
async def _run_test_suite(run_id: str, payload: Dict[str, Any]) -> None:
    run = test_runs[run_id]
    run["state"] = "running"
    run["progress"] = 0.05
    run["updated_at"] = datetime.now().isoformat()

    categories = payload.get("categories") or DEFAULT_TEMPORAL_CATEGORIES
    limit_per_category = int(payload.get("limit_per_category", 3))
    threshold = float(payload.get("threshold", ROUTER_THRESHOLD_DEFAULT))

    grouped = _load_problem_examples(limit_per_category=limit_per_category, categories=categories)
    rows: List[Dict[str, Any]] = []
    sample_pool: List[Dict[str, Any]] = []
    for cat in categories:
        sample_pool.extend(grouped.get(cat, []))

    total = len(sample_pool)
    if total == 0:
        run.update(
            {
                "state": "failed",
                "progress": 1.0,
                "error": "No samples found for selected categories.",
                "updated_at": datetime.now().isoformat(),
            }
        )
        return

    stat = {
        "DeepSeek": {"n": 0, "correct": 0, "parse": 0, "latency": 0.0, "calls": 0},
        "GPT-5-mini": {"n": 0, "correct": 0, "parse": 0, "latency": 0.0, "calls": 0},
        "Doubao": {"n": 0, "correct": 0, "parse": 0, "latency": 0.0, "calls": 0},
        "Classifier Router + Fallback": {"n": 0, "correct": 0, "parse": 0, "latency": 0.0, "calls": 0},
    }
    router_trigger_count = 0
    fallback_count = 0
    error_rows: List[Dict[str, Any]] = []

    t0 = time.time()

    for i, sample in enumerate(sample_pool, start=1):
        prompt = sample["question"]
        gold = sample.get("gold", "")

        deepseek_res, gpt_res, doubao_res, router_res = await asyncio.gather(
            query_deepseek(prompt),
            query_gpt5mini(prompt),
            query_doubao(prompt),
            _query_with_router_core(prompt, threshold),
        )

        systems = [deepseek_res, gpt_res, doubao_res, router_res]
        model_rows: Dict[str, Any] = {}

        for sys_res in systems:
            correct = False if sys_res.error else _judge_correct(sys_res.response, gold, sample)
            parse_ok = False if sys_res.error else _has_parse(sys_res.response)
            stat[sys_res.model]["n"] += 1
            stat[sys_res.model]["correct"] += int(correct)
            stat[sys_res.model]["parse"] += int(parse_ok)
            stat[sys_res.model]["calls"] += 1
            if sys_res.latency_ms is not None:
                stat[sys_res.model]["latency"] += float(sys_res.latency_ms)

            model_rows[sys_res.model] = {
                "response": sys_res.response,
                "latency_ms": sys_res.latency_ms,
                "error": sys_res.error,
                "correct": correct,
                "parse_ok": parse_ok,
            }

        rows.append(
            {
                "sample_id": sample["id"],
                "category": sample["category"],
                "question": prompt,
                "gold": gold,
                "outputs": {
                    "deepseek": model_rows.get("DeepSeek", {}),
                    "gpt5mini": model_rows.get("GPT-5-mini", {}),
                    "doubao": model_rows.get("Doubao", {}),
                    "router": model_rows.get("Classifier Router + Fallback", {}),
                },
                "router_meta": {
                    "predicted_category": router_res.predicted_category,
                    "confidence": router_res.confidence,
                    "threshold": router_res.threshold,
                    "fallback_triggered": router_res.fallback_triggered,
                    "selected_prompt_mode": router_res.selected_prompt_mode,
                    "selected_prompt_version": router_res.selected_prompt_version,
                },
            }
        )
        if router_res.predicted_category:
            router_trigger_count += 1
        if router_res.fallback_triggered:
            fallback_count += 1

        for wf, key in [
            ("DeepSeek", "deepseek"),
            ("GPT-5-mini", "gpt5mini"),
            ("Doubao", "doubao"),
            ("Classifier Router + Fallback", "router"),
        ]:
            out = model_rows.get(wf, {})
            if out and (not out.get("correct")):
                error_rows.append(
                    {
                        "sample_id": sample["id"],
                        "category": sample["category"],
                        "workflow": wf,
                        "gold": str(gold),
                        "pred": _extract_answer_text(out.get("response", "")),
                        "error_type": "parse_error" if not out.get("parse_ok") else "wrong_answer",
                    }
                )

        run["progress"] = round(0.05 + 0.9 * (i / total), 4)
        run["updated_at"] = datetime.now().isoformat()

    elapsed = round(time.time() - t0, 3)

    summary = {}
    for model_name, s in stat.items():
        n = max(1, s["n"])
        summary[model_name] = {
            "accuracy": round(s["correct"] / n, 4),
            "parse_rate": round(s["parse"] / n, 4),
            "latency_ms": round(s["latency"] / n, 2) if s["latency"] > 0 else None,
            "calls_per_query": round(s["calls"] / n, 2),
            "correct_count": s["correct"],
            "sample_count": s["n"],
        }

    run.update(
        {
            "state": "completed",
            "progress": 1.0,
            "duration_sec": elapsed,
            "rows": rows,
            "summary": summary,
            "router_trigger_rate": round(router_trigger_count / total, 4),
            "fallback_rate": round(fallback_count / total, 4),
            "error_rows": error_rows[:400],
            "error": None,
            "updated_at": datetime.now().isoformat(),
        }
    )

    out_path = TEST_RUNTIME_ROOT / run_id / "result.json"
    _write_json(out_path, run)


# -----------------------------
# API: existing real-time routes
# -----------------------------
@app.post("/api/query", response_model=List[ModelResponse])
async def run_queries(query: QueryRequest):
    tasks = [
        query_deepseek(query.prompt),
        query_gpt5mini(query.prompt),
        query_doubao(query.prompt),
    ]
    return await asyncio.gather(*tasks)


@app.post("/api/query_with_router", response_model=RouterResponse)
async def query_with_router(req: QueryWithRouterRequest):
    threshold = ROUTER_THRESHOLD_DEFAULT if req.threshold is None else float(req.threshold)
    return await _query_with_router_core(req.prompt, threshold)


@app.post("/api/classify_question", response_model=ClassifyQuestionResponse)
def classify_question(req: ClassifyQuestionRequest):
    top_k = max(1, min(int(req.top_k or 5), 10))
    return _classify_prompt(req.prompt, top_k=top_k)


# -----------------------------
# API: problem types / examples
# -----------------------------
@app.get("/api/problem_types")
def get_problem_types():
    grouped = _load_problem_examples(limit_per_category=999, categories=DEFAULT_TEMPORAL_CATEGORIES)
    rows = []
    for cat in DEFAULT_TEMPORAL_CATEGORIES:
        info = PROBLEM_TYPE_INFO.get(cat, {})
        rows.append(
            {
                "category": cat,
                "definition": info.get("definition", ""),
                "why_hard": info.get("why_hard", ""),
                "risk_hint": info.get("risk_hint", ""),
                "example_count": len(grouped.get(cat, [])),
            }
        )
    return {
        "source": str(CLASSIFIER_SPLIT_TEST),
        "problem_types": rows,
    }


@app.get("/api/problem_types/{category}/examples")
def get_problem_type_examples(category: str, limit: int = 3):
    grouped = _load_problem_examples(limit_per_category=max(1, min(limit, 10)), categories=[category])
    return {
        "category": category,
        "limit": limit,
        "examples": grouped.get(category, []),
    }


# -----------------------------
# API: train jobs
# -----------------------------
@app.post("/api/train/start")
async def train_start(req: TrainStartRequest):
    _ensure_runtime_dirs()
    payload = req.model_dump()
    categories = payload.get("categories") or DEFAULT_TEMPORAL_CATEGORIES
    payload["categories"] = categories

    cache_key = _build_train_cache_key(payload)
    cache_index = _load_train_cache_index()

    if cache_key in cache_index:
        cached = cache_index[cache_key]
        model_path = Path(cached.get("model_path", ""))
        report_path = Path(cached.get("report_path", ""))
        if model_path.exists() and report_path.exists():
            inferred_eval_predictions = _infer_eval_predictions_path(
                cached.get("eval_predictions_path"),
                cached.get("eval_report_path"),
                str(model_path),
            )
            _set_active_classifier_model(model_path)
            cached_cls_report = report_path.with_name("classification_report.txt")
            job_id = f"train-{uuid.uuid4().hex[:8]}"
            train_jobs[job_id] = {
                "job_id": job_id,
                "state": "completed",
                "stage": "completed",
                "progress": 1.0,
                "cache_hit": True,
                "cache_key": cache_key,
                "cache_reason": "cache_hit_same_split_and_config",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "model_path": str(model_path),
                "report_path": str(report_path),
                "eval_report_path": cached.get("eval_report_path"),
                "eval_predictions_path": inferred_eval_predictions,
                "confusion_matrix_path": cached.get("confusion_matrix_path"),
                "metrics": cached.get("metrics") or {},
                "active_model_version": _active_model_version(),
                "classification_report_preview": cached_cls_report.read_text(encoding="utf-8")[:1200]
                if cached_cls_report.exists()
                else "",
                "duration_sec": 0.0,
                "error": None,
                "log": ["Cache hit: existing training artifacts reused."],
            }
            if inferred_eval_predictions and (cached.get("eval_predictions_path") != inferred_eval_predictions):
                cached["eval_predictions_path"] = inferred_eval_predictions
                cache_index[cache_key] = cached
                _save_train_cache_index(cache_index)
            return train_jobs[job_id]

    job_id = f"train-{uuid.uuid4().hex[:8]}"
    job_dir = TRAIN_RUNTIME_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    train_jobs[job_id] = {
        "job_id": job_id,
        "state": "queued",
        "stage": "queued",
        "progress": 0.0,
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_reason": "cache_miss_new_or_changed_config",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "model_path": None,
        "report_path": None,
        "eval_report_path": None,
        "eval_predictions_path": None,
        "confusion_matrix_path": None,
        "metrics": None,
        "active_model_version": _active_model_version(),
        "classification_report_preview": "",
        "duration_sec": None,
        "error": None,
        "log": ["Job queued."],
    }

    asyncio.create_task(_run_train_job(job_id, job_dir, payload, cache_key))
    return train_jobs[job_id]


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str):
    job = train_jobs.get(job_id)
    if not job:
        return {"job_id": job_id, "state": "not_found", "error": "job_id not found"}
    return job


@app.get("/api/train/spec")
def train_spec():
    return {
        "model_architecture": "TF-IDF + LogisticRegression",
        "split_paths": {
            "train": str(CLASSIFIER_SPLIT_TRAIN),
            "dev": str(CLASSIFIER_SPLIT_DEV),
            "test": str(CLASSIFIER_SPLIT_TEST),
        },
        "split_counts": {
            "train": _count_jsonl(CLASSIFIER_SPLIT_TRAIN),
            "dev": _count_jsonl(CLASSIFIER_SPLIT_DEV),
            "test": _count_jsonl(CLASSIFIER_SPLIT_TEST),
        },
        "active_model_path": str(_active_classifier_model_path),
        "active_model_version": _active_model_version(),
    }


@app.get("/api/train/categories_summary")
def train_categories_summary():
    train_counts = _count_by_category(CLASSIFIER_SPLIT_TRAIN)
    dev_counts = _count_by_category(CLASSIFIER_SPLIT_DEV)
    test_counts = _count_by_category(CLASSIFIER_SPLIT_TEST)
    cats = sorted(set(train_counts.keys()) | set(dev_counts.keys()) | set(test_counts.keys()))
    rows = []
    for c in cats:
        rows.append(
            {
                "category": c,
                "train_count": int(train_counts.get(c, 0)),
                "dev_count": int(dev_counts.get(c, 0)),
                "test_count": int(test_counts.get(c, 0)),
            }
        )
    return {"rows": rows}


@app.get("/api/train/dataset_rows")
def train_dataset_rows(category: str, split: str = "train", page: int = 1, page_size: int = 20):
    try:
        path = _split_path_by_name(split)
    except ValueError as e:
        return {"available": False, "reason": str(e), "rows": [], "total": 0, "page": 1, "page_size": page_size}

    rows = _read_jsonl(path)
    filtered = []
    for r in rows:
        if str(r.get("category", "")) != category:
            continue
        filtered.append(
            {
                "id": str(r.get("id", "")),
                "question": str(r.get("question", "")),
                "gold": str(r.get("gold", "")),
                "category": str(r.get("category", "")),
            }
        )

    paged = _paginate_rows(filtered, page, page_size)
    return {
        "available": True,
        "category": category,
        "split": split,
        **paged,
    }


@app.get("/api/train/eval_rows/{job_id}")
def train_eval_rows(
    job_id: str,
    page: int = 1,
    page_size: int = 20,
    correct_filter: str = "all",
    category_filter: str = "all",
):
    job = train_jobs.get(job_id)
    if not job:
        return {"available": False, "reason": "job_id_not_found", "rows": [], "total": 0, "page": 1, "page_size": page_size}

    inferred_eval = _infer_eval_predictions_path(
        job.get("eval_predictions_path"),
        job.get("eval_report_path"),
        job.get("model_path"),
    )
    eval_path: Optional[Path] = Path(inferred_eval) if inferred_eval else None
    if inferred_eval and (job.get("eval_predictions_path") != inferred_eval):
        job["eval_predictions_path"] = inferred_eval

    if eval_path is None:
        return {"available": False, "reason": "no_eval_predictions_path", "rows": [], "total": 0, "page": 1, "page_size": page_size}

    if not eval_path.exists():
        return {"available": False, "reason": "eval_predictions_missing", "rows": [], "total": 0, "page": 1, "page_size": page_size}

    rows: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            correct = str(r.get("correct", "")).strip().lower() == "true"
            true_cat = str(r.get("true_category", ""))

            if correct_filter == "true" and not correct:
                continue
            if correct_filter == "false" and correct:
                continue
            if category_filter != "all" and true_cat != category_filter:
                continue

            rows.append(
                {
                    "id": str(r.get("id", "")),
                    "true_category": true_cat,
                    "pred_category": str(r.get("pred_category", "")),
                    "correct": correct,
                }
            )

    total = len(rows)
    correct_n = sum(1 for r in rows if r["correct"])
    wrong_n = total - correct_n
    stats = {
        "correct": correct_n,
        "wrong": wrong_n,
        "accuracy": round((correct_n / total), 4) if total else 0.0,
    }
    paged = _paginate_rows(rows, page, page_size)
    return {"available": True, "job_id": job_id, "stats": stats, **paged}


@app.get("/api/train/latest")
def train_latest():
    if not train_jobs:
        return {"state": "empty", "message": "No training jobs in current session."}

    latest = sorted(
        train_jobs.values(), key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True
    )[0]
    return latest


# -----------------------------
# API: test suite jobs
# -----------------------------
@app.post("/api/test/run_suite")
async def test_run_suite(req: TestRunRequest):
    run_id = f"suite-{uuid.uuid4().hex[:8]}"
    run_dir = TEST_RUNTIME_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = req.model_dump()
    payload["categories"] = payload.get("categories") or DEFAULT_TEMPORAL_CATEGORIES
    payload["threshold"] = (
        ROUTER_THRESHOLD_DEFAULT if payload.get("threshold") is None else float(payload.get("threshold"))
    )

    test_runs[run_id] = {
        "run_id": run_id,
        "suite_name": payload.get("suite_name", "temporal_core_suite"),
        "state": "queued",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "payload": payload,
        "rows": [],
        "summary": {},
        "duration_sec": None,
        "error": None,
    }

    _write_json(run_dir / "request.json", payload)
    asyncio.create_task(_run_test_suite(run_id, payload))
    return test_runs[run_id]


@app.get("/api/test/status/{run_id}")
def test_status(run_id: str):
    run = test_runs.get(run_id)
    if not run:
        return {"run_id": run_id, "state": "not_found", "error": "run_id not found"}
    return {
        "run_id": run["run_id"],
        "suite_name": run.get("suite_name"),
        "state": run.get("state"),
        "progress": run.get("progress"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "duration_sec": run.get("duration_sec"),
        "error": run.get("error"),
    }


@app.get("/api/test/result/{run_id}")
def test_result(run_id: str):
    run = test_runs.get(run_id)
    if not run:
        persisted = TEST_RUNTIME_ROOT / run_id / "result.json"
        if persisted.exists():
            return _read_json(persisted, {})
        return {"run_id": run_id, "state": "not_found", "error": "run_id not found"}
    return run


@app.get("/api/analysis/summary")
def analysis_summary():
    try:
        p = _load_analysis_payload()
    except Exception as e:
        return {"available": False, "error": str(e)}
    return {
        "available": True,
        "source": p.get("source", {}),
        "ruleset_version": p.get("ruleset_version"),
        "generated_at": p.get("generated_at"),
        "workflows": p.get("workflow_summary", []),
        "oracle": p.get("oracle", {}),
        "correction_breakdown": p.get("correction_breakdown", {}),
    }


@app.get("/api/analysis/categorywise")
def analysis_categorywise():
    try:
        p = _load_analysis_payload()
    except Exception as e:
        return {"available": False, "error": str(e)}
    return {
        "available": True,
        "source": p.get("source", {}),
        "ruleset_version": p.get("ruleset_version"),
        "generated_at": p.get("generated_at"),
        "rows": p.get("category_summary", []),
    }


@app.get("/api/analysis/errors")
def analysis_errors(
    page: int = 1,
    page_size: int = 50,
    category: str = "all",
    workflow: str = "all",
    corrected: str = "all",
):
    try:
        p = _load_analysis_payload()
    except Exception as e:
        return {"available": False, "error": str(e), "rows": [], "total": 0, "page": 1, "page_size": page_size}

    rows = p.get("error_rows", [])
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if category != "all" and str(r.get("category", "")) != category:
            continue
        if workflow != "all" and str(r.get("workflow", "")) != workflow:
            continue
        corr = bool(r.get("corrected_match"))
        if corrected == "true" and not corr:
            continue
        if corrected == "false" and corr:
            continue
        filtered.append(r)

    paged = _paginate_rows(filtered, page, page_size)
    return {
        "available": True,
        "source": p.get("source", {}),
        "ruleset_version": p.get("ruleset_version"),
        "generated_at": p.get("generated_at"),
        **paged,
    }


@app.get("/")
def read_root():
    return {
        "status": "Backend is running",
        "service": "temporal-reasoning-workbench",
        "active_classifier_model": str(_active_classifier_model_path),
    }


_ensure_runtime_dirs()
