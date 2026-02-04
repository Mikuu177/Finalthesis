from __future__ import annotations
import re
from typing import Optional, Dict, Any
from dateutil import parser as dateparser

# Basic punctuation mapping (Chinese/English)
PUNC_MAP = {
    '，': ',',
    '。': '.',
    '：': ':',
    '；': ';',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '、': ',',
}


def unify_punctuation(text: str) -> str:
    for k, v in PUNC_MAP.items():
        text = text.replace(k, v)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def try_parse_date(text: str) -> Optional[str]:
    """Try to parse a date/datetime and return ISO date string if plausible."""
    try:
        dt = dateparser.parse(text, fuzzy=True)
        if not dt:
            return None
        return dt.strftime('%Y-%m-%d') if dt.hour == 0 and dt.minute == 0 and dt.second == 0 else dt.isoformat()
    except Exception:
        return None


def normalize_answer(text: str) -> str:
    if text is None:
        return ''
    t = str(text).strip()
    t = unify_punctuation(t)
    t = normalize_whitespace(t)
    t = t.strip('`"\'')
    iso = try_parse_date(t)
    if iso:
        return iso
    t = re.sub(r",(?=\d{3}(\D|$))", "", t)
    return t


# -------------------------
# Arithmetic / MCQ / Time / Date helpers
# -------------------------

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")
_TIME_FULL_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_TIME_ANY_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")

# ISO date strict
_DATE_ISO_FULL_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
# date fragments (iso / slash / us)
_DATE_ANY_RE = re.compile(r"\b(\d{4}[-/](?:\d{1,2})[-/](?:\d{1,2})|(?:\d{1,2})/(?:\d{1,2})/(?:\d{4}))\b")

# timezone tokens (rough)
_TZ_TOKEN_RE = re.compile(r"\b(UTC|GMT|PST|PDT|EST|EDT|CST|CDT|MST|MDT|CET|CEST|IST|JST|KST|AEST|AEDT)\b", re.IGNORECASE)


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


def extract_time_fragment(text: str) -> Optional[str]:
    if not text:
        return None
    m = None
    for m in _TIME_ANY_RE.finditer(text):
        pass
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    return f"{hh:02d}:{mm:02d}"


def extract_date_fragment_to_iso(text: str) -> Optional[str]:
    """Extract last date-like fragment and normalize to YYYY-MM-DD using dateutil."""
    if not text:
        return None
    m = None
    for m in _DATE_ANY_RE.finditer(text):
        pass
    if not m:
        return None
    frag = m.group(1)
    # Normalize separators
    frag2 = frag.replace('/', '-')
    try:
        dt = dateparser.parse(frag2, fuzzy=True)
        if not dt:
            return None
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None


def normalize_model_answer(pred: str, sample: Dict[str, Any]) -> str:
    """MCQ-aware normalization: label->option; then time extraction; then date extraction; then numeric/text normalize."""
    pred = '' if pred is None else str(pred).strip()

    md = (sample or {}).get('metadata') or {}
    opts = md.get('mcq_options')

    # 1) label -> option
    if opts and pred.upper() in ('A', 'B', 'C', 'D'):
        opt_text = opts.get(pred.upper())
        if opt_text is not None:
            pred = str(opt_text)

    # 2) time fragment
    tf = extract_time_fragment(pred)
    if tf:
        return tf

    # 3) date fragment
    df = extract_date_fragment_to_iso(pred)
    if df:
        return df

    # 4) fallback
    return normalize_arithmetic_answer(pred)


def is_label_answer(pred: str) -> bool:
    pred = '' if pred is None else str(pred).strip()
    return pred.upper() in ('A', 'B', 'C', 'D')


def is_numeric_answer(pred: str) -> bool:
    pred = '' if pred is None else str(pred).strip()
    pred = ''.join(pred.split()).replace(',', '')
    return bool(_NUM_RE.match(pred))


def is_time_answer(pred_norm: str) -> bool:
    return bool(_TIME_FULL_RE.match(pred_norm or ''))


def contains_time_fragment(raw_text: str) -> bool:
    return bool(_TIME_ANY_RE.search(raw_text or ''))


def is_date_answer(pred_norm: str) -> bool:
    return bool(_DATE_ISO_FULL_RE.match(pred_norm or ''))


def contains_date_fragment(raw_text: str) -> bool:
    # cheap check for iso-like or slashes
    return bool(_DATE_ANY_RE.search(raw_text or ''))


def contains_tz_token(raw_text: str) -> bool:
    return bool(_TZ_TOKEN_RE.search(raw_text or ''))
