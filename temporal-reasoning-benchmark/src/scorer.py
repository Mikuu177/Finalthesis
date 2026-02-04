from __future__ import annotations
import re
from typing import Tuple, Optional

from .normalize import normalize_answer


FINAL_PAT = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    if not text:
        return ''
    # 1) 优先：最后一行是否满足契约
    lines_all = [ln.rstrip() for ln in text.splitlines()]
    lines_nonempty = [ln.strip() for ln in lines_all if ln.strip()]
    if lines_nonempty:
        last_line = lines_nonempty[-1]
        m_last = re.match(r"^\s*FINAL_ANSWER\s*:\s*(.+)\s*$", last_line, flags=re.IGNORECASE)
        if m_last:
            return m_last.group(1).strip()
    # 2) 退路：全文最后一次出现 FINAL_ANSWER:
    m = None
    for m in FINAL_PAT.finditer(text):
        pass
    if m:
        return m.group(1).strip()
    # 3) 最后退路：取最后一行非空文本
    return lines_nonempty[-1] if lines_nonempty else ''


def score(pred_raw: str, gold_raw: str, relaxed: bool = False) -> Tuple[bool, dict]:
    pred_norm = normalize_answer(pred_raw)
    gold_norm = normalize_answer(gold_raw)
    correct = pred_norm == gold_norm
    info = {
        'pred_norm': pred_norm,
        'gold_norm': gold_norm,
        'match': 'strict' if correct else 'none',
    }
    if (not correct) and relaxed:
        # relaxed numeric compare if both look like numbers
        try:
            p = float(pred_norm)
            g = float(gold_norm)
            if abs(p - g) < 1e-6:
                correct = True
                info['match'] = 'relaxed_numeric'
        except Exception:
            pass
    return correct, info

# -----------------------------
# V1 scoring enhancements
# -----------------------------

_DUR_H_RE = re.compile(r"(\d+)\s*(?:小时|时|h)\b", re.IGNORECASE)
_DUR_M_RE = re.compile(r"(\d+)\s*(?:分钟|分|min|m)\b", re.IGNORECASE)
_DUR_D_RE = re.compile(r"(\d+)\s*(?:天|日|day|days|d)\b", re.IGNORECASE)
_DUR_ONLY_MIN_RE = re.compile(r"^(\d+)\s*(?:分钟|分|min|m)\b", re.IGNORECASE)
_DUR_ONLY_H_RE = re.compile(r"^(\d+)\s*(?:小时|时|h)\b", re.IGNORECASE)


def parse_duration_minutes(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip()
    try:
        # patterns like "1 小时 45 分钟", "1h45m"
        hours = 0
        minutes = 0
        mh = _DUR_H_RE.findall(s)
        mm = _DUR_M_RE.findall(s)
        md = _DUR_D_RE.findall(s)
        if md:
            # days present -> convert to minutes
            minutes += sum(int(x) for x in md) * 24 * 60
        if mh:
            hours += sum(int(x) for x in mh)
        if mm:
            minutes += sum(int(x) for x in mm)
        if md or mh or mm:
            return hours * 60 + minutes
        # pure hour or pure minute
        m_only = _DUR_ONLY_MIN_RE.search(s)
        if m_only:
            return int(m_only.group(1))
        h_only = _DUR_ONLY_H_RE.search(s)
        if h_only:
            return int(h_only.group(1)) * 60
    except Exception:
        return None
    return None

# --- relation normalization ---
_REL_SPACES = re.compile(r"\s+")

_REL_SYNONYMS = [
    (re.compile(r"(更晚|晚于|之后|后来|在后|在之后)"), "在后"),
    (re.compile(r"(更早|早于|之前|在前|在之前)"), "在前"),
    (re.compile(r"(相同|一样|同时|same|equal)" , re.IGNORECASE), "同时"),
    (re.compile(r"(相邻且不重叠|相邻不重叠|相邻|adjacent.*no\s*overlap|no\s*overlap.*adjacent)" , re.IGNORECASE), "相邻且不重叠"),
]


def _clean_rel(s: str) -> str:
    s = s or ''
    s = s.strip()
    # lowercase english only
    s = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), s)
    # unify synonyms
    for pat, rep in _REL_SYNONYMS:
        s = pat.sub(rep, s)
    # remove spaces
    s = _REL_SPACES.sub("", s)
    # unify punctuation
    s = s.replace("，", ",").replace("。", ".")
    return s


def relation_canonical(s: str) -> str:
    s = _clean_rel(s)
    # normalize passive contain: X被A包含 -> A包含X
    m = re.search(r"([ABab])被([ABab])包含", s)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        return f"{y}包含{x}"
    return s


def relation_match_v1(pred_raw: str, gold_raw: str) -> Tuple[bool, str]:
    p = relation_canonical(pred_raw)
    g = relation_canonical(gold_raw)
    if p == g:
        return True, 'v1_relation_canonical'
    # entity-only acceptance based on original gold phrasing
    if any(k in gold_raw for k in ['在后','在前','更晚','更早','之前','之后']):
        ent = 'A' if 'A' in gold_raw else ('B' if 'B' in gold_raw else None)
        if ent and pred_raw.strip() in ('A','B') and pred_raw.strip() == ent:
            return True, 'v1_relation_entity_only_ok'
    # equality synonyms broader acceptance
    if ('同时' in gold_raw) and (('相同' in pred_raw) or ('同时' in pred_raw) or ('same' in pred_raw.lower())):
        return True, 'v1_relation_equal_syn'
    # adjacent/no-overlap english hints
    if ('相邻' in g) and ('adjacent' in pred_raw.lower() or 'no overlap' in pred_raw.lower()):
        return True, 'v1_relation_adjacent_eno'
    return False, 'none'


_DUR_DAYS_RE = re.compile(r"(\d+)\s*(?:天|日|day|days|d)\b", re.IGNORECASE)
_ONLY_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def _norm_duration_str(s: str) -> str:
    s = s or ''
    s = re.sub(r"\s+", "", s)
    # unify chinese units spacing removed already
    return s


def score_v1(pred_raw: str, gold_raw: str, category: str) -> Tuple[bool, dict]:
    category = (category or '').lower()
    # duration: numeric compare (minutes first; fallback to days or string-unified compare)
    if category == 'duration':
        pm = parse_duration_minutes(pred_raw)
        gm = parse_duration_minutes(gold_raw)
        if pm is not None and gm is not None:
            correct = (pm == gm)
            return correct, {
                'match': 'v1_duration_minutes' if correct else 'none',
                'pred_minutes': pm,
                'gold_minutes': gm,
            }
        # try day-based comparison if gold expresses days
        gd_m = _DUR_DAYS_RE.search(gold_raw or '')
        if gd_m:
            gd = int(gd_m.group(1))
            pd_m = _DUR_DAYS_RE.search(pred_raw or '')
            if pd_m:
                pd = int(pd_m.group(1))
            else:
                # bare integer like '6' interpreted as days when gold is days
                m = _ONLY_INT_RE.match(pred_raw or '')
                pd = int(m.group(1)) if m else None
            if pd is not None:
                correct = (pd == gd)
                return correct, {'match': 'v1_duration_days' if correct else 'none', 'pred_days': pd, 'gold_days': gd}
        # string-level normalization fallback (remove spaces/variants)
        if _norm_duration_str(pred_raw) == _norm_duration_str(gold_raw):
            return True, {'match': 'v1_duration_string'}
        # fallback to strict normalize
        c, info = score(pred_raw, gold_raw, relaxed=False)
        return c, info
    # relation: canonical mapping + entity-only acceptance
    if category == 'relation':
        ok, mode = relation_match_v1(pred_raw, gold_raw)
        if ok:
            return True, {'match': mode}
        # fallback to strict normalize
        c, info = score(pred_raw, gold_raw, relaxed=False)
        return c, info
    # default: original scoring
    return score(pred_raw, gold_raw, relaxed=False)
