import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .io_utils import read_yaml, read_jsonl


def _format_exemplar(ex: Dict[str, Any], mode: str) -> str:
    q = ex.get('question', '')
    ctx = ex.get('context', '')
    gold = ex.get('gold') or ex.get('answer') or ''
    rationale = ex.get('rationale', '')
    parts: List[str] = []
    parts.append(f"Question: {q}")
    if ctx:
        parts.append(f"Context: {ctx}")
    if mode == 'cot' and rationale:
        parts.append(f"Reasoning: {rationale}")
    parts.append(f"FINAL_ANSWER: {gold}")
    return '\n'.join(parts)


def category_rules(category: str) -> str:
    cat = (category or 'unspecified').lower()
    if 'hour adjustment' in cat and '24h' in cat:
        return (
            "CATEGORY_RULES:\n"
            "- Interpret A - B as time-of-day subtraction under 24-hour wrap-around (mod 24 hours).\n"
            "- Output ONLY HH:MM (24-hour).\n"
        )
    if 'hour adjustment' in cat and '12h' in cat:
        return (
            "CATEGORY_RULES:\n"
            "- Interpret time arithmetic under 12-hour clock if the question implies it; convert to 24-hour HH:MM for output.\n"
            "- Output ONLY HH:MM (24-hour).\n"
        )
    if 'date computation' in cat or 'year shift' in cat or 'month shift' in cat:
        return (
            "CATEGORY_RULES:\n"
            "- Use Gregorian calendar arithmetic.\n"
            "- Output ONLY the final date in ISO format YYYY-MM-DD.\n"
        )
    if 'time zone conversion' in cat:
        return (
            "CATEGORY_RULES:\n"
            "- Convert the given time to the target local time zone (apply the correct direction/sign).\n"
            "- Output ONLY the converted local time in HH:MM (24-hour).\n"
        )
    return ""


def get_hint_policy_signature(version: str = 'v1') -> Dict[str, str]:
    texts = [
        category_rules('Hour Adjustment (24h)'),
        category_rules('Hour Adjustment (12h)'),
        category_rules('Date Computation'),
        category_rules('Year Shift'),
        category_rules('Month Shift'),
        category_rules('Time Zone Conversion'),
        category_rules('unspecified'),
    ]
    blob = '\n---\n'.join(texts)
    h = hashlib.sha256(blob.encode('utf-8')).hexdigest()
    return {
        'hint_policy_version': version,
        'hint_policy_hash': h,
    }


def build_prompt(sample: Dict[str, Any], prompts_cfg_path: str | Path, mode: str, n_shots: int, prompt_dir: str | Path, seed: int) -> Tuple[str, str, List[str]]:
    prompts_cfg = read_yaml(prompts_cfg_path)
    template_map = prompts_cfg['prompt_templates'][mode]

    # choose template key (as strings)
    n_key = str(int(n_shots))
    if mode == 'cot' and int(n_shots) == 0:
        pv = str(sample.get('_prompt_version', ''))
        if pv == 'v2.2' and '0_v22' in template_map:
            n_key = '0_v22'

    template_file = template_map[n_key]
    template_path = Path(prompt_dir, template_file)
    template_text = template_path.read_text(encoding='utf-8')

    exemplar_ids: List[str] = []
    exemplars_text = ''
    if n_shots > 0:
        ex_file = Path(prompt_dir, prompts_cfg['exemplars_path'])
        exemplars = read_jsonl(ex_file)
        rnd = random.Random(seed)
        picks = exemplars[:]
        rnd.shuffle(picks)
        picks = picks[:n_shots]
        exemplar_ids = [str(ex.get('id')) for ex in picks]
        ex_formatted = [_format_exemplar(ex, mode) for ex in picks]
        exemplars_text = '\n\n'.join(ex_formatted)

    q = sample.get('question', '')
    ctx = sample.get('context', '')

    prompt = template_text
    if '{EXEMPLARS}' in prompt:
        prompt = prompt.replace('{EXEMPLARS}', exemplars_text)

    rules = category_rules(sample.get('category', ''))

    parts: List[str] = [prompt]
    if rules:
        parts.append(rules)
    parts.append('== Task ==')
    parts.append(f"Question: {q}")
    if ctx:
        parts.append(f"Context: {ctx}")
    parts.append('Answer strictly with the final line format: FINAL_ANSWER: <your answer>.')
    full_prompt = '\n'.join(parts)

    prompt_template = f"{mode}_{int(n_shots)}shot"
    return full_prompt, prompt_template, exemplar_ids
