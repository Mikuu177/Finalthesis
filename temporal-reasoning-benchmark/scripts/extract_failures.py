import json, sys, re
from pathlib import Path

def tail_lines(text: str, n=10):
    lines = text.splitlines()
    return "\n".join(lines[-n:])

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def looks_iso_date(s: str) -> bool:
    return bool(ISO_DATE.match(s.strip()))

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_failures.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    pred_path = run_dir / 'predictions.jsonl'
    if not pred_path.exists():
        print(f"Not found: {pred_path}")
        sys.exit(2)

    parse_fail = []
    wrong = []
    format_err = []

    with pred_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = bool(row.get('parsed'))
            correct = bool(row.get('correct'))
            cat = row.get('category', 'unspecified')
            pred = str(row.get('pred',''))
            gold = str(row.get('gold',''))

            if not parsed:
                if len(parse_fail) < 2:
                    parse_fail.append(row)
                continue

            if not correct and len(wrong) < 2:
                wrong.append(row)

            # heuristic format anomaly: gold is ISO date but pred not ISO
            if len(format_err) < 2:
                if looks_iso_date(gold) and parsed and not looks_iso_date(pred):
                    format_err.append(row)
                elif cat in ('duration',) and parsed and ('小时' not in pred and '分钟' not in pred and 'hour' not in pred and 'min' not in pred):
                    format_err.append(row)

    def show(title, rows):
        print(f"\n== {title} ==")
        if not rows:
            print("(none)")
            return
        for r in rows:
            raw_tail = tail_lines(r.get('raw',''), 10)
            print(json.dumps({
                'id': r.get('id'),
                'category': r.get('category'),
                'question': r.get('prompt','').split('Question:')[-1].strip()[:200],
                'gold': r.get('gold'),
                'pred': r.get('pred'),
                'parsed': r.get('parsed'),
                'raw_tail': raw_tail,
            }, ensure_ascii=False, indent=2))

    show('Parse Fail (up to 2)', parse_fail)
    show('Incorrect Prediction (up to 2)', wrong)
    show('Format Anomaly (up to 2)', format_err)

if __name__ == '__main__':
    main()






