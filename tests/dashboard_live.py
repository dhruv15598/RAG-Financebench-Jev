"""Opt-in live dashboard checks. Runs models and makes paid calls when Jev is selected.

Use --decision-model for each checker to test; defaults to configured choices.
Writes actual streamed results, not an accuracy benchmark. Keep output local.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--model', default='qwen3.5:2b')
    parser.add_argument('--decision-model', action='append')
    parser.add_argument('--question-id')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + '/api/config', timeout=15) as response:
        config = json.load(response)
    if args.model not in config['models']:
        parser.error('Answer model is not installed')
    choices = {m['id']: m for m in config['decision_models'] if m['available']}
    selected = args.decision_model or list(choices)
    if any(m not in choices for m in selected):
        parser.error('Requested checker is not configured')
    question = args.question_id or next(q['id'] for q in config['questions'] if q['company'] == 'PepsiCo')
    if args.out.exists():
        parser.error('Output already exists; choose a fresh path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {'started_at': datetime.now(timezone.utc).isoformat(), 'runs': []}
    for checker in selected:
        payload = {'question_id': question, 'model': args.model, 'decision_model': checker}
        run = {**payload, 'events': [], 'passed': False}
        started = time.perf_counter()
        try:
            request = urllib.request.Request(args.url + '/api/run', data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                for line in response:
                    event = json.loads(line)
                    run['events'].append(event)
                    if event['type'] == 'stage':
                        print(checker + ': ' + event['label'], flush=True)
            kinds = [e['type'] for e in run['events']]
            run['passed'] = all(k in kinds for k in ('evidence', 'precheck', 'token', 'generation', 'postcheck', 'reference', 'done')) and 'error' not in kinds
        except Exception as error:
            run['transport_error_type'] = type(error).__name__
        run['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        results['runs'].append(run)
        args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f"{checker}: {'PASS' if run['passed'] else 'FAIL'} ({run['elapsed_seconds']} s)", flush=True)
    return 0 if all(r['passed'] for r in results['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
