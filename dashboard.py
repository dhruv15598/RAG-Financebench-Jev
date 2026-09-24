"""Local live demo: saved retrieval, live Ollama generation and Jev checks."""
import argparse
import asyncio
import getpass
import json
import os
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from jev import evaluate_jev, JevError, EVIDENCE_CHECK, ANSWER_CHECK
from run import SYSTEM

ROOT = Path(__file__).resolve().parent
ROWS = json.loads((ROOT / 'data/snapshot.json').read_text(encoding='utf-8-sig'))['rows']
PAGE_EVIDENCE = {r['id']: r['evidence'] for r in json.loads((ROOT / 'data/dashboard-evidence.json').read_text(encoding='utf-8'))['rows']}
ROWS = [{**row, 'evidence': PAGE_EVIDENCE[row['id']]} for row in ROWS]
OLLAMA = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11435').rstrip('/')
app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1'])
busy = asyncio.Lock()


async def checked_events(state, criteria, kind):
    """One visible retry for transient gateway failures; no retries for bad inputs."""
    for attempt in range(2):
        try:
            result = await asyncio.to_thread(evaluate_jev, state, criteria)
            yield {'type': kind, 'result': result}
            return
        except JevError as error:
            if attempt or not error.retryable:
                raise
            yield {'type': 'stage', 'label': f'{error} Retrying Jev once in 2 seconds…'}
            await asyncio.sleep(2)


class RunRequest(BaseModel):
    question_id: str
    model: str


@app.get('/')
def home():
    return FileResponse(ROOT / 'dashboard.html')


@app.get('/api/config')
async def config():
    models = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            result = await client.get(OLLAMA + '/api/tags')
            result.raise_for_status()
            models = [m['name'] for m in result.json()['models'] if 'qwen' in m['name'].lower()]
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return {'questions': [{k: row[k] for k in ('id', 'company', 'question', 'reference_answer')} for row in ROWS],
            'models': models, 'jev_ready': bool(os.environ.get('AI_GATEWAY_API_KEY'))}


@app.post('/api/run')
async def run_demo(body: RunRequest, request: Request):
    # Reject cross-site browser requests; credentials remain server-side.
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Cross-site request rejected.')
    row = next((r for r in ROWS if r['id'] == body.question_id), None)
    if row is None:
        raise HTTPException(400, 'Unknown question.')
    cfg = await config()
    if body.model not in cfg['models']:
        raise HTTPException(400, 'Select an installed Qwen model.')
    if not cfg['jev_ready']:
        raise HTTPException(503, 'Restart with AI_GATEWAY_API_KEY set.')
    if busy.locked():
        raise HTTPException(409, 'Another demonstration is running.')
    await busy.acquire()

    async def events():
        def event(kind, **data):
            return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'

        state = {'question': row['question'], 'evidence': row['evidence']}
        stage = 'evidence check'
        try:
            yield event('evidence', passages=row['evidence'])
            yield event('stage', label='Jev is checking the evidence')
            async for item in checked_events(state, EVIDENCE_CHECK, 'precheck'):
                yield json.dumps(item) + '\n'
            stage = 'Qwen generation'
            yield event('stage', label='Qwen is generating an answer')
            answer, completed = '', False
            started = time.perf_counter()
            payload = {'model': body.model, 'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(state)}],
                'think': False, 'stream': True,
                'options': {'num_ctx': 8192, 'num_predict': 384, 'temperature': 0}}
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream('POST', OLLAMA + '/api/chat', json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if await request.is_disconnected():
                            return
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get('error'):
                            raise RuntimeError('Model returned an error')
                        token = chunk.get('message', {}).get('content', '')
                        if token:
                            answer += token
                            yield event('token', text=token)
                        if chunk.get('done'):
                            completed = chunk.get('done_reason') == 'stop'
            elapsed = time.perf_counter() - started
            if not completed or not answer.strip():
                yield event('error', message='Generation was incomplete. No answer approval was requested.')
                return
            yield event('generation', seconds=round(elapsed, 3))
            stage = 'answer check'
            yield event('stage', label='Jev is checking this new answer')
            async for item in checked_events({**state, 'generated_answer': answer}, ANSWER_CHECK, 'postcheck'):
                yield json.dumps(item) + '\n'
            yield event('reference', text=row.get('reference_answer', 'Not available'))
            yield event('done')
        except Exception as error:
            # Never forward upstream payloads, headers or exception strings to the UI.
            print(f'Demo failed during {stage}: {type(error).__name__}', flush=True)
            detail = str(error) if isinstance(error, JevError) else 'Local service or response processing failed.'
            record = {'time': time.time(), 'stage': stage, 'question_id': row['id'],
                      'error_type': type(error).__name__, 'detail': detail}
            try:
                log = ROOT / 'outputs/dashboard-errors.jsonl'
                log.parent.mkdir(exist_ok=True)
                with log.open('a', encoding='utf-8') as file:
                    file.write(json.dumps(record) + '\n')
            except OSError:
                pass
            yield event('error', stage=stage, message=f'The {stage} failed. {detail}')
        finally:
            busy.release()

    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    if not os.environ.get('AI_GATEWAY_API_KEY'):
        os.environ['AI_GATEWAY_API_KEY'] = getpass.getpass('Vercel AI Gateway key (hidden): ').strip()
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)
