"""Local live demo: saved retrieval, live Ollama generation and Jev checks."""
import argparse
import asyncio
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
from decision_backend import load_models, public_models, LocalChecker, DecisionError, unload_ollama, memory_preflight

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
    model: str = ''
    decision_model: str = 'jev'


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
    try:
        decision_models = public_models(load_models())
    except RuntimeError:
        decision_models = []
    return {'questions': [{k: row[k] for k in ('id', 'company', 'question', 'reference_answer')} for row in ROWS],
            'models': models, 'decision_models': decision_models,
            'jev_ready': bool(os.environ.get('AI_GATEWAY_API_KEY'))}


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
    if body.model and body.model not in cfg['models']:
        raise HTTPException(400, 'Select an installed Qwen model.')
    try:
        selected_decision = next((m for m in load_models() if m.id == body.decision_model and m.enabled), None)
    except DecisionError as error:
        raise HTTPException(503, str(error)) from None
    if selected_decision is None:
        raise HTTPException(400, 'Select an enabled decision model.')
    if not body.model:
        raise HTTPException(400, 'Select an installed Qwen model.')
    if selected_decision.kind == 'local' and not selected_decision.command:
        raise HTTPException(503, 'Configure a launch command for this decision model.')
    if selected_decision.kind == 'jev' and not cfg['jev_ready']:
        raise HTTPException(503, 'Restart with AI_GATEWAY_API_KEY set.')
    if busy.locked():
        raise HTTPException(409, 'Another demonstration is running.')
    await busy.acquire()

    async def events():
        def event(kind, **data):
            return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'

        state = {'question': row['question'], 'evidence': row['evidence']}
        checker = LocalChecker(selected_decision) if selected_decision.kind == 'local' else None
        stage = 'evidence check'
        async def initialize(phase):
            yield event('stage', label=f'Releasing Qwen memory before {phase}')
            started = time.perf_counter()
            await unload_ollama(OLLAMA)
            yield event('timing', label='GPU release', seconds=round(time.perf_counter() - started, 3))
            yield event('stage', label=f'Initializing {selected_decision.label} for {phase}')
            started = time.perf_counter()
            await checker.start()
            yield event('timing', label=f'{selected_decision.label} initialization ({phase})', seconds=round(time.perf_counter() - started, 3))
        async def check(current_state, criteria, kind):
            if checker:
                result = await checker.evaluate(current_state, criteria)
                yield event(kind, result=result)
            else:
                async for item in checked_events(current_state, criteria, kind):
                    yield json.dumps(item) + '\n'
        try:
            yield event('evidence', passages=row['evidence'])
            if checker:
                async for item in initialize('evidence check'):
                    yield item
            yield event('stage', label=f'{selected_decision.label} is checking the evidence')
            async for item in check(state, EVIDENCE_CHECK, 'precheck'):
                yield item
            stage = 'Qwen generation'
            if checker:
                yield event('stage', label='Stopping the local checker before Qwen generation')
                await checker.stop()
                await memory_preflight(int(os.environ.get('QWEN_MIN_FREE_MIB', '6000')))
            yield event('stage', label='Qwen is generating an answer')
            answer, completed = '', False
            started = time.perf_counter()
            payload = {'model': body.model, 'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(state)}],
                'think': False, 'stream': True,
                'options': {'num_ctx': 8192, 'num_predict': 384, 'temperature': 0}}
            generation_url = OLLAMA + '/api/chat'
            if checker:
                payload['keep_alive'] = 0
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream('POST', generation_url, json=payload) as response:
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
                        if not token and chunk.get('choices'):
                            token = chunk['choices'][0].get('delta', {}).get('content', '')
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
            if checker:
                async for item in initialize('answer check'):
                    yield item
            yield event('stage', label=f'{selected_decision.label} is checking this new answer')
            async for item in check({**state, 'generated_answer': answer}, ANSWER_CHECK, 'postcheck'):
                yield item
            yield event('reference', text=row.get('reference_answer', 'Not available'))
            yield event('done')
        except Exception as error:
            # Never forward upstream payloads, headers or exception strings to the UI.
            print(f'Demo failed during {stage}: {type(error).__name__}', flush=True)
            detail = str(error) if isinstance(error, (JevError, DecisionError)) else 'Local service or response processing failed.'
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
            async def cleanup():
                try:
                    await checker.stop()
                finally:
                    await unload_ollama(OLLAMA)
            try:
                if checker:
                    task = asyncio.create_task(cleanup())
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        # Keep the run lock until owned process cleanup finishes.
                        await task
                        raise
                    except Exception as error:
                        print(f'Demo cleanup failed: {type(error).__name__}', flush=True)
            finally:
                busy.release()

    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)
