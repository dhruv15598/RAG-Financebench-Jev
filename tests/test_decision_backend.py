"""Lifecycle tests without loading models, touching GPUs, or gateway calls."""
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from fastapi.testclient import TestClient
import dashboard
import decision_backend as backend
from jev import EVIDENCE_CHECK, ANSWER_CHECK


def valid(criteria):
    return {'answers': {key: {'type': 'choice', 'choice': next(iter(q['criteria'])),
        'probabilities': {label: 1.0 if i == 0 else 0.0 for i, label in enumerate(q['criteria'])}}
        for key, q in criteria.items()}, 'client_elapsed_seconds': .01}


def test_registry_allowlist_and_argv(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'models': [{'id': 'decider-4b-v2-q4', 'command': ['python', 'local_deciders/q4_server.py']}]}))
    models = backend.load_models(path)
    assert len(models) == 3
    assert backend.public_models(models)[2]['available']
    path.write_text(json.dumps([{'id': 'arbitrary', 'command': ['rm']}]))
    with pytest.raises(backend.DecisionError):
        backend.load_models(path)
    path.write_text(json.dumps([{'id': 'decider-4b-v2-q4', 'command': 'python local_deciders/q4_server.py'}]))
    with pytest.raises(backend.DecisionError):
        backend.load_models(path)


def test_structured_probability_validation():
    backend.validate_result(valid(EVIDENCE_CHECK), EVIDENCE_CHECK)
    result = valid(EVIDENCE_CHECK)
    result['answers']['evidence_support']['probabilities']['sufficient'] = float('nan')
    with pytest.raises(backend.DecisionError):
        backend.validate_result(result, EVIDENCE_CHECK)


def test_memory_refuses_without_killing_apps(monkeypatch):
    monkeypatch.setattr(backend, 'gpu_memory', lambda: (1000, 12000))
    with pytest.raises(backend.DecisionError, match='1000 MiB'):
        asyncio.run(backend.memory_preflight(6000))


def test_cpu_configuration_and_invalid_device(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps([{'id': 'decider-2b-coreai', 'device': 'cpu'}]))
    model = next(m for m in backend.load_models(path) if m.id == 'decider-2b-coreai')
    public = backend.public_models([model])[0]
    assert public['device'] == 'cpu'
    assert public['min_free_mib'] == 0
    path.write_text(json.dumps([{'id': 'decider-2b-coreai', 'device': 'unknown'}]))
    with pytest.raises(backend.DecisionError):
        backend.load_models(path)


def test_cpu_checker_skips_gpu_guard(monkeypatch):
    async def fail_guard(minimum):
        raise AssertionError('CPU checker must not require GPU memory')
    monkeypatch.setattr(backend, 'memory_preflight', fail_guard)
    with backend.socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    checker = backend.LocalChecker(replace(backend.DEFAULT_MODELS[1], device='cpu', command=('runtime',), port=port))
    def stop_at_launch(*args):
        raise RuntimeError('reached process launch')
    monkeypatch.setattr(checker.owner, 'start', stop_at_launch)
    with pytest.raises(RuntimeError, match='reached process launch'):
        asyncio.run(checker.start())


def test_occupied_port_never_starts_process():
    with backend.socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        checker = backend.LocalChecker(replace(backend.DEFAULT_MODELS[1], command=('never',), port=occupied.getsockname()[1]))
        with pytest.raises(backend.DecisionError, match='occupied'):
            asyncio.run(checker.start())
        assert checker.owner.process is None


@pytest.mark.parametrize('checker_id', ['decider-2b-coreai', 'decider-4b-v2-q4'])
@pytest.mark.parametrize('fail_phase', [None, 'pre-init', 'pre', 'generation', 'post-init', 'post'])
def test_local_checker_exclusive_phases_and_failure_cleanup(monkeypatch, tmp_path, fail_phase, checker_id):
    order = []
    model = replace(next(m for m in backend.DEFAULT_MODELS if m.id == checker_id), command=('runtime',))
    async def config(): return {'models': ['qwen-test'], 'jev_ready': False}
    async def unload(url): order.append('unload')
    async def preflight(minimum): order.append('memory')
    class Checker:
        def __init__(self, selected): self.starts = 0
        async def start(self):
            self.starts += 1
            order.append('start')
            if fail_phase == ('pre-init' if self.starts == 1 else 'post-init'):
                raise backend.DecisionError('Safe initialization failure.')
        async def stop(self): order.append('stop')
        async def evaluate(self, state, criteria):
            phase = 'post' if 'generated_answer' in state else 'pre'
            order.append(phase)
            assert 'reference_answer' not in state
            if fail_phase == phase: raise backend.DecisionError('Safe checker failure.')
            return valid(criteria)
    class Response:
        def raise_for_status(self): pass
        async def aiter_lines(self):
            order.append('generate')
            if fail_phase == 'generation': raise RuntimeError('private upstream body')
            yield json.dumps({'message': {'content': 'Fresh answer'}, 'done': False})
            yield json.dumps({'done': True, 'done_reason': 'stop'})
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    class Client:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, method, url, json):
            assert json['model'] == 'qwen-test'
            assert json['keep_alive'] == 0
            assert url.endswith('/api/chat')
            return Response()
    monkeypatch.setattr(dashboard, 'config', config)
    monkeypatch.setattr(dashboard, 'load_models', lambda: (model,))
    monkeypatch.setattr(dashboard, 'LocalChecker', Checker)
    monkeypatch.setattr(dashboard, 'unload_ollama', unload)
    monkeypatch.setattr(dashboard, 'memory_preflight', preflight)
    monkeypatch.setattr(dashboard.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(dashboard, 'busy', asyncio.Lock())
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    client = TestClient(dashboard.app, base_url='http://localhost')
    payload = {'question_id': dashboard.ROWS[0]['id'], 'model': 'qwen-test', 'decision_model': model.id}
    response = client.post('/api/run', json=payload)
    assert response.status_code == 200
    assert not dashboard.busy.locked()
    assert order[-2:] == ['stop', 'unload']
    assert order[:2] == ['unload', 'start']
    if fail_phase != 'pre-init':
        assert order[2] == 'pre'
    if fail_phase is None:
        assert order == ['unload', 'start', 'pre', 'stop', 'memory', 'generate', 'unload', 'start', 'post', 'stop', 'unload']
        assert '"type": "done"' in response.text
        assert 'initialization (answer check)' in response.text
    else:
        assert '"type": "error"' in response.text
        assert '"type": "done"' not in response.text
        assert 'private upstream body' not in response.text


def test_retired_config_migrates_without_losing_current_models(tmp_path):
    path = tmp_path / 'config.json'
    entries = [{'id': model_id, 'command': 'legacy shell string'} for model_id in sorted(backend.RETIRED_MODEL_IDS)]
    entries += [{'id': 'decider-2b-coreai', 'command': ['python', 'adapter.py']},
                {'id': 'decider-4b-v2-q4', 'command': ['python', 'q4.py']}]
    path.write_text(json.dumps({'models': entries}))
    with pytest.warns(UserWarning, match='Retired decision model') as warnings:
        models = backend.load_models(path)
    assert len(warnings) == 3
    assert {m.id for m in models} == {'jev', 'decider-2b-coreai', 'decider-4b-v2-q4'}
    assert all(m['available'] for m in backend.public_models(models))
    path.write_text(json.dumps(entries + [{'id': 'unknown-model'}]))
    with pytest.warns(UserWarning), pytest.raises(backend.DecisionError):
        backend.load_models(path)
