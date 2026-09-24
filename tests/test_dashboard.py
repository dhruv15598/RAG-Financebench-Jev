"""Boundary checks without model downloads or paid calls."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import dashboard


def test_local_demo_boundaries(monkeypatch):
    async def config():
        return {'models': ['qwen-test'], 'jev_ready': True}
    monkeypatch.setattr(dashboard, 'config', config)
    monkeypatch.setattr(dashboard, 'busy', asyncio.Lock())
    client = TestClient(dashboard.app, base_url='http://localhost')
    payload = {'question_id': dashboard.ROWS[0]['id'], 'model': 'qwen-test'}
    assert client.post('/api/run', json=payload, headers={'Origin': 'https://example.com'}).status_code == 403
    assert client.post('/api/run', json={**payload, 'model': 'unknown'}).status_code == 400
    assert client.post('/api/run', json={**payload, 'question_id': 'unknown'}).status_code == 400
    assert client.get('/', headers={'Host': 'evil.example'}).status_code == 400


def test_gateway_failure_releases_run_and_hides_secrets(monkeypatch, tmp_path):
    async def config():
        return {'models': ['qwen-test'], 'jev_ready': True}
    def failure(*args, **kwargs):
        raise RuntimeError('private upstream payload')
    monkeypatch.setattr(dashboard, 'config', config)
    monkeypatch.setattr(dashboard, 'busy', asyncio.Lock())
    monkeypatch.setattr(dashboard, 'evaluate_jev', failure)
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    client = TestClient(dashboard.app, base_url='http://localhost')
    response = client.post('/api/run', json={'question_id': dashboard.ROWS[0]['id'], 'model': 'qwen-test'})
    assert '"type": "error"' in response.text
    assert 'private upstream payload' not in response.text
    assert 'reference_answer' not in response.text
    assert not dashboard.busy.locked()


def test_transient_retry_is_visible_and_bounded(monkeypatch):
    calls = []
    def transient(*args):
        calls.append(1)
        if len(calls) == 1:
            raise dashboard.JevError('Gateway HTTP 503.', retryable=True)
        return {'answers': {}}
    async def no_wait(seconds):
        pass
    monkeypatch.setattr(dashboard, 'evaluate_jev', transient)
    monkeypatch.setattr(dashboard.asyncio, 'sleep', no_wait)
    async def collect():
        return [item async for item in dashboard.checked_events({}, {}, 'precheck')]
    events = asyncio.run(collect())
    assert len(calls) == 2
    assert events[0]['type'] == 'stage'
    assert events[1]['type'] == 'precheck'
    def permanent(*args):
        raise dashboard.JevError('Gateway HTTP 401.')
    monkeypatch.setattr(dashboard, 'evaluate_jev', permanent)
    import pytest
    with pytest.raises(dashboard.JevError, match='401'):
        asyncio.run(collect())
