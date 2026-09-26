"""Allow-listed decision checkers; stop only processes started by this dashboard."""
import asyncio
import json
import math
import os
import shutil
import socket
import signal
import subprocess
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
import httpx

class DecisionError(RuntimeError):
    """Safe client-authored error."""

@dataclass(frozen=True)
class DecisionModel:
    id: str
    label: str
    kind: str = 'local'
    model: str = ''
    command: tuple = ()
    port: int = 8011
    cwd: str | None = None
    endpoint: str = '/v1/systemone'
    ready_endpoint: str = '/v1/models'
    min_free_mib: int = 12000
    startup_timeout: int = 180
    enabled: bool = True
    device: str = 'cuda'

DEFAULT_MODELS = (
    DecisionModel('jev', 'Jev', 'jev', min_free_mib=0),
    DecisionModel('decider-2b-coreai', 'Decider 2B CoreAI', model='decider-2b-coreai', port=8011, min_free_mib=8000),
    DecisionModel('decider-4b-v2-q4', 'Decider 4B v2 Q4', model='decider-4b-v2-q4', port=8012, min_free_mib=6000),
)
RETIRED_MODEL_IDS = frozenset({'kev-0.8b', 'kev-4b', 'decider-4b-v2'})

def load_models(path=None):
    value = os.environ.get('DECISION_MODELS_CONFIG', '').strip()
    path = path or (Path(value).expanduser() if value else None)
    if path is None:
        return DEFAULT_MODELS
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        entries = raw['models'] if isinstance(raw, dict) else raw
        overrides = {}
        for item in entries:
            if item['id'] in RETIRED_MODEL_IDS:
                warnings.warn(f"Retired decision model '{item['id']}' was ignored. Remove its entry from DECISION_MODELS_CONFIG; use Jev, CoreAI 2B or Decider Q4.", UserWarning, stacklevel=2)
                continue
            if item['id'] not in {m.id for m in DEFAULT_MODELS} or item['id'] in overrides:
                raise ValueError('unknown or duplicate id')
            overrides[item['id']] = item
        result = []
        allowed = {'model', 'command', 'port', 'cwd', 'endpoint', 'ready_endpoint', 'min_free_mib', 'startup_timeout', 'enabled', 'device'}
        for base in DEFAULT_MODELS:
            item = {k: v for k, v in overrides.get(base.id, {}).items() if k != 'id'}
            if set(item) - allowed:
                raise ValueError('unknown config field')
            command = item.get('command', [])
            if not isinstance(command, list) or any(not isinstance(x, str) or not x for x in command):
                raise ValueError('command must be an argument list')
            item['command'] = tuple(command)
            model = replace(base, **item)
            if model.device not in {'cuda', 'cpu'}:
                raise ValueError('device must be cuda or cpu')
            if not 1024 <= model.port <= 65535 or model.min_free_mib < 0 or not 1 <= model.startup_timeout <= 900:
                raise ValueError('invalid runtime limits')
            if any(not isinstance(x, str) or not x.startswith('/') or x.startswith('//') for x in (model.endpoint, model.ready_endpoint)):
                raise ValueError('endpoint must be a local path')
            if model.cwd is not None and not isinstance(model.cwd, str):
                raise ValueError('cwd must be a string')
            result.append(model)
        return tuple(result)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DecisionError('Invalid decision runtime configuration; check the server config file.') from exc

def public_models(models=None):
    return [{'id': m.id, 'label': m.label, 'kind': m.kind,
             'device': 'remote' if m.kind == 'jev' else m.device,
             'min_free_mib': m.min_free_mib if m.device == 'cuda' else 0,
             'available': m.kind == 'jev' or bool(m.command),
             'reason': '' if m.kind == 'jev' or m.command else 'Configure a local launch command.'}
            for m in (models if models is not None else load_models()) if m.enabled]

class OwnedProcess:
    def __init__(self):
        self.process = None
    def start(self, argv, cwd=None):
        if self.process is not None:
            raise DecisionError('A checker is already managed by this run.')
        self.process = subprocess.Popen(list(argv), cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
            start_new_session=os.name == 'posix')
    def stop(self):
        process, self.process = self.process, None
        if process is None or (os.name != 'posix' and process.poll() is not None):
            return
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)
        if os.name == 'posix':
            # Reap any descendants that outlived their terminated leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

def gpu_memory():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return None
    try:
        out = subprocess.check_output([executable, '--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=4)
        return tuple(int(x.strip()) for x in out.splitlines()[0].split(','))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None

async def memory_preflight(minimum):
    memory = await asyncio.to_thread(gpu_memory)
    if memory is not None and memory[0] < minimum:
        raise DecisionError(f'Only {memory[0]} MiB GPU memory is free; this phase requires {minimum} MiB. Choose a smaller model or configure a CPU checker, or close another GPU workload. Sequential loading cannot make an oversized individual model fit.')

async def unload_ollama(url):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url + '/api/ps')
        response.raise_for_status()
        for model in response.json().get('models', []):
            response = await client.post(url + '/api/generate', json={'model': model['name'], 'keep_alive': 0})
            response.raise_for_status()
        for _ in range(20):
            response = await client.get(url + '/api/ps')
            response.raise_for_status()
            if not response.json().get('models', []):
                return
            await asyncio.sleep(.25)
    raise DecisionError('Connected Ollama models did not unload; checker loading was refused.')

class LocalChecker:
    def __init__(self, model):
        self.model = model
        self.owner = OwnedProcess()
        self.url = f'http://127.0.0.1:{model.port}'
    async def start(self):
        if not self.model.command:
            raise DecisionError('Configure a launch command for this decision model.')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', self.model.port)) == 0:
                raise DecisionError(f'Port {self.model.port} is already occupied; refusing to replace another service.')
        if self.model.device == 'cuda':
            await memory_preflight(self.model.min_free_mib)
        # Popen returns immediately. Keep ownership assignment atomic with
        # respect to task cancellation before the first readiness await.
        self.owner.start(self.model.command, self.model.cwd)
        deadline = time.monotonic() + self.model.startup_timeout
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                if self.owner.process.poll() is not None:
                    raise DecisionError('Decision checker exited during initialization. Check its launch configuration.')
                try:
                    response = await client.get(self.url + self.model.ready_endpoint)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
        raise DecisionError('Decision checker initialization timed out.')
    async def stop(self):
        await asyncio.to_thread(self.owner.stop)
    async def evaluate(self, state, criteria):
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(self.url + self.model.endpoint,
                json={'model': self.model.model, 'state': state, 'questions': criteria})
            response.raise_for_status()
            result = response.json()
        validate_result(result, criteria)
        result['client_elapsed_seconds'] = round(time.perf_counter() - started, 4)
        return result

def validate_result(result, criteria):
    if not isinstance(result, dict) or not isinstance(result.get('answers'), dict):
        raise DecisionError('Local checker returned an invalid structured result.')
    for key, question in criteria.items():
        answer = result['answers'].get(key, {})
        if not isinstance(answer, dict):
            raise DecisionError('Local checker returned an invalid choice.')
        probabilities = answer.get('probabilities', {})
        if (answer.get('type') != 'choice' or answer.get('choice') not in question['criteria']
            or not isinstance(probabilities, dict) or set(probabilities) != set(question['criteria'])
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > .02):
            raise DecisionError('Local checker returned an invalid choice or probability distribution.')
