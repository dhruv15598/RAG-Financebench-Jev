"""Downloader boundaries without network access or checkpoint writes."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


def downloader(monkeypatch):
    calls = []
    def record(*args, **kwargs): calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=record, hf_hub_download=record))
    spec = importlib.util.spec_from_file_location('local_download_test', Path(__file__).resolve().parents[1] / 'local_deciders/download.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize('model', ['coreai2', 'decider4-q4'])
def test_retained_downloads_include_pinned_inference_source(monkeypatch, tmp_path, model):
    module, calls = downloader(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['download.py', model, '--root', str(tmp_path)])
    module.main()
    assert len(calls) == 2
    assert all(call[1]['revision'] for call in calls)
    source = calls[1][1]
    assert source['allow_patterns'] == ['decider/*', 'decider_config.json', 'LICENSE*']
    assert source['revision'] == (module.CORE_SOURCE if model == 'coreai2' else module.MAP)
    if model == 'decider4-q4':
        assert calls[0][1]['filename'] == 'decider-4b.v2-Q4_K_M.gguf'


def test_retired_bf16_download_is_rejected_before_network(monkeypatch):
    module, calls = downloader(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['download.py', 'decider4'])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert calls == []
