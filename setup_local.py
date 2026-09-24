"""Portable local setup for the finance RAG share.

Run from an active Python 3.12 environment with the project dependencies already
installed. This script does not install operating-system packages.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def _url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _background(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    kwargs = {"stdout": handle, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
              "start_new_session": True}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(cmd, env=env, **kwargs)
    handle.close()


def _wait(url: str, seconds: int = 90) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _url_ok(url):
            return
        time.sleep(2)
    raise RuntimeError(f"service did not become ready: {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--qwen38", action="store_true", help="Also download Qwen 3.8 27B UD-IQ2_S (8.37 GB).")
    ap.add_argument("--runtime-dir", type=Path, default=Path(".runtime"))
    ap.add_argument("--model-dir", type=Path, default=Path(".runtime/models/reranker"))
    ap.add_argument("--ollama-port", type=int, default=11435)
    ap.add_argument("--reranker-port", type=int, default=11436)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    runtime = (root / args.runtime_dir).resolve()
    model_dir = (root / args.model_dir).resolve()
    requirements = root / "requirements.txt"
    reranker = root / "reranker.py"
    runner = root / "run.py"
    commands = {name: shutil.which(name) for name in ("git", "tesseract", "ollama")}
    report = {"python": sys.version.split()[0], "root": str(root), "runtime": str(runtime),
              "model_dir": str(model_dir), "requirements": requirements.is_file(),
              "reranker": reranker.is_file(), "run": runner.is_file(), "commands": commands,
              "ollama_url": f"http://127.0.0.1:{args.ollama_port}",
              "reranker_url": f"http://127.0.0.1:{args.reranker_port}"}
    print(json.dumps(report, indent=2))
    missing = [k for k in ("requirements", "reranker", "run") if not report[k]]
    missing += [k for k, value in commands.items() if value is None]
    if missing:
        raise SystemExit("missing prerequisites: " + ", ".join(missing))
    if args.check_only:
        return 0

    runtime.joinpath("logs").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    ollama_url = report["ollama_url"]
    ollama_env = os.environ.copy()
    ollama_env["OLLAMA_HOST"] = f"127.0.0.1:{args.ollama_port}"
    if not _url_ok(ollama_url + "/api/tags"):
        _background([commands["ollama"], "serve"], runtime / "logs/ollama.log", ollama_env)
    _wait(ollama_url + "/api/tags")
    _run([commands["ollama"], "pull", "embeddinggemma:latest"], env=ollama_env)
    _run([commands["ollama"], "pull", "qwen3.5:2b"], env=ollama_env)
    if args.qwen38:
        _run([sys.executable, str(root / "download_qwen38.py"), "--ollama-url", ollama_url])

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required in the active environment") from exc
    snapshot_download(repo_id="Qwen/Qwen3-Reranker-0.6B", revision=REVISION,
                      local_dir=str(model_dir))

    reranker_url = report["reranker_url"]
    if not _url_ok(reranker_url + "/health"):
        _background([sys.executable, str(reranker), "--host", "127.0.0.1", "--port",
                     str(args.reranker_port), "--model", str(model_dir), "--device", "auto"],
                    runtime / "logs/reranker.log", os.environ.copy())
    _wait(reranker_url + "/health")
    _run([sys.executable, str(runner), "--prepare", "--ollama-url", ollama_url,
          "--rerank-url", reranker_url + "/v1"])
    print("Runtime ready; python run.py --limit 10 for the ten-question report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
