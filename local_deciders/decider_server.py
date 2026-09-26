"""SystemOne adapter for the pinned CoreAI 2B checkpoint and inference source."""
from __future__ import annotations

import json, os, time
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MODEL_DIR = os.environ.get("DECIDER_MODEL")
INFER_ROOT = os.environ.get("DECIDER_INFER_ROOT")
if not MODEL_DIR:
    raise RuntimeError("DECIDER_MODEL must point to the selected checkpoint")
if INFER_ROOT:
    import sys
    sys.path.insert(0, INFER_ROOT)
from decider.infer import Decider  # noqa: E402

DEVICE = os.environ.get("DECIDER_DEVICE", "cuda")
if DEVICE != "cpu" and os.environ.get("DECIDER_ALLOW_CUDA") != "1":
    raise RuntimeError("CUDA requires explicit DECIDER_ALLOW_CUDA=1")
MAX_STATE_TOKENS = int(os.environ.get("DECIDER_MAX_STATE_TOKENS", "32768"))
MODEL_ALIAS = os.environ.get("DECIDER_MODEL_ALIAS", "decider-2b-coreai")
app = FastAPI(title="Local Decider")
decider = Decider(MODEL_DIR, device=DEVICE, use_graphs=False)

class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None

def exact_state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))

def state_token_count(text: str) -> int:
    return len(decider.m.tok(text, add_special_tokens=False)["input_ids"])

@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"data": [{"id": MODEL_ALIAS, "object": "model", "device": DEVICE, "dtype": "bfloat16"}]}

@app.post("/v1/systemone")
def systemone(req: SystemOneRequest) -> dict[str, Any]:
    state = exact_state_text(req.state)
    tokens = state_token_count(state)
    if tokens > MAX_STATE_TOKENS:
        raise HTTPException(413, f"state is {tokens} tokens; limit is {MAX_STATE_TOKENS}; refusing truncation")
    started = time.perf_counter()
    try:
        raw = decider.system_one(state, req.questions, independent=True, max_state_tokens=MAX_STATE_TOKENS, max_fwd_tokens=65536)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    raw["model"] = MODEL_ALIAS
    raw["usage"] = dict(raw.get("usage") or {}, state_tokens=tokens)
    raw["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return raw

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("DECIDER_HOST", "127.0.0.1"), port=int(os.environ.get("DECIDER_PORT", "8011")))
