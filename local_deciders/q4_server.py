"""Decider v2 GGUF adapter: native plain prompt, direct label logits, no generation."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--native-source', required=True, help='Pinned Mapika v2 source directory')
parser.add_argument('--port', type=int, default=8012)
parser.add_argument('--ctx', type=int, default=32768)
parser.add_argument('--gpu-layers', type=int, default=-1)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
sys.path.insert(0, args.native_source)
from decider import prompt, prompt_fast, systemone as S1

cfg = json.loads((Path(args.native_source) / 'decider_config.json').read_text())
assert cfg['layout'] == 'plain' and not cfg['schema_first']
assert cfg['temperature'] == 1.935 and cfg['neutralize_none'] is False
llm = Llama(model_path=args.model, n_ctx=args.ctx, n_gpu_layers=args.gpu_layers,
            n_threads=args.threads, n_batch=512, logits_all=False, verbose=True)

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return llm.tokenize(text.encode('utf-8'), add_bos=False, special=True)

tok = Tokenizer()
labels, label_ids, _ = prompt.label_table(tok)
assert labels[:10] == list('ABCDEFGHIJ'), 'GGUF tokenizer does not match native option labels'
lock = threading.Lock()
app = FastAPI()
MODEL = 'Decider4v2Q4'

@app.get('/ready')
@app.get('/health')
def ready():
    return {'ready': True, 'model': MODEL, 'backend': 'llama.cpp direct label logits',
            'temperature': cfg['temperature'], 'layout': 'plain', 'context_tokens': args.ctx,
            'truncation': False}

@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL, 'object': 'model', 'owned_by': 'local'}]}

def prepare(state, questions, independent=True):
    # Match dashboard BF16 adapters byte-for-byte: pass JSON as a native string state.
    # Native string states are preserved rather than rewritten with array annotations.
    context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(',', ':'))
    rqs = {key: S1.render_question(spec) for key, spec in questions.items()}
    flat, index = S1.plan_rows(rqs, cfg['isolated_levels'] and independent)
    pairs = [(row['question'], row['options']) for row in flat]
    rows = [[p] for p in pairs] if independent else [pairs]
    # Native helper accepts a limit; use the actual full context length, never its truncating default.
    full_len = len(tok.encode('Context:\n' + context))
    items, ctx_len = prompt_fast.build_rows(tok, context, rows, max_ctx_tokens=full_len)
    if any(len(item['ids']) > args.ctx for item in items):
        raise ValueError(f'Full input exceeds {args.ctx} token context; no state was truncated')
    return rqs, index, items, ctx_len

def score(items):
    result = []
    for item in items:
        llm.reset()
        start = 0
        for slot, count in zip(item['slots'], item['nopts']):
            llm.eval(item['ids'][start:slot + 1])
            # Read logits after the Answer: ( token. Restrict to native option label IDs.
            raw = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),))
            scores = np.array(raw[label_ids[:count]], dtype=np.float64) / cfg['temperature']
            scores -= scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            result.append(probabilities.tolist())
            start = slot + 1
    return result

@app.post('/v1/systemone')
def systemone(request: dict):
    begin = time.perf_counter()
    try:
        rqs, index, items, ctx_len = prepare(request.get('state', ''), request['questions'],
                                           request.get('independent', True))
        with lock:
            probabilities = score(items)
        return {'model': MODEL, 'answers': S1.assemble(rqs, index, probabilities),
                'usage': {'input_tokens': prompt_fast.unique_tokens(items, ctx_len), 'output_tokens': 0},
                'seconds': time.perf_counter() - begin}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=args.port)
