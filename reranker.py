"""Local Qwen3 reranker adapter for unmodified Docket's /v1/rerank client."""
from __future__ import annotations
import argparse, threading, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ALIAS = 'qwen3-reranker-0.6b:latest'
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def format_pair(query, document):
    return PREFIX + f'<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}' + SUFFIX

class Request(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None

def create_app(model_path, device='auto', batch_size=4):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.float16 if resolved.startswith('cuda') else torch.float32,
        attn_implementation='sdpa').to(resolved).eval()
    no_id = tokenizer.convert_tokens_to_ids('no'); yes_id = tokenizer.convert_tokens_to_ids('yes')
    app = FastAPI(); lock = threading.Lock()
    stats = {'requests': 0, 'pairs': 0, 'max_input_tokens': 0}

    @app.get('/health')
    def health():
        return {'model': ALIAS, 'device': str(model.device), 'dtype': str(model.dtype), 'truncation': False, **stats}

    @app.post('/v1/rerank')
    def rerank(req: Request):
        if req.model != ALIAS: raise HTTPException(400, 'Unknown model')
        if not req.documents: raise HTTPException(400, 'No documents')
        if req.top_n is not None and req.top_n < 1: raise HTTPException(400, 'top_n must be positive')
        ids = tokenizer([format_pair(req.query, d) for d in req.documents],
                        add_special_tokens=False, truncation=False)['input_ids']
        lengths = [len(x) for x in ids]
        if max(lengths) > 8192: raise HTTPException(422, 'Input exceeds token budget; nothing truncated')
        start = time.perf_counter(); scores = []
        with lock, torch.inference_mode():
            for i in range(0, len(ids), batch_size):
                inputs = tokenizer.pad({'input_ids': ids[i:i + batch_size]}, padding=True,
                                       return_tensors='pt').to(model.device)
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
                scores.extend(torch.softmax(logits[:, [no_id, yes_id]].float(), dim=-1)[:, 1].cpu().tolist())
            stats['requests'] += 1; stats['pairs'] += len(ids)
            stats['max_input_tokens'] = max(stats['max_input_tokens'], max(lengths))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if req.top_n is not None: order = order[:req.top_n]
        return {'model': ALIAS, 'results': [{'index': i, 'relevance_score': scores[i]} for i in order],
                'usage': {'total_tokens': sum(lengths)}, 'elapsed_seconds': time.perf_counter() - start,
                'truncated': False}
    return app

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-Reranker-0.6B')
    p.add_argument('--device', default='auto'); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=11436)
    args = p.parse_args()
    if args.batch_size < 1: p.error('--batch-size must be positive')
    import uvicorn
    uvicorn.run(create_app(args.model, args.device, args.batch_size), host=args.host, port=args.port)

if __name__ == '__main__': main()
