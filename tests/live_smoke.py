"""Optional live integration check on one report; not a benchmark run.

Needs prepared local Ollama embedding/generation and reranking services.
Downloads one pinned public PDF + FinanceBench JSONL into an ignored test cache.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ollama-url',default='http://127.0.0.1:11435')
    p.add_argument('--rerank-url',default='http://127.0.0.1:11436/v1')
    p.add_argument('--model',default='qwen3.5:2b')
    a=p.parse_args()
    manifest=run.read(run.ROOT/'data/manifest.json')
    manifest['documents']=[d for d in manifest['documents'] if d['doc_id']=='PEPSICO_2023Q1_EARNINGS']
    manifest['questions']=[q for q in manifest['questions'] if 'pepsico' in q['question'].lower()]
    args=SimpleNamespace(cache=run.ROOT/'cache/live-smoke',ollama_url=a.ollama_url,model=a.model,
        embed_model='embeddinggemma:latest',rerank_url=a.rerank_url,rerank_model='qwen3-reranker-0.6b:latest',
        jev=False,limit=1,out=run.ROOT/'outputs'/('smoke-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    run.prepare(args,manifest)
    return run.execute(args,manifest)

if __name__=='__main__':raise SystemExit(main())
