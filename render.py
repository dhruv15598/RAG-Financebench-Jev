"""Render portable FinanceBench results to a readable HTML report."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path
from typing import Any

def esc(v: Any) -> str:
    return html.escape("" if v is None else (json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else str(v)), quote=True)

def verdict(row: dict[str, Any]) -> str:
    p = row.get("postcheck") or {}
    a = p.get("answers") if isinstance(p, dict) else {}
    v = a.get("answer_verdict") if isinstance(a, dict) else {}
    return str(v.get("choice", "—")) if isinstance(v, dict) else "—"

def probs(row: dict[str, Any]) -> str:
    p = row.get("postcheck") or {}
    a = p.get("answers") if isinstance(p, dict) else {}
    v = a.get("answer_verdict") if isinstance(a, dict) else {}
    d = v.get("probabilities") if isinstance(v, dict) else {}
    if not isinstance(d, dict): return ""
    return ", ".join(f"{k}={x}" for k, x in d.items())

def retrieval(row: dict[str, Any]) -> str:
    items = row.get("retrieval") or row.get("retrieved") or row.get("retrieved_evidence") or row.get("evidence") or []
    if not isinstance(items, list): return "<p>None recorded.</p>"
    out=[]
    for i, item in enumerate(items, 1):
        item = item if isinstance(item, dict) else {"passage": item}
        label = f"{i}. {item.get('source_document', item.get('doc_name', item.get('source', 'source')))} · page {item.get('page', item.get('evidence_page_num', '—'))}"
        passage = item.get("passage", item.get("evidence_text", item.get("text", "")))
        score = item.get("score")
        if score is not None: label += f" · retrieval score {score}"
        out.append(f"<details><summary>{esc(label)}</summary><pre>{esc(passage)}</pre></details>")
    return "".join(out) or "<p>None recorded.</p>"

def render(data: dict[str, Any], source: Path) -> str:
    rows = [r for r in data.get("rows", []) if isinstance(r, dict)]
    protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else {}
    model_label = str(protocol.get("model") or "Qwen3.5")
    historical = bool(protocol.get("historical"))
    cards=[]
    for i, row in enumerate(rows, 1):
        ref = row.get("reference_answer", "—")
        cards.append(f'''<article class="card"><h2>{i}. {esc(row.get("company", "Unknown"))}</h2><div class="id">{esc(row.get("id"))}</div>
<h3>Question</h3><p>{esc(row.get("question"))}</p><h3>{esc(model_label)} answer</h3><pre>{esc(row.get("answer"))}</pre>
<h3>FinanceBench reference answer</h3><pre>{esc(ref)}</pre>
<p class="meta">generation seconds: {esc(row.get("generation_seconds", "—"))} · Jev postcheck verdict: <b>{esc(verdict(row))}</b>{(" · Jev probabilities: " + esc(probs(row))) if probs(row) else ""}</p>
<details><summary>Retrieved evidence supplied to {esc(model_label)}</summary>{retrieval(row)}</details></article>''')
    return f'''<!doctype html><meta charset="utf-8"><title>FinanceBench hybrid reranking results</title><style>
body{{font:15px/1.5 system-ui,sans-serif;background:#f5f7fb;color:#172033;max-width:1100px;margin:auto;padding:24px}}.card{{background:white;border:1px solid #d9dfeb;border-radius:10px;padding:18px;margin:16px 0}}h1{{margin-bottom:4px}}h3{{font-size:13px;text-transform:uppercase;color:#5e6b82;letter-spacing:.04em}}pre{{white-space:pre-wrap;background:#f5f7fb;padding:10px;border-radius:6px;overflow-wrap:anywhere}}.id,.meta{{color:#5e6b82;font-size:13px}}summary{{cursor:pointer;color:#315bd6;font-weight:600}}</style>
<h1>FinanceBench hybrid reranking results</h1><p class="meta">Source: {esc(source.name)} · {len(rows)} cases. {"Historical saved snapshot; this is not a fresh run. " if historical else ""}Reference answers are evaluation data and were not sent to {esc(model_label)}. Jev verdicts are diagnostics, not correctness labels.</p>{''.join(cards)}'''

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("results", type=Path); ap.add_argument("-o", "--output", type=Path)
    a=ap.parse_args(); data=json.loads(a.results.read_text(encoding="utf-8")); out=a.output or a.results.with_suffix(".html")
    out.write_text(render(data, a.results), encoding="utf-8", newline="\n"); print(out); return 0
if __name__ == "__main__": raise SystemExit(main())
