"""Expand retrieved chunks to complete source pages without changing retrieval ranks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def page_lookup(chunks):
    pages = defaultdict(list)
    for chunk in chunks:
        item = chunk if isinstance(chunk, dict) else vars(chunk)
        pages[(item['doc_id'], item['page'])].append(item)
    result = {}
    for key, items in pages.items():
        items.sort(key=lambda c: int(c['chunk_id'].rsplit('#', 1)[-1]))
        words = []
        for item in items:
            part = item['text'].split()
            overlap = 0
            for size in range(min(len(words), len(part)), 0, -1):
                if words[-size:] == part[:size]:
                    overlap = size
                    break
            words.extend(part[overlap:])
        result[key] = ' '.join(words)
    return result


def expand_pages(hits, pages, max_words=3600):
    """Keep ranked page order; include full pages and deduplicate same-page hits.

    No question, benchmark annotation or company routing enters this operation.
    Refuse oversized context rather than silently cutting table continuations.
    """
    expanded, seen = [], set()
    for hit in hits:
        key = (hit['doc_id'], hit['page'])
        if key in seen:
            continue
        seen.add(key)
        if key not in pages:
            raise ValueError('Retrieved page is missing from the source index.')
        expanded.append({**hit, 'text': pages[key]})
    if sum(len(h['text'].split()) for h in expanded) > max_words:
        raise ValueError('Expanded evidence exceeds the 3600-word safety budget; no passages were truncated.')
    return expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='corpus.jsonl containing every chunk')
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).parent/'data/snapshot.json')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'data/dashboard-evidence.json')
    args = parser.parse_args()
    pages = page_lookup([json.loads(line) for line in args.index.read_text(encoding='utf-8').splitlines() if line.strip()])
    rows = []
    for row in json.loads(args.snapshot.read_text(encoding='utf-8-sig'))['rows']:
        hits = [{'doc_id': e['source_document'], 'page': e['page'], 'text': e['passage']} for e in row['evidence']]
        expanded = expand_pages(hits, pages)
        rows.append({'id': row['id'], 'evidence': [{'source_document': h['doc_id'], 'page': h['page'], 'passage': h['text']} for h in expanded]})
    args.out.write_text(json.dumps({'method': 'Same ranked source pages, reconstructed from all indexed page chunks; overlap removed. No new retrieval or model judgments.', 'rows': rows}, indent=2), encoding='utf-8')
    print(f'Prepared complete-page evidence for {len(rows)} cases.')


if __name__ == '__main__':
    main()
