import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import page_lookup, expand_pages


def test_page_continuation_order_overlap_and_document_boundary():
    chunks = [
        {'doc_id': 'A', 'page': 2, 'chunk_id': 'A#10', 'text': 'operating 10 investing 20 financing -5'},
        {'doc_id': 'B', 'page': 2, 'chunk_id': 'B#1', 'text': 'unrelated company'},
        {'doc_id': 'A', 'page': 2, 'chunk_id': 'A#9', 'text': 'Company A year 2022 operating 10'}]
    hits = [chunks[0], chunks[2]]
    expanded = expand_pages(hits, page_lookup(chunks))
    assert len(expanded) == 1
    assert expanded[0]['text'] == 'Company A year 2022 operating 10 investing 20 financing -5'
    assert 'unrelated' not in expanded[0]['text']
    with pytest.raises(ValueError, match='budget'):
        expand_pages(hits, page_lookup(chunks), max_words=3)
    with pytest.raises(ValueError, match='missing'):
        expand_pages(hits, {})


def test_amd_regression_has_all_totals_and_original_page_selection():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/'data/snapshot.json').read_text(encoding='utf-8'))['rows']
    new = json.loads((root/'data/dashboard-evidence.json').read_text(encoding='utf-8'))['rows']
    assert [r['id'] for r in old] == [r['id'] for r in new]
    for before, after in zip(old, new):
        assert set(after) == {'id', 'evidence'}  # no reference answers or stale verdicts
        selected = list(dict.fromkeys((e['source_document'], e['page']) for e in before['evidence']))
        assert [(e['source_document'], e['page']) for e in after['evidence']] == selected
    amd = next(r for r in new if r['id'] == 'financebench_id_01279')
    text = next(e['passage'] for e in amd['evidence'] if e['source_document']=='AMD_2022_10K' and e['page']==58)
    for row in ['operating activities 3,565', 'investing activities 1,999', 'financing activities (3,264)']:
        assert row in text
    assert 'December 31, 2022' in text and '(In millions)' in text
