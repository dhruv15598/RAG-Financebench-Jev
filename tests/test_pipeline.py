import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import pytest
import run
from reranker import format_pair

def test_generation_inputs_exclude_gold_and_preserve_evidence():
    hits = [{'doc_id':'EXAMPLE_2023_10K','page':4,'text':'EPS grew 9%.'}]
    state = run.question_state('What grew?', hits)
    requests = []
    def handle(req):
        body = json.loads(req.content); requests.append(body)
        return httpx.Response(200, json={'model':'test','message':{'content':'EPS [p4]'},'done_reason':'stop'})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = run.generate(client, 'http://localhost:11435', 'test', state)
    assert result['answer_complete']
    assert json.loads(requests[0]['messages'][1]['content']) == state
    assert set(state) == {'question','evidence'}
    assert set(state['evidence'][0]) == {'source_document','page','passage'}

def test_no_truncation_mislabeled_complete():
    def handle(req):
        return httpx.Response(200,json={'message':{'content':'partial'},'done_reason':'length'})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert not run.generate(client,'http://localhost','test',{'question':'Q','evidence':[]})['answer_complete']

def test_invalid_vectors_and_cache_hash_fail_closed(tmp_path):
    for vectors in [[], [[float('nan')]], [[1.],[1.,2.]]]:
        with pytest.raises(ValueError): run.validate_vectors(vectors,2)
    p=tmp_path/'report.pdf';p.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Checksum'): run.download('https://unused.invalid',p,'different')

def test_snapshot_manifest_and_no_gold_question_fields():
    manifest=run.read(run.ROOT/'data/manifest.json');snapshot=run.read(run.ROOT/'data/snapshot.json')
    assert len(manifest['documents']) == 28
    assert len(snapshot['rows']) == len(manifest['questions']) == 10
    assert snapshot['protocol']['historical'] is True
    assert all(set(q)=={'id','company','question'} for q in manifest['questions'])
    assert all(r['answer'] and r['reference_answer'] for r in snapshot['rows'])
    assert [r['id'] for r in snapshot['rows']] == [q['id'] for q in manifest['questions']]

def test_reranker_pair_keeps_full_text():
    text='evidence ' * 10000
    assert text in format_pair('question',text)

def test_actual_empty_page_ocr_when_tesseract_available(tmp_path):
    import shutil
    if not shutil.which('tesseract'): pytest.skip('Local Tesseract not installed')
    fitz=pytest.importorskip('fitz')
    with fitz.open() as source:
        p=source.new_page(width=400,height=200)
        p.insert_text((30,70),'Revenue 123 million',fontsize=24)
        image=p.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes('png')
    path=tmp_path/'scanned.pdf'
    with fitz.open() as scanned:
        p=scanned.new_page(width=400,height=200)
        p.insert_image(p.rect,stream=image)
        scanned.save(path)
    pages=[{'page':1,'text':''}]
    report=run.ocr_empty_pages(path,pages)
    assert '123' in pages[0]['text']
    assert report==[{'page':1,'empty_after_ocr':False}]
