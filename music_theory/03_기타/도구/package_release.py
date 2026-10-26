#!/usr/bin/env python3
"""Build separate training/evaluation/admin ZIPs with package-boundary checks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

from _layout import PROJECT_ROOT,Layout
from baseline_contract import BASELINE_ROOT, assert_baseline


def render_review(rows,path,title):
    blocks=[]
    for row in rows:
        options=''
        if row.get('options'):
            options='<ol type="A">'+''.join('<li>'+html.escape(v)+'</li>' for v in row['options'].values())+'</ol>'
        blocks.append('<article><p class="meta">'+html.escape(row['id']+' | '+row['task'])+'</p><p>'+html.escape(row['question'])+'</p>'+options+
                      '<details><summary>정답과 검수 근거</summary><p><b>'+html.escape(row['answer_text'])+'</b></p><p>'+html.escape(row['explanation'])+'</p>'+ 
                      '<pre>'+html.escape(json.dumps(row['spec'],ensure_ascii=False,indent=2))+'</pre></details></article>')
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music theory data review</title><style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#182b40;background:#f5f7fa;line-height:1.65}
article{background:white;border:1px solid #dce3eb;border-radius:10px;padding:20px;margin:16px 0}
.meta{color:#52677d;font-size:13px}summary{cursor:pointer;color:#175c9f}pre{white-space:pre-wrap;font-size:12px}
h1{font-size:26px}li{padding:4px}header{position:sticky;top:0;background:#f5f7fa;padding:12px 0;border-bottom:1px solid #ccd6e2}
input{width:100%;box-sizing:border-box;padding:10px;font-size:16px}</style><h1>'''+html.escape(title)+'''</h1>
<p>개념 정확성, 질문의 모호성, 보기의 복수 정답 여부를 검수하세요. 이 화면은 자동 검증 결과이며 전문가 검수 완료를 뜻하지 않습니다.</p>
<header><input id="filter" placeholder="ID, 유형, 음이름으로 검색" aria-label="문항 검색"></header>'''+''.join(blocks)+'''
<script>document.querySelector('#filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q));});</script></html>'''
    path.write_text(content,encoding='utf-8')


def add_package(path,blobs):
    checks=[]
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for rel,data in sorted(blobs.items()):
            archive.writestr('MMDL/music_theory/'+rel,data)
            checks.append(hashlib.sha256(data).hexdigest()+'  '+rel)
        archive.writestr('MMDL/music_theory/03_기타/검증기록/SHA256SUMS.txt','\n'.join(checks)+'\n')
        assert_baseline()
        for file in sorted(BASELINE_ROOT.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':
                archive.writestr('MMDL/baseline/'+file.relative_to(BASELINE_ROOT).as_posix(),file.read_bytes())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,default=PROJECT_ROOT)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'03_기타'/'배포결과')
    args=p.parse_args()
    layout=Layout(args.release)
    for report in ['validation.json','public_eval_screening.json','tokenizer_check.json']:
        if json.loads((layout.audit/report).read_text())['status']!='PASS':
            raise SystemExit('Failed audit: '+report)
    train=[json.loads(x) for x in layout.records('train').read_text().splitlines()]
    sample=[r for r in train if r['task']=='concept']
    for task in sorted(set(r['task'] for r in train)-{'concept'}):
        sample.extend([r for r in train if r['task']==task][:5])
    layout.training_review.parent.mkdir(parents=True,exist_ok=True)
    render_review(sample,layout.training_review,'학습 데이터 검수: 개념 카드 전부 + 유형별 표본')
    test=[json.loads(x) for x in layout.records('test').read_text().splitlines()]
    layout.test_review.parent.mkdir(parents=True,exist_ok=True)
    render_review(test,layout.test_review,'평가 담당자 전용: 최종 평가 300문항 검수')
    if not layout.human_review.exists():
        layout.human_review.parent.mkdir(parents=True,exist_ok=True)
        review={'status':'pending','reviewer':None,'review_date':None,'release':'v2.2',
                'train_concept_review':'pending','train_stratified_sample_review':'pending',
                'dev_review':'pending','test_review_by_eval_custodian':'pending','corrections':[]}
        layout.human_review.write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')

    # Canonical organized archive paths, also when packaging a raw generated version.
    mapping={}
    def put(relative,source):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        mapping[relative]=source
    put('README.md',PROJECT_ROOT/'README.md')
    put('.gitignore',PROJECT_ROOT/'.gitignore')
    for directory in ['01_학습용','02_평가용']:
        for file in (PROJECT_ROOT/directory).glob('*.py'):put(directory+'/'+file.name,file)
    for directory in ['01_학습용','02_평가용']:
        for file in (PROJECT_ROOT/directory).glob('*.txt'):put(directory+'/'+file.name,file)
    for directory,pattern in [('03_기타/도구','*.py'),('03_기타/설치파일','*.txt'),('04_문서','*.md')]:
        for file in (PROJECT_ROOT/directory).glob(pattern):put(directory+'/'+file.name,file)
    for file in layout.training.glob('*.jsonl'):put('01_학습용/데이터/'+file.name,file)
    for file in layout.evaluation.glob('*.jsonl'):put('02_평가용/데이터/'+file.name,file)
    for file in layout.audit.glob('*.json'):
        if file.name!='manifest.json':put('03_기타/검증기록/'+file.name,file)
    for split in ['train','dev']:put('03_기타/검증기록/'+split+'_records.jsonl',layout.records(split))
    put('03_기타/검증기록/training_review_sample.jsonl',layout.audit/'training_review_sample.jsonl')
    put('03_기타/검증기록/manifest.json',layout.manifest)
    put('03_기타/검수자료/human_review.json',layout.human_review)
    put('03_기타/검수자료/training_review.html',layout.training_review)
    put('02_평가용/검수자료/test_records.jsonl',layout.records('test'))
    put('02_평가용/검수자료/private_test_review.html',layout.test_review)

    manifest=json.loads(layout.manifest.read_text())
    manifest['file_sha256']={file.relative_to(layout.root).as_posix():hashlib.sha256(file.read_bytes()).hexdigest()
                             for file in layout.data_files()}
    manifest['public_reference_counts']=json.loads((layout.audit/'public_eval_screening.json').read_text())['reference_sources']
    manifest['folder_layout']='roles_v1'
    layout.manifest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    blobs={relative:file.read_bytes() for relative,file in mapping.items()}
    if not layout.raw:
        (layout.audit/'SHA256SUMS.txt').write_text(''.join(
            hashlib.sha256(data).hexdigest()+'  '+relative+'\n'
            for relative,data in sorted(blobs.items())),encoding='utf-8')
    archive_manifest=dict(manifest)
    archive_manifest['file_sha256']={relative:hashlib.sha256(data).hexdigest()
                                     for relative,data in blobs.items() if relative.endswith('.jsonl')}
    blobs['03_기타/검증기록/manifest.json']=json.dumps(archive_manifest,indent=2,ensure_ascii=False).encode()

    shared={'README.md','.gitignore','03_기타/도구/_layout.py','03_기타/도구/baseline_contract.py','03_기타/검증기록/manifest.json',
            '03_기타/검수자료/human_review.json'}
    shared|={r for r in blobs if r.startswith('04_문서/')}
    shared|={r for r in blobs if r.startswith('03_기타/검증기록/') and r.endswith('.json')}
    training=shared|{r for r in blobs if r.startswith('01_학습용/')}
    training|={'02_평가용/predict.py','02_평가용/score.py',
               '02_평가용/run_eval.py','02_평가용/requirements-eval.txt',
               '02_평가용/데이터/dev_questions.jsonl','02_평가용/데이터/dev_answer_key.jsonl',
               '03_기타/도구/check_tokenizer.py','03_기타/설치파일/requirements-tokenizer.txt',
               '03_기타/검수자료/training_review.html'}
    evaluation=shared|{r for r in blobs if r.startswith('02_평가용/')}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={}
    for name,selected in [('training',training),('evaluation',evaluation),('pipeline',set(blobs))]:
        path=args.output_dir/f'music_theory_text_v2_{name}.zip'
        payload={r:blobs[r] for r in selected}
        if name=='training':
            assert not any('test_' in r for r in payload)
            assert not any(r.endswith('_records.jsonl') for r in payload)
        add_package(path,payload)
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        summary[name]={'file':path.name,'bytes':path.stat().st_size,
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'contains_test_answers':name!='training','boundary_check':'PASS'}
    (args.output_dir/'package_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
