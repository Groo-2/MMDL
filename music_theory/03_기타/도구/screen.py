#!/usr/bin/env python3
"""Lexical overlap screening. No claim of exhaustive semantic decontamination."""
import argparse
from collections import defaultdict
import difflib
import hashlib
import json
from pathlib import Path
from theory import normalize
from _layout import PROJECT_ROOT,Layout,CACHE_ROOT

ROOT=PROJECT_ROOT
STOP=set('a an the of in on to is are what which for from and or by with as this these it be its give '
         'following question answer note notes music musical theory'.split())


def tokens(text):
    return set(normalize(text).split())-STOP


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,default=ROOT)
    p.add_argument('--references',type=Path,default=CACHE_ROOT/'exclusion_cache'/'public_eval_text.jsonl')
    args=p.parse_args()
    layout=Layout(args.release)
    refs=[json.loads(x) for x in args.references.read_text().splitlines()]
    reftext=[normalize(r['question']) for r in refs]
    refsets=[tokens(r['question']) for r in refs]
    inverted=defaultdict(set)
    for i,ts in enumerate(refsets):
        for token in ts: inverted[token].add(i)
    exact_index=defaultdict(list)
    for i,text in enumerate(reftext):
        if text:exact_index[text].append(i)
    rows=[]
    for split in ['train','dev','test']:
        rows.extend(json.loads(x) for x in layout.records(split).read_text().splitlines())
    exact=[];near=[];review=[]
    for row in rows:
        q=normalize(row['question']);ts=tokens(q)
        candidates=set().union(*(inverted[t] for t in ts))
        for i in exact_index.get(q,[]):
            exact.append({'id':row['id'],'reference_source':refs[i]['source'],'reference_id':refs[i]['source_id']})
        best=(0,None,0)
        for i in candidates:
            other=refsets[i]
            union=ts|other
            jac=len(ts&other)/len(union) if union else 0
            if jac<0.50:continue
            seq=difflib.SequenceMatcher(None,q,reftext[i],autojunk=False).ratio()
            if jac>best[0]:best=(jac,i,seq)
            if jac>=0.85 and seq>=0.82:
                near.append({'id':row['id'],'reference_source':refs[i]['source'],'reference_id':refs[i]['source_id'],
                             'content_token_jaccard':round(jac,4),'sequence_ratio':round(seq,4)})
        if best[1] is not None and best[0]>=0.65:
            i=best[1]
            review.append({'id':row['id'],'reference_source':refs[i]['source'],'reference_id':refs[i]['source_id'],
                           'content_token_jaccard':round(best[0],4),'sequence_ratio':round(best[2],4)})
    report={'status':'PASS' if not exact and not near else 'FAIL','records_screened':len(rows),
            'reference_rows':len(refs),'reference_sha256':hashlib.sha256(args.references.read_bytes()).hexdigest(),
            'reference_sources':dict((s,sum(r['source']==s for r in refs)) for s in sorted(set(r['source'] for r in refs))),
            'normalized_exact_matches':exact,'high_similarity_matches':near,'lower_threshold_review_hits':review,
            'thresholds':{'blocking_token_jaccard':0.85,'blocking_sequence_ratio':0.82,'review_token_jaccard':0.65},
            'scope':'question text only; all reference records excluded from generation sources',
            'limits':['Does not inspect benchmark image content.',
                      'Does not prove absence of paraphrases or the same underlying theory calculation.',
                      'Shared music theory facts and concepts are expected, not a leak by themselves.',
                      'Does not establish non-overlap with private Qwen pretraining or internal evaluations.']}
    (layout.audit/'public_eval_screening.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(bool(exact or near))


if __name__=='__main__':main()
