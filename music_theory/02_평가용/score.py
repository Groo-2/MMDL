#!/usr/bin/env python3
"""Strict MCQ grading + family-cluster bootstrap + paired McNemar test."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'03_기타'/'도구'))
from baseline_contract import PROTOCOL, baseline_module


def read_unique(path):
    out={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['id'] in out:raise ValueError(f'Duplicate id: {row["id"]}')
        out[row['id']]=row
    return out


def extract(text):
    return baseline_module('parsing').parse_multiple_choice(text,4)


def cluster_ci(rows,field='correct',reps=2000):
    groups=defaultdict(list)
    for row in rows:groups[row['family_id']].append(row[field])
    keys=sorted(groups);rng=random.Random(20261026);values=[]
    for _ in range(reps):
        sampled=[v for key in rng.choices(keys,k=len(keys)) for v in groups[key]]
        values.append(sum(sampled)/len(sampled))
    values.sort()
    return [values[int(0.025*reps)],values[min(reps-1,int(0.975*reps))]]


def evaluate(predictions,key,judged=None):
    judged=judged or {}
    if set(predictions)!=set(key):
        raise ValueError(f'ID mismatch: missing={len(set(key)-set(predictions))}, extra={len(set(predictions)-set(key))}')
    rows=[]
    for id_,truth in key.items():
        pred=predictions[id_]
        parsed=baseline_module('parsing').rule_parse(pred)
        method='answer_line'
        if parsed is None:
            if id_ not in judged:raise ValueError(f'Missing baseline judge result: {id_}')
            if judged[id_].get('error'):raise ValueError(f'Judge request failed: {id_}')
            parsed=judged[id_]['judge_pred']
            method='judge' if parsed is not None else 'judge_failed'
        gold=dict(pred,answer=truth['answer_letter'])
        rows.append({'id':id_,'category':truth['category'],'task':truth['task'],
                     'family_id':truth['family_id'],'parsed':parsed,
                     'correct':int(baseline_module('parsing').is_correct(gold,parsed)),
                     'parser_method':method,'invalid':int(parsed is None),
                     'length_limited':int(pred['finish_reason']=='length')})
    by=defaultdict(list)
    for r in rows:by[r['category']].append(r)
    summary={'n':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
             'family_cluster_bootstrap_95pct_ci':cluster_ci(rows),
             'family_count':len({r['family_id'] for r in rows}),
             'invalid_answer_rate':sum(r['invalid'] for r in rows)/len(rows),
             'length_limit_rate':sum(r['length_limited'] for r in rows)/len(rows),
             'category_scores':{k:{'n':len(v),'accuracy':sum(r['correct'] for r in v)/len(v)} for k,v in by.items()}}
    return summary,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--key',type=Path,required=True);p.add_argument('--baseline',type=Path)
    p.add_argument('--judge-results',type=Path)
    p.add_argument('--baseline-judge-results',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    key=read_unique(args.key)
    metadata=validated_metadata(args.predictions)
    result,rows=evaluate(read_unique(args.predictions),key,validated_judges(args.judge_results,args.predictions))
    if args.baseline:
        for pred_path in [args.predictions,args.baseline]:
            if not pred_path.with_suffix('.manifest.json').exists():
                raise ValueError('Paired real runs require predict.py manifests for both model outputs.')
        newmeta=metadata
        basemeta=validated_metadata(args.baseline)
        for field in ['question_sha256','comparison_conditions']:
            if newmeta.get(field)!=basemeta.get(field):raise ValueError(f'Unequal before/after conditions: {field}')
        base,base_rows=evaluate(read_unique(args.baseline),key,validated_judges(args.baseline_judge_results,args.baseline))
        baseline={r['id']:r for r in base_rows}
        gained=sum(not baseline[r['id']]['correct'] and r['correct'] for r in rows)
        lost=sum(baseline[r['id']]['correct'] and not r['correct'] for r in rows)
        discordant=gained+lost
        pvalue=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/(2**discordant)) if discordant else 1.0
        delta_rows=[dict(r,delta=r['correct']-baseline[r['id']]['correct']) for r in rows]
        result['paired']={'baseline_accuracy':base['accuracy'],'difference':result['accuracy']-base['accuracy'],
                          'gained':gained,'lost':lost,'unchanged':len(rows)-discordant,
                          'mcnemar_exact_pvalue':pvalue,
                          'family_cluster_bootstrap_delta_95pct_ci':cluster_ci(delta_rows,'delta'),
                          'note':'McNemar assumes independent pairs; related items exist. Prefer the family bootstrap for uncertainty.'}
    result['protocol']=PROTOCOL
    result['grading']='Unchanged team baseline rule_parse -> judge extraction -> is_correct; no random fallback.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def validated_metadata(path):
    from predict import sha256
    meta=json.loads(path.with_suffix('.manifest.json').read_text())
    if meta.get('protocol')!=PROTOCOL or meta.get('status')!='complete' or meta.get('output_sha256')!=sha256(path):
        raise ValueError('Legacy, incomplete, or changed prediction output.')
    return meta


def validated_judges(path,predictions):
    if path is None:return {}
    from predict import sha256
    meta=json.loads(path.with_suffix('.manifest.json').read_text())
    if (meta.get('protocol')!=PROTOCOL or meta.get('status')!='complete'
        or meta.get('prediction_sha256')!=sha256(predictions) or meta.get('output_sha256')!=sha256(path)):
        raise ValueError('Judge results do not match these predictions.')
    return read_unique(path)


if __name__=='__main__':main()
