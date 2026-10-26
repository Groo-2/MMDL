#!/usr/bin/env python3
"""Strict MCQ grading + family-cluster bootstrap + paired McNemar test."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re


def read_unique(path):
    out={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['id'] in out:raise ValueError(f'Duplicate id: {row["id"]}')
        out[row['id']]=row
    return out


def extract(text):
    found=re.findall(r'^\s*Answer\s*:\s*([ABCD])(?=\s|[.,;)]|$)',text,re.I|re.M)
    return found[0].upper() if len(found)==1 else None


def cluster_ci(rows,field='correct',reps=2000):
    groups=defaultdict(list)
    for row in rows:groups[row['family_id']].append(row[field])
    keys=sorted(groups);rng=random.Random(20261026);values=[]
    for _ in range(reps):
        sampled=[v for key in rng.choices(keys,k=len(keys)) for v in groups[key]]
        values.append(sum(sampled)/len(sampled))
    values.sort()
    return [values[int(0.025*reps)],values[min(reps-1,int(0.975*reps))]]


def evaluate(predictions,key):
    if set(predictions)!=set(key):
        raise ValueError(f'ID mismatch: missing={len(set(key)-set(predictions))}, extra={len(set(predictions)-set(key))}')
    rows=[]
    for id_,truth in key.items():
        pred=predictions[id_];parsed=extract(pred['response'])
        rows.append({'id':id_,'category':truth['category'],'task':truth['task'],
                     'family_id':truth['family_id'],'parsed':parsed,
                     'correct':int(parsed==truth['answer_letter']),
                     'invalid':int(parsed is None),'length_limited':int(pred.get('length_limited',False))})
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
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    key=read_unique(args.key)
    result,rows=evaluate(read_unique(args.predictions),key)
    if args.baseline:
        for pred_path in [args.predictions,args.baseline]:
            if not pred_path.with_suffix('.manifest.json').exists():
                raise ValueError('Paired real runs require predict.py manifests for both model outputs.')
        newmeta=json.loads(args.predictions.with_suffix('.manifest.json').read_text())
        basemeta=json.loads(args.baseline.with_suffix('.manifest.json').read_text())
        for field in ['question_sha256','decoding','model','revision','versions','gpu']:
            if newmeta.get(field)!=basemeta.get(field):raise ValueError(f'Unequal before/after conditions: {field}')
        base,base_rows=evaluate(read_unique(args.baseline),key)
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
    result['grading']='One explicit Answer: A/B/C/D line; missing, ambiguous, or repeated answer lines count as incorrect.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
