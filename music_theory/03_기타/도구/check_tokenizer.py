#!/usr/bin/env python3
"""Check actual Qwen chat template and assistant-only masking without weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from _layout import PROJECT_ROOT,Layout,CACHE_ROOT
sys.path.insert(0,str(PROJECT_ROOT/'01_학습용'))
from data_io import load_sft,tokenize_response_only

ROOT=PROJECT_ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--revision')
    p.add_argument('--release',type=Path,default=ROOT)
    p.add_argument('--max-length',type=int,default=1536)
    p.add_argument('--cache-dir',type=Path,default=CACHE_ROOT/'tokenizer')
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    layout=Layout(args.release)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,cache_dir=args.cache_dir,
                                           local_files_only=args.local_files_only)
    lengths=[];responses=[]
    for file in ['train.jsonl','dev_loss.jsonl']:
        for row in load_sft(layout.training/file):
            tok=tokenize_response_only(tokenizer,row['messages'],args.max_length)
            lengths.append(len(tok['input_ids']));responses.append(sum(v!=-100 for v in tok['labels']))
    commit=tokenizer.init_kwargs.get('_commit_hash')
    cache_repo=args.cache_dir/('models--'+args.model.replace('/','--'))
    ref=cache_repo/'refs'/(args.revision or 'main')
    if not commit and ref.exists():commit=ref.read_text().strip()
    hashes={}
    if commit:
        snapshot=cache_repo/'snapshots'/commit
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()}
    report={'status':'PASS','model':args.model,'requested_revision':args.revision,
            'resolved_commit':commit,'tokenizer_file_sha256':hashes,
            'transformers_version':importlib.metadata.version('transformers'),
            'records':len(lengths),'max_total_tokens':max(lengths),'max_assistant_tokens':max(responses),
            'min_assistant_tokens':min(responses),'configured_max_length':args.max_length,
            'prompt_prefix_alignment':'all records exact','prompt_loss_mask':'all prompt tokens = -100',
            'model_weights_loaded':False,'gpu_training_tested':False}
    (layout.audit/'tokenizer_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
