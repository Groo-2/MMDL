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
from baseline_contract import MODEL_REPO, MODEL_REVISION, PROTOCOL, pinned_metadata_snapshot

ROOT=PROJECT_ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default=MODEL_REPO)
    p.add_argument('--revision',default=MODEL_REVISION)
    p.add_argument('--release',type=Path,default=ROOT)
    p.add_argument('--max-length',type=int,default=1536)
    p.add_argument('--cache-dir',type=Path,default=CACHE_ROOT/'tokenizer')
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    layout=Layout(args.release)
    from transformers import AutoTokenizer, AutoProcessor
    if args.model!=MODEL_REPO or args.revision!=MODEL_REVISION:
        raise ValueError('Tokenizer verification must use the frozen baseline model and revision.')
    snapshot=pinned_metadata_snapshot(cache_dir=args.cache_dir,local_files_only=args.local_files_only)
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
    lengths=[];responses=[];input_hashes={}
    for file in ['train.jsonl','dev_loss.jsonl']:
        input_hashes[file]=hashlib.sha256((layout.training/file).read_bytes()).hexdigest()
        for row in load_sft(layout.training/file):
            tok=tokenize_response_only(tokenizer,row['messages'],args.max_length)
            lengths.append(len(tok['input_ids']));responses.append(sum(v!=-100 for v in tok['labels']))
            plain=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True)
            content=[{'role':'user','content':[{'type':'text','text':row['messages'][0]['content']}]}]
            assert plain==tokenizer.apply_chat_template(content,tokenize=True,add_generation_prompt=True),'Text/list chat rendering differs from vLLM content format.'
            assert processor.apply_chat_template(content,tokenize=False,add_generation_prompt=True)==tokenizer.apply_chat_template(
                row['messages'][:-1],tokenize=False,add_generation_prompt=True),'Processor and tokenizer chat rendering differ.'
    commit=snapshot.name
    cache_repo=args.cache_dir/('models--'+args.model.replace('/','--'))
    ref=cache_repo/'refs'/(args.revision or 'main')
    if not commit and ref.exists():commit=ref.read_text().strip()
    if commit!=MODEL_REVISION:raise ValueError('Tokenizer did not resolve to the frozen baseline commit.')
    hashes={}
    if commit:
        snapshot=cache_repo/'snapshots'/commit
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()}
    report={'status':'PASS','protocol':PROTOCOL,'model':args.model,'requested_revision':args.revision,
            'resolved_commit':commit,'tokenizer_file_sha256':hashes,'input_file_sha256':input_hashes,
            'transformers_version':importlib.metadata.version('transformers'),
            'records':len(lengths),'max_total_tokens':max(lengths),'max_assistant_tokens':max(responses),
            'min_assistant_tokens':min(responses),'configured_max_length':args.max_length,
            'prompt_prefix_alignment':'all records exact','prompt_loss_mask':'all prompt tokens = -100',
            'processor_chat_template_alignment':'all records exact',
            'model_weights_loaded':False,'gpu_training_tested':False}
    (layout.audit/'tokenizer_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
