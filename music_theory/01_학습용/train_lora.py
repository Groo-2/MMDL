#!/usr/bin/env python3
"""Qwen3-VL text-only LLM LoRA entry point. No automatic CUDA/driver changes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'03_기타'/'도구'))
from _layout import Layout
from data_io import load_sft,tokenize_response_only


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,default=ROOT,help='Project root or an older generated release directory')
    p.add_argument('--model',required=True,help='Exact local baseline checkpoint path or HF model identifier')
    p.add_argument('--revision',help='Pinned HF commit if using a repository identifier')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=float,default=2)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--max-length',type=int,default=1536)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--grad-accum',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dry-run',action='store_true',help='Validate inputs without loading model or CUDA')
    args=p.parse_args()
    layout=Layout(args.release)
    train_path=layout.training/'train.jsonl';dev_path=layout.training/'dev_loss.jsonl'
    train=load_sft(train_path);dev=load_sft(dev_path)
    frozen=json.loads(layout.manifest.read_text()).get('file_sha256',{})
    for path in [train_path,dev_path]:
        expected=frozen.get(path.relative_to(layout.root).as_posix())
        if not expected:
            raise SystemExit('Release is not frozen. Data manager must run package_release.py first.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit(f'Dataset changed after validation/freeze: {path}')
    for report_name in ['validation.json','public_eval_screening.json']:
        report=json.loads((layout.audit/report_name).read_text())
        if report['status']!='PASS':raise SystemExit(f'Required release audit failed: {report_name}')
    print(f'Inputs: train={len(train)}, dev={len(dev)}. Final test files are not read.')
    if args.dry_run:
        print('DRY RUN PASS: no model loaded; no GPU touched.');return
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Output folder must be new or empty.')
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer,Qwen3VLForConditionalGeneration,
                              Trainer,TrainingArguments,set_seed)
    from peft import LoraConfig,get_peft_model
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required for this training entry point.')
    if torch.cuda.device_count()!=1:
        raise SystemExit('Expose exactly one assigned GPU with CUDA_VISIBLE_DEVICES before using this single-GPU recipe.')
    set_seed(args.seed)
    precision=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,revision=args.revision,
                                                         torch_dtype=precision,attn_implementation='sdpa')
    model.config.use_cache=False
    for param in model.parameters():param.requires_grad=False
    # Explicitly target language-model attention only. Vision attention modules
    # must not match, even if the installed Transformers naming changes.
    names=[name for name,module in model.named_modules()
           if name.endswith(('q_proj','k_proj','v_proj','o_proj'))
           and 'language_model.layers.' in name and 'visual' not in name]
    if not names:raise RuntimeError('No language-model attention projections found; inspect model naming.')
    targets='^(?:'+'|'.join(re.escape(name) for name in names)+')$'
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=2*args.rank,
                        target_modules=targets,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
    assert not any('visual' in name and param.requires_grad for name,param in model.named_parameters())
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class Tokenized(Dataset):
        def __init__(self,rows):
            self.rows=[tokenize_response_only(tokenizer,r['messages'],args.max_length) for r in rows]
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):return self.rows[index]

    def collate(batch):
        width=max(len(r['input_ids']) for r in batch)
        result={}
        for name,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]:
            result[name]=torch.tensor([r[name]+[pad]*(width-len(r[name])) for r in batch])
        return result

    train_data=Tokenized(train);dev_data=Tokenized(dev)
    training_args=TrainingArguments(output_dir=str(args.output),num_train_epochs=args.epochs,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,learning_rate=args.lr,
        bf16=precision==torch.bfloat16,fp16=precision==torch.float16,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=2,
        warmup_ratio=0.05,lr_scheduler_type='cosine',logging_steps=10,
        dataloader_num_workers=0,report_to='none',seed=args.seed)
    trainer=Trainer(model=model,args=training_args,train_dataset=train_data,
                    eval_dataset=dev_data,data_collator=collate)
    trainer.train()
    trainer.save_model(str(args.output/'adapter'))
    tokenizer.save_pretrained(str(args.output/'adapter'))
    run={'model':args.model,'revision':args.revision,'settings':vars(args),
         'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
         'dev_sha256':hashlib.sha256(dev_path.read_bytes()).hexdigest(),
         'targeted_modules':names,'max_tokens_observed':max(len(r['input_ids']) for r in train_data.rows),
         'versions':{pkg:importlib.metadata.version(pkg) for pkg in ['torch','transformers','peft']},
         'gpu':torch.cuda.get_device_name(0),'test_data_read':False,
         'status':'training_completed','selected_checkpoint':trainer.state.best_model_checkpoint}
    (args.output/'run_manifest.json').write_text(json.dumps(run,default=str,indent=2),encoding='utf-8')


if __name__=='__main__':main()
