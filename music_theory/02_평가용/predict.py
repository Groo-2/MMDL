#!/usr/bin/env python3
"""Text-only inference for before/after comparisons. No answer-key loading."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--revision')
    p.add_argument('--adapter',type=Path)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    rows=[json.loads(x) for x in args.questions.read_text(encoding='utf-8').splitlines()]
    assert rows and all(not any(k in r for k in ['answer_text','answer_letter','spec']) for r in rows)
    if args.output.exists():raise SystemExit('Prediction output already exists; use a new filename.')
    import torch
    from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration,set_seed
    if not torch.cuda.is_available():raise SystemExit('This inference recipe expects a CUDA GPU.')
    set_seed(args.seed)
    precision=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,revision=args.revision,
                        torch_dtype=precision,attn_implementation='sdpa').to('cuda')
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(args.adapter))
    model.eval();model.config.use_cache=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle,torch.inference_mode():
        for index,row in enumerate(rows):
            encoded=tokenizer.apply_chat_template(row['messages'],tokenize=True,
                            add_generation_prompt=True,return_tensors='pt').to('cuda')
            result=model.generate(input_ids=encoded,attention_mask=torch.ones_like(encoded),
                do_sample=False,max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            generated=result[0,encoded.shape[-1]:]
            text=tokenizer.decode(generated,skip_special_tokens=True)
            item={'id':row['id'],'response':text,'output_tokens':len(generated),
                  'length_limited':len(generated)>=args.max_new_tokens}
            handle.write(json.dumps(item,ensure_ascii=False)+'\n');handle.flush()
            if (index+1)%20==0:print(f'{index+1}/{len(rows)}',flush=True)
    metadata={'model':args.model,'revision':args.revision,'adapter':str(args.adapter) if args.adapter else None,
              'question_sha256':hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              'decoding':{'do_sample':False,'max_new_tokens':args.max_new_tokens,'seed':args.seed},
              'versions':{pkg:importlib.metadata.version(pkg) for pkg in ['torch','transformers']},
              'gpu':torch.cuda.get_device_name(0),'answer_key_read':False}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':main()
