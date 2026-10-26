#!/usr/bin/env python3
"""Pinned-baseline text SFT in the separate Transformers training environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '03_기타' / '도구'))
from _layout import Layout
from baseline_contract import MODEL_REPO, MODEL_REVISION, PROTOCOL, assert_baseline, check_model_source
from data_io import load_sft

EXPECTED_VERSIONS = {'transformers': '4.57.1', 'peft': '0.17.1'}
PROJECTION = re.compile(r'(?:model\.)?language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def validate_training_environment(version_lookup=None):
    lookup = version_lookup or importlib.metadata.version
    try:
        lookup('vllm')
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError('vLLM is installed. Activate the separate music-theory training environment.')
    versions = {}
    for package, expected in EXPECTED_VERSIONS.items():
        actual = lookup(package)
        if actual != expected:
            raise ValueError(f'Training requires {package}=={expected}; found {actual}.')
        versions[package] = actual
    for package in ['torch', 'accelerate', 'safetensors', 'Pillow']:
        versions[package] = lookup(package)
    return versions


def validate_release(release):
    layout = Layout(release)
    manifest = json.loads(layout.manifest.read_text())
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Dataset protocol differs from the current baseline-aligned release.')
    paths = [layout.training / 'train.jsonl', layout.training / 'dev_loss.jsonl']
    data = []
    for path in paths:
        relative = path.relative_to(layout.root).as_posix()
        expected = manifest.get('file_sha256', {}).get(relative)
        if not expected or sha256(path) != expected:
            raise ValueError(f'Dataset changed after validation/freeze: {relative}')
        data.append(load_sft(path))
    for name in ['validation.json', 'public_eval_screening.json', 'tokenizer_check.json']:
        report = json.loads((layout.audit / name).read_text())
        if report.get('status') != 'PASS':
            raise ValueError(f'Required release audit failed: {name}')
        if name == 'tokenizer_check.json':
            expected_identity = {'protocol': PROTOCOL, 'model': MODEL_REPO,
                                 'requested_revision': MODEL_REVISION, 'resolved_commit': MODEL_REVISION}
            if any(report.get(field) != expected for field, expected in expected_identity.items()):
                raise ValueError('Tokenizer report differs from the pinned baseline/protocol.')
            expected_inputs = {path.name: sha256(path) for path in paths}
            if report.get('input_file_sha256') != expected_inputs:
                raise ValueError('Tokenizer report does not cover the current training/development inputs.')
    return layout, data[0], data[1]


def require_empty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output folder must be new or empty.')


def target_pattern(names):
    if not names or any(not PROJECTION.fullmatch(name) for name in names):
        raise ValueError('LoRA targets must be language-model self-attention q/k/v/o projections.')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate LoRA target names.')
    return '^(?:' + '|'.join(re.escape(name) for name in names) + ')$'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, default=ROOT)
    parser.add_argument('--model', default=MODEL_REPO)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--local-files-only', action='store_true', help='Use the pinned snapshot already cached under HF_HOME.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=1536, help='Training example length; separate from inference max_model_len.')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Validate the baseline contract and training data without loading weights or CUDA.')
    args = parser.parse_args()
    if any(value <= 0 for value in [args.epochs, args.lr, args.max_length, args.rank, args.grad_accum]):
        parser.error('Training hyperparameters must be positive.')
    if args.model != MODEL_REPO:
        parser.error('Training must start from the pinned baseline repository; use HF_HOME for an offline snapshot.')
    check_model_source(args.model, args.revision)
    contract = assert_baseline()
    layout, train, dev = validate_release(args.release)
    require_empty_output(args.output)
    print(f'Inputs: train={len(train)}, dev={len(dev)}; protocol={PROTOCOL}.')
    if args.dry_run:
        print('DRY RUN PASS: data/contract only; no weights loaded and no GPU accessed.')
        return

    versions = validate_training_environment()
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from data_io import tokenize_response_only

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Expose exactly one assigned CUDA GPU through CUDA_VISIBLE_DEVICES.')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('This baseline-aligned recipe requires bf16; fp16 fallback is disabled.')
    set_seed(args.seed)
    source = {'revision': MODEL_REVISION, 'local_files_only': args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, **source)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_REPO, torch_dtype=torch.bfloat16, attn_implementation='sdpa', **source)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    names = [name for name, _ in model.named_modules() if PROJECTION.fullmatch(name)]
    targets = target_pattern(names)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, target_modules=targets,
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', revision=MODEL_REVISION))
    if any('visual' in name and parameter.requires_grad for name, parameter in model.named_parameters()):
        raise RuntimeError('Vision parameters must remain frozen.')
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class Tokenized(Dataset):
        def __init__(self, rows):
            self.rows = [tokenize_response_only(tokenizer, row['messages'], args.max_length) for row in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def collate(batch):
        width = max(len(row['input_ids']) for row in batch)
        return {name: torch.tensor([row[name] + [pad] * (width - len(row[name])) for row in batch])
                for name, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    train_data, dev_data = Tokenized(train), Tokenized(dev)
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=2,
        save_safetensors=True, warmup_ratio=0.05, lr_scheduler_type='cosine',
        logging_steps=10, dataloader_num_workers=0, report_to='none', seed=args.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=dev_data, data_collator=collate)
    trainer.train()
    adapter = args.output / 'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    run = {
        'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION, 'protocol': PROTOCOL,
        'status': 'training_completed', 'baseline_contract': contract,
        'settings': vars(args), 'targeted_modules': names, 'precision': 'bfloat16',
        'train_sha256': sha256(layout.training / 'train.jsonl'),
        'dev_sha256': sha256(layout.training / 'dev_loss.jsonl'),
        'adapter_file_sha256': file_hashes(adapter),
        'max_tokens_observed': max(len(row['input_ids']) for row in train_data.rows),
        'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'test_data_read': False,
        'selected_checkpoint': trainer.state.best_model_checkpoint,
    }
    (args.output / 'run_manifest.json').write_text(json.dumps(run, default=str, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
