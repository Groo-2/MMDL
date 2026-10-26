#!/usr/bin/env python3
"""Validate a completed training run and export a full pinned-base VLM for vLLM."""
import argparse
import json
from pathlib import Path
import shutil

from train_lora import (MODEL_REPO, MODEL_REVISION, PROTOCOL, EXPECTED_VERSIONS,
                        assert_baseline, check_model_source, file_hashes,
                        require_empty_output, sha256, target_pattern,
                        validate_training_environment)
import baseline_contract

REQUIRED_BASE_METADATA = frozenset({
    'config.json', 'generation_config.json', 'preprocessor_config.json',
    'video_preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
    'vocab.json', 'merges.txt',
})


def copy_base_metadata(snapshot, output):
    """Copy pinned metadata verbatim and remove generated-only alternatives; never touch weights/indexes."""
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if snapshot == output:
        raise ValueError('The merged output must differ from the immutable base snapshot.')
    missing = sorted(name for name in REQUIRED_BASE_METADATA if not (snapshot / name).is_file())
    if missing:
        raise ValueError('Pinned snapshot is missing required metadata: ' + ', '.join(missing))
    if not any((snapshot / name).is_file() for name in ['chat_template.jinja', 'chat_template.json']):
        raise ValueError('Pinned snapshot is missing its original chat template.')
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in baseline_contract.METADATA_FILES:
        source, target = snapshot / name, output / name
        if target.exists() or target.is_symlink():
            if target.is_dir():
                raise ValueError(f'Expected a metadata file, found directory: {target}')
            target.unlink()
        if source.is_file():
            shutil.copyfile(source, target)
            original_hash = sha256(source)
            if sha256(target) != original_hash:
                raise RuntimeError(f'Pinned metadata copy differs: {name}')
            hashes[name] = original_hash
    return hashes


def validate_adapter_metadata(adapter, run_manifest=None):
    adapter = Path(adapter).resolve()
    manifest_path = Path(run_manifest) if run_manifest else adapter.parent / 'run_manifest.json'
    run = json.loads(manifest_path.read_text())
    for field, expected in [('base_model', MODEL_REPO), ('base_revision', MODEL_REVISION),
                            ('protocol', PROTOCOL), ('status', 'training_completed'),
                            ('precision', 'bfloat16')]:
        if run.get(field) != expected:
            raise ValueError(f'Adapter training provenance mismatch: {field}.')
    if run.get('baseline_contract') != assert_baseline():
        raise ValueError('Adapter was trained against a different baseline contract.')
    for package, expected in EXPECTED_VERSIONS.items():
        if run.get('versions', {}).get(package) != expected:
            raise ValueError(f'Adapter training version mismatch: {package}.')
    expected_hashes = run.get('adapter_file_sha256')
    if not expected_hashes or file_hashes(adapter) != expected_hashes:
        raise ValueError('Adapter contents differ from the completed training manifest.')
    if not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('Expected the safetensors LoRA export from train_lora.py.')
    config = json.loads((adapter / 'adapter_config.json').read_text())
    for field, expected in [('base_model_name_or_path', MODEL_REPO), ('revision', MODEL_REVISION),
                            ('peft_type', 'LORA'), ('task_type', 'CAUSAL_LM'), ('bias', 'none')]:
        if config.get(field) != expected:
            raise ValueError(f'Adapter configuration mismatch: {field}.')
    names = run.get('targeted_modules', [])
    expected_pattern = target_pattern(names)
    targets = config.get('target_modules')
    if not (targets == expected_pattern or isinstance(targets, list) and set(targets) == set(names)):
        raise ValueError('Adapter targets differ from the recorded language-only projections.')
    if config.get('modules_to_save'):
        raise ValueError('Unexpected full-model modules in the adapter export.')
    if config.get('r') != run.get('settings', {}).get('rank'):
        raise ValueError('Adapter rank differs from the training settings.')
    return run, manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path)
    parser.add_argument('--model', default=MODEL_REPO)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check adapter metadata/hashes without weights, downloads or GPU access.')
    args = parser.parse_args()
    if args.model != MODEL_REPO:
        parser.error('Merge requires the original pinned baseline repository; use HF_HOME for an offline snapshot.')
    check_model_source(args.model, args.revision)
    contract = assert_baseline()
    run, manifest_path = validate_adapter_metadata(args.adapter, args.run_manifest)
    require_empty_output(args.output)
    if args.dry_run:
        print('MERGE DRY RUN PASS: pinned base, adapter hashes, language-only targets and training provenance verified.')
        return

    versions = validate_training_environment()
    import torch
    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    source = {'revision': MODEL_REVISION, 'local_files_only': args.local_files_only}
    snapshot = baseline_contract.pinned_metadata_snapshot(local_files_only=args.local_files_only)
    # CPU merge avoids allocating a GPU on the shared server. Preserve bf16 and the full VLM architecture.
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='cpu', attn_implementation='sdpa', **source)
    adapted = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    if any('lora_' in name for name, _ in merged.named_parameters()):
        raise RuntimeError('LoRA modules remain after merge.')
    # A local pinned snapshot also prevents old AutoProcessor versions from probing resolve/main.
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    generation = GenerationConfig.from_pretrained(str(snapshot), local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True, max_shard_size='4GB')
    processor.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    generation.save_pretrained(str(args.output))
    base_metadata_hashes = copy_base_metadata(snapshot, args.output)
    # Reload processing metadata only, not model weights, to check the multimodal export is complete.
    restored_processor = AutoProcessor.from_pretrained(str(args.output), local_files_only=True)
    restored_tokenizer = AutoTokenizer.from_pretrained(str(args.output), local_files_only=True)
    if (not hasattr(restored_processor, 'image_processor')
            or not hasattr(restored_processor, 'video_processor')
            or restored_tokenizer.chat_template != tokenizer.chat_template):
        raise RuntimeError('Merged export lacks the pinned multimodal processor or chat template.')
    report = {
        'status': 'merge_completed', 'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION,
        'protocol': PROTOCOL, 'precision': 'bfloat16', 'baseline_contract': contract,
        'training_manifest_sha256': sha256(manifest_path), 'adapter_file_sha256': run['adapter_file_sha256'],
        'file_sha256': file_hashes(args.output), 'versions': versions,
        'base_metadata_file_sha256': base_metadata_hashes,
        'processor_exported': True, 'merge_device': 'cpu',
    }
    (args.output / 'merge_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Merged full VLM exported to {args.output}. Use this directory as the baseline --model_path.')


if __name__ == '__main__':
    main()
