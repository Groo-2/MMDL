"""Pinned baseline protocol shared by generation, training, and evaluation."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from _layout import PROJECT_ROOT

MODEL_REPO = 'Qwen/Qwen3-VL-4B-Instruct'
MODEL_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
PROTOCOL = 'mmmu_pro_cot_answer_last_v1'
BASELINE_ROOT = Path(os.environ.get('MMDL_BASELINE_ROOT', PROJECT_ROOT.parent/'baseline')).resolve()
CONTRACT_PATH = PROJECT_ROOT/'03_기타'/'검증기록'/'baseline_contract.json'
METADATA_FILES = ['config.json','generation_config.json','preprocessor_config.json',
                  'video_preprocessor_config.json','processor_config.json','tokenizer_config.json',
                  'tokenizer.json','added_tokens.json','special_tokens_map.json','chat_template.jinja',
                  'chat_template.json','vocab.json','merges.txt']


def pinned_metadata_snapshot(cache_dir=None, local_files_only=False):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=METADATA_FILES,
                                  cache_dir=cache_dir, local_files_only=local_files_only))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_baseline():
    contract = json.loads(CONTRACT_PATH.read_text())
    if (contract['protocol'], contract['base_model'], contract['base_revision']) != (PROTOCOL, MODEL_REPO, MODEL_REVISION):
        raise ValueError('Unexpected baseline contract identity.')
    for relative, expected in contract['file_sha256'].items():
        source = BASELINE_ROOT/relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Baseline missing or changed: {source}. Use the pinned baseline alongside music_theory.')
    return contract


def baseline_module(name):
    assert_baseline()
    module_dir = BASELINE_ROOT/'mmmu_eval'
    sys.path.insert(0, str(module_dir)) if str(module_dir) not in sys.path else None
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != module_dir/f'{name}.py':
        raise ImportError(f'Conflicting module named {name}.')
    return module


def as_sample(row):
    options = row.get('options', {})
    if row.get('format') == 'mcq':
        if not isinstance(options, dict) or list(options) != list('ABCD'):
            raise ValueError('Expected ordered A/B/C/D options.')
        options = [options[k] for k in 'ABCD']
        kind = 'multiple-choice'
    else:
        options = []
        kind = 'open'
    return {'question': row['question'], 'question_type': kind, 'options': options}


def prompt_messages(row):
    text, image_order, _ = baseline_module('prompting').build_mmmu_pro_cot_text(as_sample(row))
    if image_order:
        raise ValueError('Music theory records must be text-only.')
    return [{'role': 'user', 'content': text}]


def sft_messages(row):
    answer = row['answer_letter'] if row['format'] == 'mcq' else row['answer_text']
    explanation = row['explanation'].strip()
    if not explanation or 'Answer:' in explanation or '\n' in str(answer):
        raise ValueError('Expected explanation followed by one final answer line.')
    return prompt_messages(row)+[{'role': 'assistant', 'content': explanation+'\nAnswer: '+str(answer)}]


def check_model_source(model, revision=None):
    """Only the frozen base or a provenance-checked merged checkpoint is accepted."""
    if model == MODEL_REPO:
        if revision not in (None, MODEL_REVISION):
            raise ValueError('Base revision differs from team baseline.')
        return {'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION,
                'protocol': PROTOCOL, 'model': model, 'revision': MODEL_REVISION}
    path = Path(model).resolve()
    manifest_file = path/'merge_manifest.json'
    if revision is not None or not manifest_file.is_file():
        raise ValueError('Local evaluation models must be complete merged exports with merge_manifest.json.')
    manifest = json.loads(manifest_file.read_text())
    if any(manifest.get(k) != v for k, v in {'base_model': MODEL_REPO,
           'base_revision': MODEL_REVISION, 'protocol': PROTOCOL, 'status': 'merge_completed',
           'precision': 'bfloat16', 'processor_exported': True}.items()):
        raise ValueError('Merged checkpoint provenance differs from baseline.')
    if manifest.get('baseline_contract') != assert_baseline():
        raise ValueError('Merged checkpoint uses a different baseline contract.')
    hashes = manifest.get('file_sha256', {})
    if not hashes or not any(p.endswith('.safetensors') for p in hashes):
        raise ValueError('Missing merged model checksums.')
    metadata = manifest.get('base_metadata_file_sha256', {})
    required = {'config.json','generation_config.json','preprocessor_config.json',
                'video_preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'}
    if (not required.issubset(metadata) or not any(name in metadata for name in ['chat_template.jinja','chat_template.json'])
        or any(hashes.get(name) != expected for name,expected in metadata.items())):
        raise ValueError('Merged export is missing the original pinned processing metadata.')
    for relative, expected in hashes.items():
        file = (path/relative).resolve()
        if not file.is_relative_to(path) or not file.is_file() or file_sha256(file) != expected:
            raise ValueError(f'Merged checkpoint changed: {relative}')
    return {'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION, 'protocol': PROTOCOL,
            'model': str(path), 'revision': None,
            'merge_manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest()}
