#!/usr/bin/env python3
"""Theory inference/judge through the unchanged team's vLLM baseline functions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'03_기타'/'도구'))
from baseline_contract import (MODEL_REPO, PROTOCOL, assert_baseline, as_sample,
                               baseline_module, check_model_source, prompt_messages)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_unique(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty input or duplicate IDs.')
    return rows


def read_questions(path):
    rows = read_unique(path)
    for row in rows:
        if any(k in row for k in ('answer_text', 'answer_letter', 'spec', 'explanation', 'family_id')):
            raise ValueError('Evaluation questions contain answer/audit fields.')
        if row['messages'] != prompt_messages(row):
            raise ValueError('Question prompt differs from the frozen baseline; regenerate v2.2 data.')
    return rows


def evaluation_versions():
    versions = {name: importlib.metadata.version(name) for name in ['vllm', 'torch', 'transformers']}
    if versions['vllm'] != assert_baseline()['server']['vllm']:
        raise ValueError('Use the separate baseline vLLM 0.27.1 evaluation environment.')
    if Version(versions['transformers']) < Version('5.5.3') or Version(versions['torch']).base_version != '2.13.0':
        raise ValueError('Evaluation environment does not satisfy baseline vLLM dependencies.')
    return versions


def infer_record(row, api_base, served_model_name='qwen3-vl-4b'):
    baseline = baseline_module('run_mmmu_eval')
    content, options = baseline.build_content(as_sample(row), 'mmmu_pro_cot')
    args = SimpleNamespace(api_base=api_base, served_model_name=served_model_name,
                           max_new_tokens=baseline.DEFAULT_MAX_NEW_TOKENS, request_timeout=3600)
    text, reason, usage = baseline.call_model(content, args)
    return {'id': row['id'], **as_sample(row), 'options': options, 'raw_output': text,
            'finish_reason': reason, 'usage': usage, 'error': None,
            'response': text, 'output_tokens': usage.get('completion_tokens'),
            'length_limited': reason == 'length'}


def judge_record(row, api_base, model='judge'):
    return baseline_module('judge_utils').judge_row(row, api_base, model)


def write_predictions(rows, output, worker, concurrency):
    with Path(output).open('x', encoding='utf-8') as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker, row) for row in rows]
            for future in as_completed(futures):
                handle.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                handle.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['infer', 'judge'], default='infer')
    p.add_argument('--model', default=MODEL_REPO)
    p.add_argument('--revision')
    p.add_argument('--questions', type=Path)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api-base', required=True)
    p.add_argument('--server-manifest', type=Path, required=True,
                   help='Written by run_eval.py for the server it owns.')
    args = p.parse_args()
    contract = assert_baseline()
    server = json.loads(args.server_manifest.read_text())
    if server.get('protocol') != PROTOCOL or server.get('api_base') != args.api_base or server.get('stage') != args.stage:
        raise ValueError('Server manifest does not match this evaluation stage.')
    if server.get('baseline_contract') != contract:
        raise ValueError('Server protocol differs from pinned baseline.')
    versions = evaluation_versions()
    if args.output.exists():
        raise ValueError('Use a new output directory; old and new protocols must not be mixed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'infer':
        if not args.questions:
            p.error('--questions is required for infer')
        source = check_model_source(args.model, args.revision)
        if source != server['model_source']:
            raise ValueError('Model provenance differs from the owned server.')
        rows = read_questions(args.questions)
        conditions = {'protocol': PROTOCOL, 'baseline_contract': contract,
                      'base_model': source['base_model'], 'base_revision': source['base_revision'],
                      'versions': versions, 'gpu': server['gpu']}
        metadata = {'status': 'incomplete', 'protocol': PROTOCOL, 'question_sha256': sha256(args.questions),
                    'model_source': source, 'comparison_conditions': conditions, 'answer_key_read': False}
        worker = lambda row: infer_record(row, args.api_base)
    else:
        if not args.predictions:
            p.error('--predictions is required for judge')
        source_meta = json.loads(args.predictions.with_suffix('.manifest.json').read_text())
        if source_meta.get('status') != 'complete' or source_meta.get('protocol') != PROTOCOL:
            raise ValueError('Incomplete or legacy predictions.')
        predictions = read_unique(args.predictions)
        rows = [r for r in predictions if baseline_module('parsing').rule_parse(r) is None]
        metadata = {'status': 'incomplete', 'protocol': PROTOCOL,
                    'prediction_sha256': sha256(args.predictions), 'judge': contract['judge'], 'versions': versions}
        worker = lambda row: judge_record(row, args.api_base)
    manifest = args.output.with_suffix('.manifest.json')
    manifest.write_text(json.dumps(metadata, indent=2)+'\n')
    concurrency = contract['server']['concurrency'] if args.stage == 'infer' else contract['judge']['concurrency']
    write_predictions(rows, args.output, worker, concurrency)
    metadata.update(status='complete', records=len(rows), output_sha256=sha256(args.output))
    manifest.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'stage': args.stage, 'records': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
