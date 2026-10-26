#!/usr/bin/env python3
"""Run owned vLLM infer/judge servers with the frozen baseline protocol."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from predict import (MODEL_REPO, PROTOCOL, ROOT, assert_baseline, baseline_module,
                     check_model_source, evaluation_versions, read_questions)


def check_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise ValueError(f'Port {port} is already in use; choose another --port.') from exc


def server_command(model, revision, stage, port):
    contract = assert_baseline()
    judge = stage == 'judge'
    command = ['vllm', 'serve', model]
    if revision:
        command += ['--revision', revision]
    command += ['--served-model-name', 'judge' if judge else 'qwen3-vl-4b', 'mmdl-owned-'+uuid.uuid4().hex,
                '--dtype', contract['server']['dtype'], '--seed', '42',
                '--max-model-len', str(contract['judge']['max_model_len'] if judge else contract['server']['max_model_len']),
                '--gpu-memory-utilization', str(contract['server']['gpu_memory_utilization']),
                '--host', '127.0.0.1', '--port', str(port)]
    return command


def stop_owned_process(process):
    """Terminate only the process group created by this invocation."""
    if process.poll() is None:
        if os.getpgid(process.pid) != process.pid or process.pid == os.getpgrp():
            raise RuntimeError('Refusing to signal an unowned process group.')
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    # Workers can outlive an exited server parent. Their session/group is still ours.
    for _ in range(10):
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@contextmanager
def owned_server(command, log, stage, port):
    check_port(port)
    env = dict(os.environ)
    if stage == 'infer':
        env['VLLM_USE_V2_MODEL_RUNNER'] = assert_baseline()['server']['VLLM_USE_V2_MODEL_RUNNER']
    else:
        env.pop('VLLM_USE_V2_MODEL_RUNNER', None)
    baseline = baseline_module('run_mmmu_eval')
    process = None
    try:
        with log.open('x') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            for _ in range(480):
                if process.poll() is not None:
                    raise RuntimeError(f'Owned vLLM server exited; inspect {log}.')
                try:
                    baseline.check_server(f'http://127.0.0.1:{port}/v1')
                except RuntimeError:
                    if process.poll() is not None:
                        raise RuntimeError(f'Owned vLLM server exited; inspect {log}.')
                    time.sleep(5)
                    continue
                if '--served-model-name' in command:
                    unique_alias = command[command.index('--served-model-name')+2]
                    response = baseline.session().get(f'http://127.0.0.1:{port}/v1/models', timeout=10)
                    response.raise_for_status()
                    if unique_alias not in {entry['id'] for entry in response.json()['data']}:
                        raise RuntimeError('Port belongs to another server; refusing to use it.')
                yield process
                return
            raise TimeoutError(f'Server did not become ready; inspect {log}.')
    finally:
        if process is not None:
            stop_owned_process(process)


def gpu_identity():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selected or ',' in selected or selected in ('-1', 'all'):
        raise ValueError('Expose exactly one assigned GPU using CUDA_VISIBLE_DEVICES.')
    result = subprocess.check_output(['nvidia-smi', '--id='+selected,
                                     '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    if not result or '\n' in result:
        raise ValueError('Expected exactly one assigned physical GPU.')
    active = subprocess.check_output(['nvidia-smi', '--id='+selected,
                                     '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise ValueError('Assigned GPU already has compute processes; do not interrupt them.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL_REPO)
    p.add_argument('--revision')
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=18000)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    contract = assert_baseline()
    source = check_model_source(args.model, args.revision)
    questions = read_questions(args.questions)
    keys = [json.loads(line) for line in args.key.read_text().splitlines() if line.strip()]
    if len(keys) != len(questions) or {r['id'] for r in keys} != {r['id'] for r in questions}:
        raise ValueError('Questions and answer key IDs differ.')
    if not 1024 <= args.port <= 65534:
        raise ValueError('Use a non-privileged port with room for the judge port.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be new or empty.')
    check_port(args.port)
    check_port(args.port+1)
    commands = {
        'infer': server_command(source['model'], source['revision'], 'infer', args.port),
        'judge': server_command(contract['judge']['model'], contract['judge']['revision'], 'judge', args.port+1)}
    if args.dry_run:
        print(json.dumps({'status': 'DRY RUN PASS', 'protocol': PROTOCOL,
                          'records': len(questions), 'server_commands': commands, 'gpu_used': False}, indent=2))
        return
    versions = evaluation_versions()
    gpu = gpu_identity()
    # Access to the gated judge must be arranged before starting this run.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir/'predictions.jsonl'
    judges = args.output_dir/'judge_results.jsonl'
    for stage, port in [('infer', args.port), ('judge', args.port+1)]:
        if gpu_identity() != gpu:
            raise ValueError('Assigned GPU identity changed between stages.')
        manifest = args.output_dir/f'{stage}_server.json'
        api = f'http://127.0.0.1:{port}/v1'
        with owned_server(commands[stage], args.output_dir/f'{stage}_server.log', stage, port) as process:
            manifest.write_text(json.dumps({'protocol': PROTOCOL, 'stage': stage, 'api_base': api,
                'model_source': source, 'baseline_contract': contract, 'versions': versions,
                'gpu': gpu, 'owned_pid': process.pid, 'command': commands[stage]}, indent=2)+'\n')
            command = [sys.executable, str(ROOT/'02_평가용'/'predict.py'), '--stage', stage,
                       '--api-base', api, '--server-manifest', str(manifest)]
            if stage == 'infer':
                command += ['--model', source['model'], '--questions', str(args.questions), '--output', str(predictions)]
                if source['revision']:
                    command += ['--revision', source['revision']]
            else:
                command += ['--predictions', str(predictions), '--output', str(judges)]
            subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(ROOT/'02_평가용'/'score.py'), '--predictions', str(predictions),
                    '--judge-results', str(judges), '--key', str(args.key),
                    '--output', str(args.output_dir/'summary.json')], check=True)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Evaluation interrupted; cleaning up owned servers.')
    signal.signal(signal.SIGTERM, interrupted)
    main()
