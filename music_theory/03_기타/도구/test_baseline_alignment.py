"""Protocol parity and HTTP infer/judge/scoring checks, without model weights."""
import contextlib
import http.server
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

from _layout import PROJECT_ROOT, Layout
from baseline_contract import (PROTOCOL, MODEL_REPO, MODEL_REVISION, assert_baseline,
                               as_sample, baseline_module, prompt_messages, sft_messages,
                               check_model_source)
sys.path.insert(0, str(PROJECT_ROOT/'02_평가용'))
import predict
import score
import run_eval


class Handler(http.server.BaseHTTPRequestHandler):
    payloads = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.payloads.append((self.path, body))
        if self.path == '/tokenize':
            result = {'tokens': [1, 2, 3]}
        else:
            judge = body['model'] == 'judge'
            text = str(body['messages'])
            if judge:
                answer = 'B'
            elif 'fixture-fallback' in text:
                answer = 'The selected option is not explicitly formatted.'
            else:
                answer = 'Synthetic reasoning\nAnswer: B'
            result = {'choices': [{'message': {'content': answer},
                                  'finish_reason': 'length' if 'fixture-length' in text else 'stop'}],
                      'usage': {'completion_tokens': 15}}
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def fixture(name='fixture-rule', format_='mcq'):
    row = {'id': name, 'question': name, 'format': format_, 'task': 'fixture', 'category': 'fixture'}
    if format_ == 'mcq':
        row['options'] = dict(zip('ABCD', ['one', 'two', 'three', 'four']))
    row['messages'] = prompt_messages(row)
    return row


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.api = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_all_data_use_exact_baseline_prompts_and_final_answers(self):
        layout = Layout()
        for split in ['train', 'dev', 'test']:
            records = predict.read_unique(layout.records(split))
            for row in records:
                self.assertEqual(row['messages'], sft_messages(row))
                answer = row['answer_letter'] if row['format'] == 'mcq' else row['answer_text']
                self.assertEqual(row['messages'][-1]['content'].splitlines()[-1], 'Answer: '+answer)
                self.assertEqual(row['messages'][-1]['content'].count('Answer:'), 1)
        for split in ['dev', 'test']:
            self.assertEqual(len(predict.read_questions(layout.evaluation/f'{split}_questions.jsonl')), 300)

    def test_mcq_and_open_prompt_and_payload_parity(self):
        baseline = baseline_module('run_mmmu_eval')
        for kind in ['mcq', 'short_answer']:
            row = fixture(format_=kind)
            content, _ = baseline.build_content(as_sample(row), 'mmmu_pro_cot')
            self.assertEqual(row['messages'][0]['content'], content[0]['text'])
            result = predict.infer_record(row, self.api)
            path, payload = Handler.payloads[-1]
            self.assertEqual(path, '/v1/chat/completions')
            self.assertEqual(payload['messages'], [{'role': 'user', 'content': content}])
            for field, value in [('temperature', baseline.TEMPERATURE), ('top_p', baseline.TOP_P),
                                 ('top_k', baseline.TOP_K), ('seed', baseline.SEED),
                                 ('presence_penalty', baseline.PRESENCE_PENALTY),
                                 ('repetition_penalty', baseline.REPETITION_PENALTY),
                                 ('max_tokens', baseline.DEFAULT_MAX_NEW_TOKENS)]:
                self.assertEqual(payload[field], value)
            self.assertEqual(payload['mm_processor_kwargs'],
                             {'min_pixels': baseline.MIN_PIXELS, 'max_pixels': baseline.MAX_PIXELS})
            self.assertEqual(result['finish_reason'], 'stop')

    def test_parser_keeps_baseline_semantics(self):
        parser = baseline_module('parsing')
        for text in ['Answer: B\nReason', 'Reason\nAnswer: B', 'Answer: A\nAnswer: B',
                     'Final Answer: B', 'Answer: (B)', 'answer: b', 'No explicit answer']:
            self.assertEqual(score.extract(text), parser.parse_multiple_choice(text, 4))

    def test_real_http_judge_fallback_and_finish_reason(self):
        records = [predict.infer_record(fixture(name), self.api) for name in ['fixture-rule', 'fixture-fallback', 'fixture-length']]
        judged = predict.judge_record(records[1], self.api)
        keys = {r['id']: {'answer_letter': 'B', 'category': 'fixture', 'task': 'fixture', 'family_id': r['id']} for r in records}
        summary, rows = score.evaluate({r['id']: r for r in records}, keys, {judged['id']: judged})
        self.assertEqual(summary['accuracy'], 1)
        self.assertEqual(summary['length_limit_rate'], 1/3)
        self.assertEqual(rows[1]['parser_method'], 'judge')
        with self.assertRaisesRegex(ValueError, 'Missing baseline judge'):
            score.evaluate({r['id']: r for r in records}, keys)

    def test_cli_pipeline_and_paired_scoring(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            questions = root/'questions.jsonl'
            rows = [fixture('fixture-rule'), fixture('fixture-fallback')]
            questions.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            key = root/'key.jsonl'
            key.write_text(''.join(json.dumps({'id': r['id'], 'answer_letter': 'B', 'category': 'fixture',
                              'task': 'fixture', 'family_id': r['id']})+'\n' for r in rows))
            source = check_model_source(MODEL_REPO)
            server = {'protocol': PROTOCOL, 'stage': 'infer', 'api_base': self.api,
                      'baseline_contract': assert_baseline(), 'model_source': source, 'gpu': 'mock-gpu'}
            manifest = root/'server.json'
            versions = {'vllm': '0.27.1', 'torch': '2.13.0', 'transformers': '5.5.3'}
            for name in ['base', 'lora']:
                predictions = root/f'{name}.jsonl'
                judges = root/f'{name}_judges.jsonl'
                server['stage'] = 'infer'; manifest.write_text(json.dumps(server))
                argv = ['predict.py', '--questions', str(questions), '--output', str(predictions),
                        '--api-base', self.api, '--server-manifest', str(manifest)]
                with patch.object(sys, 'argv', argv), patch.object(predict, 'evaluation_versions', return_value=versions):
                    predict.main()
                server['stage'] = 'judge'; manifest.write_text(json.dumps(server))
                argv = ['predict.py', '--stage', 'judge', '--predictions', str(predictions), '--output', str(judges),
                        '--api-base', self.api, '--server-manifest', str(manifest)]
                with patch.object(sys, 'argv', argv), patch.object(predict, 'evaluation_versions', return_value=versions):
                    predict.main()
            argv = ['score.py', '--predictions', str(root/'lora.jsonl'), '--judge-results', str(root/'lora_judges.jsonl'),
                    '--baseline', str(root/'base.jsonl'), '--baseline-judge-results', str(root/'base_judges.jsonl'),
                    '--key', str(key), '--output', str(root/'summary.json')]
            # Different deployment paths are legitimate when common base/protocol match.
            meta_file = (root/'lora.jsonl').with_suffix('.manifest.json')
            meta = json.loads(meta_file.read_text()); meta['model_source']['model'] = '/fixture/merged'
            meta_file.write_text(json.dumps(meta))
            with patch.object(sys, 'argv', argv):score.main()
            self.assertEqual(json.loads((root/'summary.json').read_text())['paired']['difference'], 0)
            meta['comparison_conditions']['versions']['transformers'] = 'different'
            meta_file.write_text(json.dumps(meta))
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'Unequal'):
                score.main()

    def test_server_commands_and_port_guard(self):
        infer = run_eval.server_command(MODEL_REPO, MODEL_REVISION, 'infer', 18000)
        self.assertIn('40960', infer); self.assertIn('bfloat16', infer)
        judge = run_eval.server_command(assert_baseline()['judge']['model'], assert_baseline()['judge']['revision'], 'judge', 18001)
        self.assertIn('20480', judge)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            with self.assertRaisesRegex(ValueError, 'already in use'):
                run_eval.check_port(sock.getsockname()[1])

    def test_legacy_outputs_and_wrong_sources_rejected(self):
        with self.assertRaises(ValueError):check_model_source(MODEL_REPO, 'main')
        with self.assertRaises(ValueError):check_model_source('/fixture/adapter')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.jsonl'; path.write_text('{}\n')
            path.with_suffix('.manifest.json').write_text(json.dumps({'protocol': 'legacy', 'status': 'complete'}))
            with self.assertRaisesRegex(ValueError, 'Legacy'):score.validated_metadata(path)

    def test_evaluation_environment_dependencies(self):
        versions = {'vllm': '0.27.1', 'torch': '2.13.0+cu130', 'transformers': '5.5.3'}
        with patch.object(predict.importlib.metadata, 'version', side_effect=lambda name: versions[name]):
            self.assertEqual(predict.evaluation_versions(), versions)
            versions['transformers'] = '4.57.1'
            with self.assertRaisesRegex(ValueError, 'dependencies'):predict.evaluation_versions()

    def test_merged_source_integrity_and_processing_provenance(self):
        from baseline_contract import file_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = ['config.json','generation_config.json','preprocessor_config.json',
                        'video_preprocessor_config.json','tokenizer_config.json','tokenizer.json',
                        'vocab.json','merges.txt','chat_template.json']
            for name in metadata+['model.safetensors']:
                (root/name).write_text('synthetic metadata fixture')
            hashes = {name: file_sha256(root/name) for name in metadata+['model.safetensors']}
            report = {'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION, 'protocol': PROTOCOL,
                      'status': 'merge_completed', 'precision': 'bfloat16', 'processor_exported': True,
                      'baseline_contract': assert_baseline(), 'file_sha256': hashes,
                      'base_metadata_file_sha256': {name: hashes[name] for name in metadata}}
            (root/'merge_manifest.json').write_text(json.dumps(report))
            self.assertEqual(check_model_source(str(root))['base_revision'], MODEL_REVISION)
            (root/'preprocessor_config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):check_model_source(str(root))

    def test_cleanup_handles_exited_server_parent(self):
        process = Mock(); process.pid = 999999; process.poll.return_value = 0
        with patch.object(run_eval.os, 'killpg', side_effect=[None, ProcessLookupError]) as kill:
            run_eval.stop_owned_process(process)
        self.assertEqual(kill.call_args_list[0].args, (process.pid, run_eval.signal.SIGTERM))

    def test_readiness_rejects_foreign_server_identity(self):
        process = Mock(); process.pid = 999999; process.poll.return_value = None
        command = run_eval.server_command(MODEL_REPO, MODEL_REVISION, 'infer', 18039)
        response = Mock(); response.json.return_value = {'data': [{'id': 'somebody-elses-server'}]}
        baseline = baseline_module('run_mmmu_eval')
        with tempfile.TemporaryDirectory() as directory, patch.object(run_eval.subprocess, 'Popen', return_value=process), \
             patch.object(baseline, 'check_server'), patch.object(baseline, 'session') as session, \
             patch.object(run_eval, 'stop_owned_process') as stop:
            session.return_value.get.return_value = response
            with self.assertRaisesRegex(RuntimeError, 'another server'):
                with run_eval.owned_server(command, Path(directory)/'server.log', 'infer', 18039):
                    self.fail('Foreign server accepted')
            stop.assert_called_once_with(process)

    def test_body_errors_are_not_readiness_errors(self):
        process = Mock(); process.pid = 999999; process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch.object(run_eval.subprocess, 'Popen', return_value=process), \
             patch.object(baseline_module('run_mmmu_eval'), 'check_server'), patch.object(run_eval, 'stop_owned_process') as stop:
            with self.assertRaisesRegex(RuntimeError, 'body failed'):
                with run_eval.owned_server(['fixture'], Path(directory)/'server.log', 'infer', 18039):
                    raise RuntimeError('body failed')
            stop.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
