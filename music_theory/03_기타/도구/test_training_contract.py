"""Training/merge regression tests using metadata fixtures, never model weights or CUDA."""
import contextlib
import copy
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from _layout import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / '01_학습용'))
import merge_lora
import train_lora
from baseline_contract import MODEL_REPO, MODEL_REVISION, PROTOCOL, assert_baseline
from data_io import load_sft, tokenize_response_only, validate_messages


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join('<' + m['role'] + '>' + m['content'] + '</turn>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode())


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='theory-training-contract.')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.messages = [{'role': 'user', 'content': '<synthetic instruction>'},
                         {'role': 'assistant', 'content': '<synthetic explanation>\nAnswer: <placeholder>'}]
        self.versions = {'transformers': '4.57.1', 'peft': '0.17.1', 'torch': 'fixture',
                         'accelerate': 'fixture', 'safetensors': 'fixture', 'Pillow': 'fixture'}

    def version_lookup(self, name):
        if name not in self.versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return self.versions[name]

    def write_release(self):
        training, audit = self.root / 'training', self.root / 'audit'
        training.mkdir(); audit.mkdir()
        paths = [training / 'train.jsonl', training / 'dev_loss.jsonl']
        for path in paths:
            path.write_text(json.dumps({'messages': self.messages}) + '\n')
        manifest = {'protocol': PROTOCOL, 'file_sha256': {
            p.relative_to(self.root).as_posix(): train_lora.sha256(p) for p in paths}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        for name in ['validation.json', 'public_eval_screening.json']:
            (audit / name).write_text('{"status":"PASS"}')
        report = {'status': 'PASS', 'protocol': PROTOCOL, 'model': MODEL_REPO,
                  'requested_revision': MODEL_REVISION, 'resolved_commit': MODEL_REVISION,
                  'input_file_sha256': {p.name: train_lora.sha256(p) for p in paths}}
        (audit / 'tokenizer_check.json').write_text(json.dumps(report))
        return report

    def write_adapter(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        names = ['model.language_model.layers.0.self_attn.q_proj']
        config = {'base_model_name_or_path': MODEL_REPO, 'revision': MODEL_REVISION,
                  'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none', 'r': 16,
                  'target_modules': train_lora.target_pattern(names)}
        (adapter / 'adapter_config.json').write_text(json.dumps(config))
        # Deliberately not a real safetensors file: tests only validate metadata and never load tensors.
        (adapter / 'adapter_model.safetensors').write_bytes(b'metadata fixture only')
        run = {'base_model': MODEL_REPO, 'base_revision': MODEL_REVISION, 'protocol': PROTOCOL,
               'status': 'training_completed', 'precision': 'bfloat16',
               'baseline_contract': assert_baseline(), 'versions': copy.deepcopy(self.versions),
               'targeted_modules': names, 'settings': {'rank': 16},
               'adapter_file_sha256': train_lora.file_hashes(adapter)}
        (self.root / 'run_manifest.json').write_text(json.dumps(run))
        return adapter, run, config

    def save_run(self, run):
        (self.root / 'run_manifest.json').write_text(json.dumps(run))

    def test_training_environment_isolated(self):
        checked = train_lora.validate_training_environment(self.version_lookup)
        self.assertEqual(checked['transformers'], '4.57.1')
        self.versions['vllm'] = '0.27.1'
        with self.assertRaisesRegex(ValueError, 'separate'):
            train_lora.validate_training_environment(self.version_lookup)

    def test_training_environment_rejects_wrong_versions(self):
        for package, wrong in [('transformers', '5.5.3'), ('peft', '0.18.0')]:
            with self.subTest(package=package):
                previous = self.versions[package]
                self.versions[package] = wrong
                with self.assertRaises(ValueError):
                    train_lora.validate_training_environment(self.version_lookup)
                self.versions[package] = previous

    def test_answer_must_follow_explanation_once(self):
        validate_messages(self.messages)
        for target in ['Answer: <placeholder>\nExplanation: <reason>', 'Answer: <placeholder>',
                       '<reason>\nAnswer: A\nAnswer: B', '<reason>\nAnswer: ']:
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_messages([self.messages[0], {'role': 'assistant', 'content': target}])
        with self.assertRaises(ValueError):
            validate_messages([{'role': 'system', 'content': '<custom system>'}] + self.messages)

    def test_training_export_rejects_audit_fields(self):
        source = self.root / 'train.jsonl'
        source.write_text(json.dumps({'messages': self.messages, 'answer_letter': '<private>'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'messages exports'):
            load_sft(source)

    def test_assistant_only_mask_and_no_silent_truncation(self):
        tokenizer = FakeTokenizer()
        encoded = tokenize_response_only(tokenizer, self.messages, 1024)
        prefix = tokenizer.apply_chat_template(self.messages[:-1], True, True)
        self.assertEqual(encoded['labels'][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(bytes(encoded['labels'][len(prefix):]).decode(),
                         self.messages[-1]['content'] + '</turn>')
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            tokenize_response_only(tokenizer, self.messages, 8)

    def test_prefix_mismatch_is_rejected(self):
        class BrokenTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                tokens = super().apply_chat_template(messages, tokenize, add_generation_prompt)
                return tokens if add_generation_prompt else [0] + tokens
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            tokenize_response_only(BrokenTokenizer(), self.messages, 1024)

    def test_release_binds_tokenizer_commit_and_inputs(self):
        report = self.write_release()
        _, train, dev = train_lora.validate_release(self.root)
        self.assertEqual((len(train), len(dev)), (1, 1))
        report_path = self.root / 'audit/tokenizer_check.json'
        for field, wrong in [('protocol', 'legacy'), ('model', 'other/model'),
                             ('requested_revision', 'main'), ('resolved_commit', 'wrong')]:
            changed = dict(report, **{field: wrong})
            report_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                train_lora.validate_release(self.root)
        report['input_file_sha256']['train.jsonl'] = 'stale'
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'current training'):
            train_lora.validate_release(self.root)

    def test_release_rejects_changed_data_and_legacy_protocol(self):
        self.write_release()
        path = self.root / 'training/train.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            train_lora.validate_release(self.root)
        manifest_path = self.root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['protocol'] = 'legacy'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'protocol'):
            train_lora.validate_release(self.root)

    def test_training_dry_run_does_not_read_final_test_or_load_libraries(self):
        self.write_release()
        evaluation = self.root / 'evaluation'
        evaluation.mkdir()
        (evaluation / 'test_answer_key.jsonl').write_bytes(b'\xff not readable as JSON')
        output = self.root / 'new_training_run'
        argv = ['train_lora.py', '--release', str(self.root), '--output', str(output), '--dry-run']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(train_lora, 'validate_training_environment', side_effect=AssertionError('heavy environment reached')):
            train_lora.main()
        self.assertFalse(output.exists())

    def test_projection_guard_rejects_vision_and_duplicate_targets(self):
        self.assertIn('q_proj', train_lora.target_pattern(['model.language_model.layers.0.self_attn.q_proj']))
        for names in [[], ['visual.blocks.0.attn.q_proj'],
                      ['model.language_model.layers.0.self_attn.q_proj'] * 2]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                train_lora.target_pattern(names)

    def test_merge_accepts_recorded_adapter_and_rejects_tampering(self):
        adapter, run, _ = self.write_adapter()
        verified, _ = merge_lora.validate_adapter_metadata(adapter)
        self.assertEqual(verified['base_revision'], MODEL_REVISION)
        (adapter / 'adapter_model.safetensors').write_bytes(b'changed after training')
        with self.assertRaisesRegex(ValueError, 'contents differ'):
            merge_lora.validate_adapter_metadata(adapter)

    def test_merge_rejects_wrong_training_provenance(self):
        adapter, run, _ = self.write_adapter()
        for field, wrong in [('base_revision', 'main'), ('protocol', 'legacy'),
                             ('status', 'incomplete'), ('precision', 'float16')]:
            changed = dict(run, **{field: wrong})
            self.save_run(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_lora.validate_adapter_metadata(adapter)

    def test_merge_rejects_rehashed_wrong_adapter_config(self):
        adapter, run, config = self.write_adapter()
        config['revision'] = 'main'
        (adapter / 'adapter_config.json').write_text(json.dumps(config))
        run['adapter_file_sha256'] = train_lora.file_hashes(adapter)
        self.save_run(run)
        with self.assertRaisesRegex(ValueError, 'configuration mismatch: revision'):
            merge_lora.validate_adapter_metadata(adapter)

    def test_merge_dry_run_does_not_load_weights_or_environment(self):
        adapter, _, _ = self.write_adapter()
        output = self.root / 'merged'
        argv = ['merge_lora.py', '--adapter', str(adapter), '--output', str(output), '--dry-run']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(merge_lora, 'validate_training_environment', side_effect=AssertionError('heavy environment reached')):
            merge_lora.main()
        self.assertFalse(output.exists())

    def test_merge_preserves_original_metadata_bytes_and_weight_index(self):
        snapshot, output = self.root / 'snapshot', self.root / 'merged'
        snapshot.mkdir(); output.mkdir()
        names = set(merge_lora.REQUIRED_BASE_METADATA) | {'chat_template.json'}
        originals = {name: (name + ': pinned original\n').encode() for name in names}
        for name, content in originals.items():
            (snapshot / name).write_bytes(content)
            (output / name).write_bytes(b'rewritten by training Transformers')
        for generated in ['chat_template.jinja', 'special_tokens_map.json', 'processor_config.json']:
            (output / generated).write_bytes(b'generated-only override')
        weights = {'model-00001-of-00001.safetensors': b'merged weight fixture',
                   'model.safetensors.index.json': b'merged shard index fixture'}
        for name, content in weights.items():
            (output / name).write_bytes(content)
        hashes = merge_lora.copy_base_metadata(snapshot, output)
        self.assertEqual(set(hashes), names)
        for name, content in originals.items():
            self.assertEqual((output / name).read_bytes(), content)
            self.assertEqual(hashes[name], train_lora.sha256(snapshot / name))
        for generated in ['chat_template.jinja', 'special_tokens_map.json', 'processor_config.json']:
            self.assertFalse((output / generated).exists())
        for name, content in weights.items():
            self.assertEqual((output / name).read_bytes(), content)

    def test_merge_rejects_incomplete_metadata_before_touching_output(self):
        snapshot, output = self.root / 'snapshot', self.root / 'merged'
        snapshot.mkdir(); output.mkdir()
        marker = output / 'config.json'
        marker.write_bytes(b'unchanged when source is incomplete')
        with self.assertRaisesRegex(ValueError, 'missing required metadata'):
            merge_lora.copy_base_metadata(snapshot, output)
        self.assertEqual(marker.read_bytes(), b'unchanged when source is incomplete')


if __name__ == '__main__':
    unittest.main()
