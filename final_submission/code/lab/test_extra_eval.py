"""MMMU test와 MMMU-Pro 평가 코드의 Mac(GPU 없음, 네트워크 없음) 검증 테스트.

  python -m unittest lab.test_extra_eval -v     (code/ 에서)

GPU/vLLM/데이터 내려받기가 필요한 부분은 여기서 확인할 수 없다(데이터는 lab/check_benchmarks.py, 모델 서버는 smoke 단계).
이 테스트가 보장하는 것: 벤치마크 정의(문항 수, 고정 revision, 기본값이 MMMU validation인 것), 프롬프트와 요청 형식,
추론 → 채점 전 과정(가짜 서버), 전체 정확도 정의(validation은 macro, 나머지는 micro), 결과 폴더 분리와 이어받기.
"""

import argparse
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lab"))
sys.path.insert(0, str(REPO / "mmmu_eval"))

import benchmarks as B  # noqa: E402
import prompting as PR  # noqa: E402
import run_mmmu_eval as R  # noqa: E402
from parsing import is_correct, rule_parse  # noqa: E402

PROMPTS_YAML_COT_VISION = (
    "Write out the multiple-choice question in the image and then solve it. The last line of your response should be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. "
    "Think step by step before answering."
)
PROMPTS_YAML_COT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
)


def tiny_image():
    from PIL import Image

    return Image.new("RGB", (8, 8), (200, 30, 30))


def pro_standard_sample(n_options=10, answer="C"):
    return {
        "id": "test_Music_1",
        "question": "What does <image 1> show?",
        "options": str([f"option {i}" for i in range(n_options)]),
        "answer": answer,
        "subject": "Music",
        **{f"image_{i}": None for i in range(1, 8)},
        "image_1": tiny_image(),
    }


def pro_vision_sample(n_options=10, answer="C"):
    return {"id": "test_Music_1", "options": str([f"option {i}" for i in range(n_options)]), "answer": answer,
            "subject": "Music", "image": tiny_image()}


class FakeDataset:
    """datasets.Dataset 중에서 평가 코드가 쓰는 것만 흉내 낸다: len, dataset[i], dataset["열"]."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]


class BenchmarkDefinitions(unittest.TestCase):
    def test_default_is_validation_and_frozen_values_are_unchanged(self):
        bench = B.BENCHMARKS[B.DEFAULT_BENCHMARK]
        self.assertEqual(bench.name, "mmmu_val")
        self.assertEqual((bench.repo, bench.revision, bench.split), (R.DATASET_REPO, R.DATASET_REVISION, R.SPLIT))
        self.assertEqual(bench.revision, "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
        self.assertEqual(bench.expected_total, 900)
        self.assertEqual(set(bench.counts.values()), {R.SAMPLES_PER_SUBJECT})
        self.assertEqual(bench.overall, "macro")
        self.assertIsNone(bench.results_subdir)             # 기존 결과 폴더 구조 그대로
        self.assertEqual(bench.prompt, "mmmu_pro_cot")

    def test_sizes_subjects_and_pins(self):
        self.assertEqual(B.BENCHMARKS["mmmu_test"].expected_total, 10500)
        for name in ("mmmu_pro_standard10", "mmmu_pro_standard4", "mmmu_pro_vision"):
            self.assertEqual(B.BENCHMARKS[name].expected_total, 1730)
        for name, bench in B.BENCHMARKS.items():
            self.assertEqual(set(bench.counts), set(R.SUBJECTS), name)         # 30과목, 이름이 평가 코드와 같다
            self.assertRegex(bench.revision, r"^[0-9a-f]{40}$", name)           # revision을 고정한다
            if name != "mmmu_val":
                self.assertEqual(bench.overall, "micro", name)                  # 과목별 문항 수가 달라서 전체 문항 평균
                self.assertEqual(bench.results_subdir, name)                    # 결과 폴더가 validation과 섞이지 않는다

    def test_prompts(self):
        self.assertEqual(B.BENCHMARKS["mmmu_pro_vision"].prompt, "mmmu_pro_cot_vision")
        self.assertEqual(B.BENCHMARKS["mmmu_pro_standard10"].prompt, "mmmu_pro_cot")
        self.assertEqual(B.BENCHMARKS["mmmu_test"].prompt, "mmmu_pro_cot")
        self.assertEqual(PR.MMMU_PRO_COT_VISION_INSTRUCTION, PROMPTS_YAML_COT_VISION)      # 공식 prompts.yaml(cot.vision) 원문
        self.assertEqual(PR.MMMU_PRO_COT_MC_INSTRUCTION, PROMPTS_YAML_COT_STANDARD)        # 공식 prompts.yaml(cot.standard) 원문
        for name in PR.PROMPT_VARIANTS:
            self.assertTrue(callable(PR.PROMPT_VARIANTS[name]))


class NormalizeAndContent(unittest.TestCase):
    def test_mmmu_rows_are_not_touched(self):
        row = {"id": "validation_Music_1", "question": "q", "options": "['a']", "question_type": "open", "answer": "x"}
        for name in ("mmmu_val", "mmmu_test"):
            self.assertIs(B.BENCHMARKS[name].normalize(row), row)

    def test_pro_standard_request(self):
        bench = B.BENCHMARKS["mmmu_pro_standard10"]
        sample = bench.normalize(pro_standard_sample())
        self.assertEqual(sample["question_type"], "multiple-choice")
        content, options = PR.build_content(sample, bench.prompt)
        self.assertEqual(len(options), 10)
        self.assertEqual(content[0]["type"], "text")
        text = content[0]["text"]
        self.assertIn("What does [image] show?", text)                         # <image 1> 표시가 [image]로 바뀐다
        self.assertIn("\nA. option 0\n", text)
        self.assertIn("\nJ. option 9\n", text)                                 # 10지선다는 J까지
        self.assertTrue(text.endswith(PROMPTS_YAML_COT_STANDARD))
        self.assertEqual([p["type"] for p in content], ["text", "image_url"])  # 텍스트 먼저, 이미지는 그 뒤

    def test_pro_standard_with_twelve_options(self):
        bench = B.BENCHMARKS["mmmu_pro_standard10"]                           # 실제 데이터에 선택지 12개인 문항이 1개 있다
        content, options = PR.build_content(bench.normalize(pro_standard_sample(12, "L")), bench.prompt)
        self.assertIn("\nL. option 11\n", content[0]["text"])
        self.assertEqual(len(options), 12)

    def test_pro_vision_request(self):
        bench = B.BENCHMARKS["mmmu_pro_vision"]
        sample = bench.normalize(pro_vision_sample())
        self.assertEqual(sample["question"], B.VISION_QUESTION_PLACEHOLDER)
        content, options = PR.build_content(sample, bench.prompt)
        self.assertEqual([p["type"] for p in content], ["text", "image_url"])
        self.assertEqual(content[0]["text"], PROMPTS_YAML_COT_VISION)           # 텍스트는 지시문뿐(질문은 스크린샷 안에 있다)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(options), 10)

    def test_scoring_uses_option_count(self):
        bench = B.BENCHMARKS["mmmu_pro_standard10"]
        row = bench.normalize({**pro_standard_sample(10, "J"), "raw_output": "step\nAnswer: J", "options": [f"o{i}" for i in range(10)]})
        self.assertEqual(rule_parse(row), "J")
        self.assertTrue(is_correct(row, "J"))
        four = {**row, "options": ["a", "b", "c", "d"], "raw_output": "Answer: E", "answer": "A"}   # 4개 중 E는 범위 밖
        self.assertIsNone(rule_parse(four))


class RunsAndQueue(unittest.TestCase):
    def test_resolve_runs_separates_results_by_benchmark(self):
        import eval_repeat as E

        def resolve(benchmark=None, preset=None):
            a = argparse.Namespace(tag="t", out_root="/tmp/r", preset=preset, subject=None, runs=2, run_start=0, seed0=42)
            if benchmark:
                a.benchmark = benchmark
            return E.resolve_runs(a)

        root, runs = resolve()                                         # 기본(validation)은 지금까지와 같은 경로
        self.assertEqual(str(root), "/tmp/r/t")
        root, runs = resolve("mmmu_val")
        self.assertEqual(str(root), "/tmp/r/t")
        root, runs = resolve("mmmu_test")
        self.assertEqual(str(root), "/tmp/r/t/mmmu_test")
        self.assertEqual([(k, s) for k, s, _ in runs], [(0, 42), (1, 43)])
        self.assertEqual(str(runs[0][2]), "/tmp/r/t/mmmu_test/run_00")
        root, _ = resolve("mmmu_pro_vision")
        self.assertEqual(str(root), "/tmp/r/t/mmmu_pro_vision")
        with self.assertRaises(SystemExit):                            # 프리셋(music4, full2)은 validation 전용
            resolve("mmmu_test", preset="full2")

    def test_eval_queue_paths_and_command(self):
        import eval_queue as Q

        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "a" / "mmmu_test" / "run_00"
            run.mkdir(parents=True)
            (run / "summary.json").write_text("{}")
            self.assertEqual(Q.run_status(d, "a", 2, "mmmu_test"), [(0, True), (1, False)])
            self.assertEqual(Q.run_status(d, "a", 2), [(0, False), (1, False)])          # validation 폴더는 따로
        cmd = Q.build_command("t", "m", None, 1, python="py", repo="/r", benchmark="mmmu_pro_vision")
        self.assertEqual(cmd[-2:], ["--benchmark", "mmmu_pro_vision"])
        self.assertNotIn("--benchmark", Q.build_command("t", "m", None, 1, python="py", repo="/r"))


def synthetic_rows(bench, subjects=None, correct_every=2):
    """채점 테스트용 가짜 predictions. 과목별 문항 수를 벤치마크 정의와 같게 만든다."""

    rows = []
    for subject, n in bench.counts.items():
        if subjects and subject not in subjects:
            continue
        for i in range(1, n + 1):
            good = i % correct_every == 0
            rows.append({
                "id": f"test_{subject}_{i}", "subject": subject, "question_type": "multiple-choice", "question": "q",
                "options": ["a", "b", "c", "d"], "answer": "A", "prompt_variant": bench.prompt,
                "raw_output": "Answer: A" if good else "Answer: B", "finish_reason": "stop",
                "usage": {"completion_tokens": 10, "prompt_tokens": 20}, "error": None,
                "t_start": 1.0, "t_end": 2.0, "elapsed_sec": 1.0})
    return rows


class ScoringDefinitions(unittest.TestCase):
    def score(self, name, rows, **kw):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            args = argparse.Namespace(output_dir=d, benchmark=name, all=True, subject=None, limit=None, **kw)
            with mock.patch("builtins.print"):
                R.cmd_score(args)
            return json.loads((Path(d) / "summary.json").read_text())

    def test_validation_overall_is_macro_and_unchanged(self):
        s = self.score("mmmu_val", synthetic_rows(B.BENCHMARKS["mmmu_val"]))
        self.assertEqual(s["samples"], 900)
        self.assertEqual(s["experiment"], "mmmu_val_mmmu_pro_cot")          # 기존 결과와 같은 이름
        self.assertEqual(s["overall_accuracy"], s["macro_accuracy"])
        self.assertTrue(s["overall_definition"].startswith("macro average"))
        self.assertEqual(s["dataset"], {"repo": R.DATASET_REPO, "revision": R.DATASET_REVISION, "split": "validation"})

    def test_pro_overall_is_micro_over_uneven_subjects(self):
        bench = B.BENCHMARKS["mmmu_pro_standard10"]
        s = self.score("mmmu_pro_standard10", synthetic_rows(bench))
        self.assertEqual(s["samples"], 1730)
        self.assertEqual(s["overall_accuracy"], s["micro_accuracy"])
        self.assertNotAlmostEqual(s["macro_accuracy"], s["micro_accuracy"], places=6)    # 과목별 문항 수가 달라서 둘이 다르다
        self.assertTrue(s["overall_definition"].startswith("micro average"))
        self.assertEqual(s["dataset"]["config"], "standard (10 options)")
        self.assertEqual(s["benchmark"], "mmmu_pro_standard10")

    def test_incomplete_run_is_rejected(self):
        for name in ("mmmu_val", "mmmu_pro_vision", "mmmu_test"):
            rows = synthetic_rows(B.BENCHMARKS[name])[:-1]                   # 한 문항 모자란다
            with self.assertRaises(RuntimeError, msg=name):
                self.score(name, rows)

    def test_test_split_total(self):
        s = self.score("mmmu_test", synthetic_rows(B.BENCHMARKS["mmmu_test"]))
        self.assertEqual(s["samples"], 10500)
        self.assertEqual(s["subjects"], 30)


class ProOriginSplit(unittest.TestCase):
    def test_split_by_id_prefix(self):
        import pro_origin_split as S

        rows = [{"id": "validation_Music_1", "correct": True}, {"id": "validation_Music_2", "correct": False},
                {"id": "test_Music_1", "correct": True}, {"id": "test_Music_2", "correct": True}]
        with tempfile.TemporaryDirectory() as d:
            for k in (0, 1):
                run = Path(d) / f"run_{k:02d}"
                run.mkdir()
                (run / "final_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertEqual(S.split_accuracy(Path(d) / "run_00"), {"validation": (2, 1), "test": (2, 2)})
            text = S.summarize(d)
            self.assertIn("validation 50.00%", text)
            self.assertIn("test 100.00%", text)
            self.assertIn("전체 75.00%", text)
            self.assertIn("끝난 run이 없다", S.summarize(Path(d) / "nothing"))


class InferEndToEnd(unittest.TestCase):
    """가짜 서버로 추론 → 채점 전 과정과 이어받기를 확인한다."""

    def args(self, name, out, **kw):
        base = dict(output_dir=out, benchmark=name, subject="Music", all=False, limit=None, data_root="unused", prompt=None,
                    max_new_tokens=32, concurrency=2, request_timeout=5, api_base="http://x/v1", served_model_name="m",
                    model_path="p", model_revision="r")
        base.update(kw)
        return argparse.Namespace(**base)

    def run_infer(self, name, out, rows, **kw):
        sent = []

        def fake_call(content, args):
            sent.append(content)
            return "reasoning\nAnswer: A", "stop", {"completion_tokens": 5, "prompt_tokens": 7}

        with mock.patch.object(R, "load_subject", return_value=FakeDataset(rows)), \
                mock.patch.object(R, "call_model", side_effect=fake_call), \
                mock.patch.object(R, "check_server"), mock.patch("builtins.print"):
            R.cmd_infer(self.args(name, out, **kw))
        return sent

    def test_vision_infer_then_score(self):
        rows = [{**pro_vision_sample(4, "A"), "id": f"test_Music_{i}"} for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_infer("mmmu_pro_vision", d, rows)
            self.assertEqual(len(sent), 3)
            self.assertEqual(sent[0][0]["text"], PROMPTS_YAML_COT_VISION)
            preds = [json.loads(l) for l in (Path(d) / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual({p["prompt_variant"] for p in preds}, {"mmmu_pro_cot_vision"})
            self.assertTrue(all(p["question_type"] == "multiple-choice" and p["question"] == B.VISION_QUESTION_PLACEHOLDER for p in preds))
            with mock.patch("builtins.print"):
                R.cmd_score(argparse.Namespace(output_dir=d, benchmark="mmmu_pro_vision", subject="Music", all=False, limit=None))
            summary = json.loads((Path(d) / "summary.json").read_text())
            self.assertEqual((summary["samples"], summary["correct"]), (3, 3))
            # 두 번째 실행은 끝난 문항을 건너뛴다(이어받기)
            self.assertEqual(len(self.run_infer("mmmu_pro_vision", d, rows)), 0)

    def test_standard_infer_with_limit(self):
        rows = [{**pro_standard_sample(10, "A"), "id": f"test_Music_{i}"} for i in range(1, 6)]
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_infer("mmmu_pro_standard10", d, rows, limit=2)
            self.assertEqual(len(sent), 2)                                   # 과목당 앞 2문항만(스모크용)
            self.assertIn("J. option 9", sent[0][0]["text"])

    def test_wrong_prompt_for_benchmark_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(R, "check_server"):
            with self.assertRaises(RuntimeError):
                R.cmd_infer(self.args("mmmu_pro_vision", d, prompt="mmmu_pro_cot"))
            with self.assertRaises(RuntimeError):
                R.cmd_infer(self.args("mmmu_pro_standard10", d, prompt="mmmu_pro_cot_vision"))

    def test_validation_default_still_uses_mmmu_pro_cot(self):
        rows = [{"id": "validation_Music_1", "question": "q <image 1>", "options": "['a', 'b']", "question_type": "multiple-choice",
                 "answer": "A", "image_1": tiny_image(), **{f"image_{i}": None for i in range(2, 8)}}]
        with tempfile.TemporaryDirectory() as d:
            self.run_infer(None, d, rows)           # --benchmark 없이(기존 호출 방식)
            pred = json.loads((Path(d) / "predictions.jsonl").read_text().splitlines()[0])
            self.assertEqual(pred["prompt_variant"], "mmmu_pro_cot")
            self.assertEqual(pred["question_type"], "multiple-choice")


if __name__ == "__main__":
    unittest.main()
