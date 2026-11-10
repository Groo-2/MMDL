"""lab/ 와 music_ft/ 의 Mac(GPU 없음) 검증 테스트.

  python -m unittest lab.test_lab -v     (저장소 루트에서)

GPU/vLLM이 필요한 부분(서버 기동, 실제 학습)은 여기서 확인할 수 없다. 이 테스트가 보장하는 것:
판정 로직, 선별 로직, 프롬프트/요청 형식, 채점·요약, 안전장치(8번 확장), 파이프라인 단계 순서와 실패 시 triage 호출.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lab"))
sys.path.insert(0, str(REPO / "music_ft"))

import env_triage as T  # noqa: E402
import preflight as P  # noqa: E402

SMI = """
| NVIDIA-SMI 575.57.08   Driver Version: 575.57.08   CUDA Version: 12.9     |
"""


class TriageLogic(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        info = T.parse_nvidia_smi(SMI)
        self.assertEqual(info["driver"], "575.57.08")
        self.assertEqual(info["cuda"], (12, 9))
        self.assertEqual(T.parse_nvidia_smi("no gpu"), {"driver": None, "cuda": None})

    def test_driver_vs_torch(self):
        self.assertEqual(T.driver_vs_torch((12, 9), "12.9")[0], T.OK)
        self.assertEqual(T.driver_vs_torch((13, 0), "12.9")[0], T.OK)
        # 같은 12.x 안에서 드라이버가 더 낮으면(예: 이 실습실의 12.6 + cu129) 마이너 버전 호환으로 실행되므로 경고
        level, msg = T.driver_vs_torch((12, 6), "12.9")
        self.assertEqual(level, T.WARN)
        self.assertIn("마이너 버전 호환", msg)
        self.assertEqual(T.driver_vs_torch((12, 4), "12.9")[0], T.WARN)
        self.assertEqual(T.driver_vs_torch((12, 9), "13.0")[0], T.FAIL)    # 메이저가 다르면 실패
        self.assertEqual(T.driver_vs_torch((11, 8), "12.9")[0], T.FAIL)
        self.assertEqual(T.driver_vs_torch(None, "12.9")[0], T.FAIL)
        self.assertEqual(T.driver_vs_torch((12, 9), None)[0], T.FAIL)       # CPU 빌드 torch

    def test_nvidia_wheel_mix_is_detected(self):
        # Colab 사고 재현: cu12와 cu13 cuDNN이 같이 설치
        installed = {"nvidia-cudnn-cu12": "9.27.0.42", "nvidia-cudnn-cu13": "9.20.0.48", "torch": "2.13.0+cu130"}
        hard, soft = T.find_nvidia_wheel_mix(installed, "13.0")
        self.assertTrue(any("cudnn" in p and "cu12" in p and "cu13" in p for p in hard))      # 버전이 다르면 FAIL 대상
        self.assertEqual(soft, [])

    def test_same_version_cu12_and_cu13_is_only_a_warning(self):
        # 이 실습실 mmdl 환경: cu12와 cu13이 같은 버전(9.20.0.48)으로 함께 설치됨. conv3d 스모크는 통과했다
        installed = {"nvidia-cudnn-cu12": "9.20.0.48", "nvidia-cudnn-cu13": "9.20.0.48", "nvidia-nccl-cu12": "2.29.7",
                     "nvidia-nccl-cu13": "2.29.7", "torch": "2.13.0+cu129"}
        hard, soft = T.find_nvidia_wheel_mix(installed, "12.9")
        self.assertEqual(hard, [])
        self.assertEqual(len(soft), 2)
        self.assertTrue(all("같은 버전" in s for s in soft))

    def test_nvidia_wheel_clean_and_wrong_family(self):
        self.assertEqual(T.find_nvidia_wheel_mix({"nvidia-cudnn-cu12": "9.1", "nvidia-cublas-cu12": "12.9"}, "12.9"), ([], []))
        hard, _ = T.find_nvidia_wheel_mix({"nvidia-cudnn-cu13": "9.20"}, "12.9")   # torch는 12.9인데 cu13 wheel만 있음
        self.assertTrue(hard and "cu12 빌드가 없고" in hard[0])

    def test_check_versions(self):
        good = {"torch": "2.13.0+cu129", "vllm": "0.27.1+cu129", "trl": "1.16.0", "transformers": "5.20.2"}
        self.assertTrue(all(level == T.OK for level, _ in T.check_versions(good)))
        bad = dict(good, transformers="4.57.1", torch="2.9.0")
        levels = {msg.split()[0]: level for level, msg in T.check_versions(bad)}
        self.assertEqual(levels["transformers"], T.FAIL)
        self.assertEqual(levels["torch"], T.FAIL)
        missing = {k: v for k, v in good.items() if k != "vllm"}
        self.assertTrue(any(level == T.FAIL and "vllm 미설치" in msg for level, msg in T.check_versions(missing)))

    def test_compare_lock(self):
        lock = "torch==2.13.0+cu129\ntrl==1.16.0\nvllm==0.27.1+cu129\n"
        same = {"torch": "2.13.0+cu129", "trl": "1.16.0", "vllm": "0.27.1+cu129"}
        self.assertEqual(T.compare_lock(same, lock), [])
        changed = dict(same, trl="1.17.0")
        diffs = T.compare_lock(changed, lock)
        self.assertEqual(len(diffs), 1)
        self.assertIn("trl", diffs[0])
        self.assertTrue(T.compare_lock({"torch": "2.13.0+cu129"}, lock))   # trl/vllm이 사라짐

    def test_render_mentions_failure(self):
        text = T.render([(T.FAIL, "x", "나쁨"), (T.OK, "y", "")], after_failure=True)
        self.assertIn("코드를 고치기 전에", text)
        self.assertIn("FAIL 1개", text)

    def test_cli_without_gpu_reports_failure(self):
        # GPU가 없는 PC에서는 실패(종료 코드 1)와 이유가 나와야 한다. --allow-no-gpu면 스킵.
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, str(REPO / "lab" / "env_triage.py"), "--log-dir", d],
                                 capture_output=True, text=True)
            if "nvidia-smi가 없거나" in out.stdout or "GPU를 쓸 수 없다" in out.stdout:
                self.assertEqual(out.returncode, 1)
            self.assertTrue(list(Path(d).glob("env_triage_*.txt")))


class PreflightLogic(unittest.TestCase):
    def test_disk(self):
        self.assertEqual(P.check_disk(tempfile.gettempdir(), 1)[0], P.OK)
        self.assertEqual(P.check_disk(tempfile.gettempdir(), 10 ** 7)[0], P.FAIL)
        self.assertEqual(P.check_disk(Path(tempfile.gettempdir()) / "not" / "yet" / "created", 1)[0], P.OK)

    def test_port(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            busy = s.getsockname()[1]
            self.assertEqual(P.check_port_free(busy)[0], P.FAIL)
        self.assertEqual(P.check_port_free(busy)[0], P.OK)

    def test_gpu_memory(self):
        self.assertEqual(P.parse_gpu_memory("23500, 24564\n"), (23500 / 1024, 24564 / 1024))
        self.assertEqual(P.check_gpu_free("23500, 24564", 20)[0], P.OK)
        self.assertEqual(P.check_gpu_free("9000, 24564", 20)[0], P.FAIL)   # 다른 프로세스가 GPU를 쓰는 중
        self.assertEqual(P.check_gpu_free("garbage", 20)[0], P.FAIL)

    def test_gpu_free_for_vllm_uses_utilization_fraction(self):
        # 실습실 로그: 시작 시 여유 22.46GiB, vLLM 0.90이 요구한 21.59GiB(=23.99 x 0.9). 여유가 이보다 적으면 서버가 안 뜬다
        total = "24564"
        self.assertEqual(P.check_gpu_free(f"{int(22.46 * 1024)}, {total}", frac=0.90)[0], P.OK)
        level, msg = P.check_gpu_free(f"{int(21.0 * 1024)}, {total}", frac=0.90)      # 20GB 기준이면 통과했을 값
        self.assertEqual(level, P.FAIL)
        self.assertIn("Windows", msg)
        self.assertEqual(P.check_gpu_free(f"{int(21.0 * 1024)}, {total}", need_gb=20)[0], P.OK)

    def test_each_stage_uses_its_own_gpu_threshold(self):
        # 실제 단계(stage_checks)가 vLLM 단계에는 0.90 x 전체, 학습 단계에는 20GB를 쓰는지 가짜 nvidia-smi로 확인
        d = Path(tempfile.mkdtemp())
        smi = d / "nvidia-smi"
        smi.write_text(f'#!/bin/bash\necho "{int(21.0 * 1024)}, 24564"\n')     # 여유 21.0GiB: 20GB는 넘고 21.59GB(vLLM)는 못 넘음
        smi.chmod(0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = f"{d}:{old}"
        try:
            def gpu(stage):
                return next(r for r in P.stage_checks(stage, offline=True, data_dir=d) if "GPU 여유" in r[1])

            for stage in ("eval", "screen", "holdout"):
                self.assertEqual(gpu(stage)[0], P.FAIL, stage)
            self.assertEqual(gpu("train")[0], P.OK)
            self.assertEqual(gpu("vllm")[0], P.OK)          # vllm_check는 0.85 x 전체 = 20.4GiB만 요구 (여유 21.0GiB면 통과)
        finally:
            os.environ["PATH"] = old

    def test_files_and_token(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.jsonl").write_text("x")
            res = P.check_files([Path(d) / "a.jsonl", Path(d) / "missing.jsonl"])
            self.assertEqual([r[0] for r in res], [P.OK, P.FAIL])
        old = os.environ.pop("HF_TOKEN", None)
        try:
            self.assertEqual(P.check_hf_token(True)[0], P.FAIL)
            self.assertEqual(P.check_hf_token(False)[0], P.WARN)
            os.environ["HF_TOKEN"] = "hf_x"
            self.assertEqual(P.check_hf_token(True)[0], P.OK)
            self.assertEqual(P.check_llama_access(offline=True)[0], P.WARN)
        finally:
            os.environ.pop("HF_TOKEN", None)
            if old:
                os.environ["HF_TOKEN"] = old


class SelectLogic(unittest.TestCase):
    @staticmethod
    def fake_rows(n_per_cat):
        rows = []
        for cat, n in n_per_cat.items():
            rows += [{"id": f"{cat}_{i:04d}", "category": cat} for i in range(n)]
        return rows

    def test_pool_is_category_balanced_and_fills_shortage(self):
        from select_ssmr import build_pool

        rows = self.fake_rows({"Beat": 2000, "Chord": 100, "Interval": 2000, "Scale": 1000})
        pool = build_pool(rows, 1000)
        counts = {c: sum(1 for r in pool if r["category"] == c) for c in ("Beat", "Chord", "Interval", "Scale")}
        self.assertEqual(len(pool), 1000)
        self.assertEqual(counts["Chord"], 100)                 # 모자란 카테고리는 가진 만큼
        self.assertEqual(counts["Beat"], counts["Interval"])   # 나머지는 균등
        self.assertEqual(build_pool(rows, 1000), pool)          # 재현성

    def test_select_filters_degenerate_and_balances(self):
        from select_ssmr import select_subset

        screened = []
        for cat in ("Beat", "Chord", "Interval", "Scale"):
            for i in range(40):
                screened.append({"id": f"{cat}_{i}", "category": cat, "k": i % 9, "n": 8})   # k=0..8 고르게
        by_id = {s["id"]: s for s in screened}
        ids, rep = select_subset(screened, per_category=10)
        self.assertEqual(len(ids), 40)
        self.assertEqual(set(rep["chosen_per_category"].values()), {10})
        self.assertLess(abs(rep["mean_pass_rate_chosen"] - 0.5), 0.1)        # 0.5에 가까운 문항 우선
        self.assertEqual(select_subset(screened, per_category=10)[0], ids)   # 재현성
        # 요구량(35)이 적격 문항 수(카테고리당 31개)보다 많은 조건: 필터가 없으면 0/8, 8/8까지 끌려 들어온다
        ids, rep = select_subset(screened, per_category=35)
        self.assertEqual(len(ids), 31 * 4)
        self.assertTrue(all(1 <= by_id[i]["k"] <= 7 for i in ids))           # 0/8, 8/8은 어떤 경우에도 제외

    def test_select_fills_shortage_from_other_categories(self):
        from select_ssmr import select_subset

        screened = [{"id": f"Chord_{i}", "category": "Chord", "k": 8, "n": 8} for i in range(30)]    # Chord는 전부 8/8
        for cat in ("Beat", "Interval", "Scale"):
            screened += [{"id": f"{cat}_{i}", "category": cat, "k": 4, "n": 8} for i in range(30)]
        ids, rep = select_subset(screened, per_category=10)
        self.assertEqual(len(ids), 40)                       # 총수는 유지
        self.assertEqual(rep["chosen_per_category"]["Chord"], 0)
        self.assertTrue(rep["notes"])                        # 부족 사실을 알림

    def test_select_not_enough_total(self):
        from select_ssmr import select_subset

        screened = [{"id": f"Beat_{i}", "category": "Beat", "k": 4, "n": 8} for i in range(5)]
        ids, rep = select_subset(screened, per_category=10)
        self.assertEqual(len(ids), 5)
        self.assertTrue(any("부족" in n for n in rep["notes"]))


class StressLogic(unittest.TestCase):
    def test_pick_stress_examples(self):
        from train_grpo import pick_stress_examples

        def ex(i, n_img, text_len):
            return {"id": i, "images": ["x"] * n_img, "prompt": [{"content": [{"text": "a" * text_len}]}]}

        examples = [ex("a", 1, 900), ex("b", 5, 100), ex("c", 5, 300), ex("d", 1, 50)]
        self.assertEqual([e["id"] for e in pick_stress_examples(examples, 2)], ["c", "b"])   # 이미지 5장, 긴 텍스트 우선


class _StubHandler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _StubHandler.seen.append(body)
        n = body.get("n", 1)
        text = body["messages"][0]["content"][0]["text"]
        truncated = "TRUNCATE" in text
        choices = [{"message": {"content": "thinking..." if truncated else "reasoning\nAnswer: A"},
                    "finish_reason": "length" if truncated else "stop"} for _ in range(n)]
        data = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class HoldoutEndToEnd(unittest.TestCase):
    """가짜 vLLM 서버로 요청 형식 -> 채점 -> 요약까지 확인한다."""

    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "images").mkdir()
        rows = []
        for i, (cat, ans, n_opt_img) in enumerate([("Beat", "A", 0), ("Chord", "B", 0), ("Scale", "A", 4), ("Interval", "A", 0)]):
            paths = {"image_1": f"images/{i}_q.png"}
            Image.new("RGB", (201, 66), "white").save(d / paths["image_1"])
            options = ["x", "y", "z", "w"]
            if n_opt_img:
                options = [f"<image {k + 2}>" for k in range(4)]
                for k in range(4):
                    paths[f"image_{k + 2}"] = f"images/{i}_o{k + 2}.png"
                    Image.new("RGB", (100, 40), "white").save(d / paths[f"image_{k + 2}"])
            q = "TRUNCATE " if i == 3 else ""
            rows.append({"id": f"h{i}", "question": f"<image 1> {q}What?", "options": options, "question_type": "multiple-choice",
                         "answer": ans, "image_paths": paths, "category": cat})
        (d / "holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.dir, self.rows = d, rows
        self.server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        _StubHandler.seen.clear()

    def tearDown(self):
        self.server.shutdown()
        self.tmp.cleanup()

    def test_run_seed(self):
        import eval_holdout as H
        from ssmr_client import load_rows

        rows = load_rows(self.dir / "holdout.jsonl")
        s = H.run_seed(rows, self.dir, f"http://127.0.0.1:{self.port}/v1", "qwen3-vl-4b", 42, 8192, 4, self.dir / "out")
        # 정답: h0 A(맞음), h1 B(서버는 A라 틀림), h2 A(맞음), h3 잘림(오답)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertEqual(s["truncated"], 1)
        self.assertAlmostEqual(s["accuracy_among_finished"], 2 / 3)
        self.assertEqual(set(s["by_category"]), {"Beat", "Chord", "Scale", "Interval"})
        # 요청 형식 검증: 평가와 같은 필드
        req = next(b for b in _StubHandler.seen if len(b["messages"][0]["content"]) == 6)    # 이미지 5장짜리(텍스트 1 + 이미지 5)
        self.assertEqual(req["messages"][0]["content"][0]["type"], "text")
        self.assertTrue(all(c["type"] == "image_url" for c in req["messages"][0]["content"][1:]))
        self.assertEqual(req["seed"], 42)
        self.assertEqual(req["presence_penalty"], 1.5)
        self.assertEqual((req["temperature"], req["top_p"], req["top_k"]), (0.7, 0.8, 20))
        self.assertEqual(req["mm_processor_kwargs"], {"min_pixels": 1003520, "max_pixels": 4014080})
        self.assertEqual(req["max_tokens"], 8192)
        # 프롬프트 문구가 동결 프롬프트(mmmu_pro_cot)와 같다
        self.assertIn("Think step by step before answering.", req["messages"][0]["content"][0]["text"])
        # resume: 다시 실행하면 새 요청이 없다
        _StubHandler.seen.clear()
        H.run_seed(rows, self.dir, f"http://127.0.0.1:{self.port}/v1", "qwen3-vl-4b", 42, 8192, 4, self.dir / "out")
        self.assertEqual(_StubHandler.seen, [])

    def test_screen_rows(self):
        from select_ssmr import screen_rows
        from ssmr_client import load_rows

        rows = load_rows(self.dir / "holdout.jsonl")
        res = screen_rows(rows, self.dir, f"http://127.0.0.1:{self.port}/v1", "qwen3-vl-4b", 8, 2048, 4, self.dir / "screen.jsonl")
        by = {r["id"]: r for r in res}
        self.assertEqual((by["h0"]["k"], by["h0"]["n"]), (8, 8))     # 서버는 항상 A: 정답 A면 8/8
        self.assertEqual(by["h1"]["k"], 0)                            # 정답 B면 0/8
        self.assertEqual(by["h3"]["truncated"], 8)                    # 잘린 응답은 0점
        self.assertTrue(all(b["n"] == 8 and b["max_tokens"] == 2048 for b in _StubHandler.seen))
        self.assertTrue(all("presence_penalty" not in b for b in _StubHandler.seen))    # 학습 rollout과 같은 sampling


class EvalRepeatGuard(unittest.TestCase):
    def run_cli(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if k != "CONFIRM_EXTEND_TO_8"}
        e.update(env or {})
        return subprocess.run([sys.executable, str(REPO / "lab" / "eval_repeat.py"), *args],
                              capture_output=True, text=True, env=e)

    def test_resolve_runs(self):
        import argparse
        import eval_repeat as E

        def resolve(preset):
            a = argparse.Namespace(tag="t", out_root="/tmp/r", preset=preset, subject=None, runs=0, run_start=0, seed0=42)
            return E.resolve_runs(a)

        root, runs = resolve("music4")
        self.assertEqual([(k, s) for k, s, _ in runs], [(0, 42), (1, 43), (2, 44), (3, 45)])
        self.assertEqual(str(root), "/tmp/r/t/music")
        root, runs = resolve("full2")
        self.assertEqual([(k, s) for k, s, _ in runs], [(0, 42), (1, 43)])
        self.assertTrue(str(root).endswith("/t/full"))
        root, runs = resolve("music_extend")
        self.assertEqual([(k, s) for k, s, _ in runs], [(4, 46), (5, 47), (6, 48), (7, 49)])   # seed 46~49, 같은 폴더에 이어 붙임
        self.assertTrue(str(root).endswith("/t/music"))

    def test_extend_is_blocked_without_confirmation(self):
        out = self.run_cli("--tag", "t", "--model", "m", "--preset", "music_extend", "--dry-run")
        self.assertEqual(out.returncode, 3)
        self.assertIn("사용자 확인 없이 실행하지 않는다", out.stdout)
        self.assertNotIn("계획:", out.stdout)               # 아무 계획도 세우지 않고 중단

    def test_extend_needs_music4_even_with_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.run_cli("--tag", "t", "--model", "m", "--preset", "music_extend", "--out-root", d, "--dry-run",
                               env={"CONFIRM_EXTEND_TO_8": "yes"})
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("music4가 끝나지 않았다", out.stdout + out.stderr)

    def test_extend_allowed_after_music4_with_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            for k in range(4):
                run = Path(d) / "t" / "music" / f"run_{k:02d}"
                run.mkdir(parents=True)
                (run / "final_predictions.jsonl").write_text("")
            out = self.run_cli("--tag", "t", "--model", "m", "--preset", "music_extend", "--out-root", d, "--dry-run",
                               env={"CONFIRM_EXTEND_TO_8": "yes"})
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("seed [46, 47, 48, 49]", out.stdout)

    def test_wrong_confirmation_value_is_rejected(self):
        out = self.run_cli("--tag", "t", "--model", "m", "--preset", "music_extend", "--dry-run",
                           env={"CONFIRM_EXTEND_TO_8": "true"})      # 정확히 'yes'만 인정
        self.assertEqual(out.returncode, 3)


class PipelineScript(unittest.TestCase):
    def run_pipeline(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if k != "CONFIRM_EXTEND_TO_8"}
        e.update({"DRY_RUN": "1"})
        e.update(env or {})
        return subprocess.run(["bash", str(REPO / "lab" / "run_pipeline.sh"), *args], capture_output=True, text=True, env=e, cwd=REPO)

    def test_all_order_excludes_extend(self):
        out = self.run_pipeline("all")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        stages = [l.split("stage: ")[1].split()[0] for l in out.stdout.splitlines() if "stage:" in l]
        self.assertEqual(stages, ["data", "screen", "stress", "pilot", "baseline_music4", "holdout_base", "baseline_full2",
                                  "grpo", "merge", "holdout_new", "music4", "full2", "report"])
        self.assertNotIn("music_extend", out.stdout)
        self.assertNotIn("CONFIRM_EXTEND_TO_8=yes", out.stdout.split("report")[0])

    def test_vllm_check_stage(self):
        out = self.run_pipeline("vllm_check")
        self.assertEqual(out.returncode, 0, out.stdout)
        cmds = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
        self.assertTrue(cmds[0].startswith("+ python lab/preflight.py --stage vllm"))
        self.assertEqual(cmds[1], "+ python lab/check_vllm.py")

    def test_each_long_job_runs_preflight_first(self):
        out = self.run_pipeline("grpo", "baseline_full2", "holdout_new").stdout.splitlines()
        cmds = [l for l in out if l.startswith("+ ")]
        self.assertTrue(cmds[0].startswith("+ python lab/preflight.py --stage train"))
        self.assertIn("train_grpo.py", cmds[1])
        self.assertTrue(any("preflight.py --stage eval" in c for c in cmds))
        self.assertTrue(any("preflight.py --stage holdout" in c for c in cmds))

    def test_extend_stage_blocked(self):
        for stage in ("music_extend_base", "music_extend_new"):
            out = self.run_pipeline(stage)
            self.assertEqual(out.returncode, 3, out.stdout)
            self.assertIn("사용자 확인 없이 실행하지 않는다", out.stdout)
            self.assertNotIn("eval_repeat.py", out.stdout)
            self.assertNotIn("env_triage", out.stdout)       # 안전장치 대기는 환경 문제가 아니므로 triage를 부르지 않는다

    def test_extend_stage_allowed_with_confirmation(self):
        out = self.run_pipeline("music_extend_base", env={"CONFIRM_EXTEND_TO_8": "yes"})
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertIn("--preset music_extend", out.stdout)

    def test_failure_triggers_triage_first(self):
        out = self.run_pipeline("nonexistent_stage")
        self.assertEqual(out.returncode, 1)
        self.assertIn("코드를 고치기 전에 CUDA/패키지 버전부터 점검합니다", out.stdout)
        self.assertIn("env_triage.py --after-failure", out.stdout)


class SummarizeSubject(unittest.TestCase):
    def test_subject_filter(self):
        import summarize_runs as S

        with tempfile.TemporaryDirectory() as d:
            for k in range(2):
                run = Path(d) / f"run_{k:02d}"
                run.mkdir()
                rows = [{"id": f"m{i}", "subject": "Music", "correct": i < 1 + k, "finish_reason": "stop"} for i in range(4)]
                rows += [{"id": f"a{i}", "subject": "Art", "correct": True, "finish_reason": "stop"} for i in range(4)]
                (run / "final_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            s = S.summarize(d, subject="Music")
            self.assertAlmostEqual(s["overall_acc_mean_std"][0], 0.375)     # (1/4 + 2/4) / 2, Art는 제외
            self.assertAlmostEqual(S.summarize(d)["overall_acc_mean_std"][0], (0.625 + 0.75) / 2)


class SetupPlan(unittest.TestCase):
    """가짜 nvidia-smi로 드라이버 CUDA별 설치 경로 선택을 확인한다 (설치는 하지 않는다: DRY_RUN=1)."""

    def run_setup(self, cuda, with_smi=True):
        d = Path(tempfile.mkdtemp())
        if with_smi:
            smi = d / "nvidia-smi"
            smi.write_text('#!/bin/bash\nif [[ "$*" == *--query-gpu* ]]; then echo "NVIDIA GeForce RTX 4090, 24564 MiB"; '
                           'else echo "| NVIDIA-SMI 560.35.02   Driver Version: 560.94   CUDA Version: $FAKE_CUDA |"; fi\n')
            smi.chmod(0o755)
        env = {"PATH": f"{d}:/usr/bin:/bin", "DRY_RUN": "1", "FAKE_CUDA": cuda, "HOME": str(d)}
        return subprocess.run(["bash", str(REPO / "lab" / "setup_env.sh")], capture_output=True, text=True, env=env, cwd=REPO)

    def test_driver_12_6_uses_cu129_with_minor_compat_note(self):
        out = self.run_setup("12.6")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("설치 경로: cu129", out.stdout)
        self.assertIn("마이너 버전 호환", out.stdout)

    def test_driver_12_9_uses_cu129_without_note(self):
        out = self.run_setup("12.9")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("설치 경로: cu129", out.stdout)
        self.assertNotIn("마이너 버전 호환", out.stdout)

    def test_driver_13_uses_default_wheel(self):
        out = self.run_setup("13.0")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("설치 경로: cu130", out.stdout)

    def test_force_plan_cu129_on_driver_13_for_colab(self):
        d = Path(tempfile.mkdtemp())
        smi = d / "nvidia-smi"
        smi.write_text('#!/bin/bash\nif [[ "$*" == *--query-gpu* ]]; then echo "NVIDIA A100-SXM4-40GB, 40960 MiB"; '
                       'else echo "| NVIDIA-SMI 580.82   Driver Version: 580.82   CUDA Version: 13.0 |"; fi\n')
        smi.chmod(0o755)
        env = {"PATH": f"{d}:/usr/bin:/bin", "DRY_RUN": "1", "HOME": str(d), "FORCE_PLAN": "cu129"}
        out = subprocess.run(["bash", str(REPO / "lab" / "setup_env.sh")], capture_output=True, text=True, env=env, cwd=REPO)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("설치 경로: cu129", out.stdout)
        self.assertNotIn("마이너 버전 호환", out.stdout)            # 드라이버가 13이라 12.x 마이너 안내가 나오면 안 된다
        env["FORCE_PLAN"] = "cu128"
        bad = subprocess.run(["bash", str(REPO / "lab" / "setup_env.sh")], capture_output=True, text=True, env=env, cwd=REPO)
        self.assertEqual(bad.returncode, 2)

    def test_driver_below_12_is_rejected(self):
        out = self.run_setup("11.8")
        self.assertEqual(out.returncode, 2)
        self.assertIn("지원하지 않는다", out.stdout + out.stderr)

    def test_missing_nvidia_smi(self):
        import shutil

        if shutil.which("nvidia-smi", path="/usr/bin:/bin"):
            self.skipTest("이 PC는 /usr/bin에 nvidia-smi가 있다")
        out = self.run_setup("12.6", with_smi=False)
        self.assertEqual(out.returncode, 1)
        self.assertIn("nvidia-smi가 없다", out.stdout + out.stderr)


class ServerEnv(unittest.TestCase):
    def test_flashinfer_sampler_off_by_default_but_overridable(self):
        import eval_repeat as E

        old = os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        try:
            self.assertEqual(E.server_env()["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertEqual(E.server_env()["VLLM_USE_V2_MODEL_RUNNER"], "0")
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
            self.assertEqual(E.server_env()["VLLM_USE_FLASHINFER_SAMPLER"], "1")
        finally:
            os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
            if old is not None:
                os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = old



class GeneralConversion(unittest.TestCase):
    """DeepVision 변환 규칙과 주관식·가변 선택지 채점 (일반 데이터 RFT용). 채점이 틀리면 학습에 오답 응답이 섞이므로 규칙을 고정해 둔다."""

    def test_split_mcq(self):
        from build_general import split_mcq

        stem, opts = split_mcq("How many segments equal AO? \nA. 3 \nB. 4 \nC. 5 \nD. 6")
        self.assertEqual(stem, "How many segments equal AO?")
        self.assertEqual(opts, ["3", "4", "5", "6"])
        self.assertEqual(split_mcq("Q?\n(A) yes\n(B) no")[1], ["yes", "no"])               # 괄호 표기
        self.assertIsNone(split_mcq("Which is symmetric?\nΔABO\nΔCDO\nΔEFO"))             # 글자 없는 보기
        self.assertIsNone(split_mcq("Q?\nA. 1\nC. 3"))                                      # 글자가 연속이 아님
        self.assertIsNone(split_mcq("A. only options"))                                       # 질문 본문이 없음

    @staticmethod
    def _row(question, gt, eq=("__EMPTY__",), rate="3/8", images=1):
        return {"question": question, "reward_model": {"ground_truth": gt, "equivalent_answers": list(eq)},
                "mimo-pass_rate": rate, "domain-anno": json.dumps({"Level_1": "Geometry"}),
                "images": [{"bytes": b"x", "path": "a.png"}] * images}

    def test_convert_rules(self):
        from build_general import convert_row

        ok, why = convert_row(1, self._row("Q?\nA. 3\nB. 4", "B"))
        self.assertEqual((why, ok["question_type"], ok["answer"], ok["options"]), ("ok", "multiple-choice", "B", ["3", "4"]))
        self.assertTrue(ok["question"].startswith("<image 1> "))
        num, why = convert_row(2, self._row("Find the area.", "18.5", eq=("37/2", "__EMPTY__")))
        self.assertEqual((why, num["question_type"], num["answer"]), ("ok", "open", ["18.5", "37/2"]))
        self.assertEqual(convert_row(3, self._row("Which?\nΔAB\nΔCD", "C"))[1], "unlabeled_options")
        self.assertEqual(convert_row(4, self._row("Find the equation.", "(x-1)^2+y^2=1"))[1], "non_numeric_answer")
        self.assertEqual(convert_row(5, self._row("求面积 area?", "3"))[1], "cjk")
        self.assertEqual(convert_row(6, self._row("Q?\nA. 3\nB. 4", "B", images=2))[1], "image_count")
        self.assertEqual(convert_row(7, self._row("Q?\nA. 3\nB. 4", "B", rate="0/8"))[1], "pass_rate")
        self.assertEqual(convert_row(8, self._row("Q?\nA. 3\nB. 4", "B", rate="8/8"))[1], "pass_rate")
        self.assertEqual(convert_row(9, self._row("Q?\nA. 3\nB. 4", "E"))[1], "mc_answer_mismatch")

    def test_scoring_open_and_variable_options(self):
        from reward import accuracy_reward, score_completion

        self.assertEqual(score_completion("Answer: 3.18", ["3.18", "11/6"], "open"), 1.0)
        self.assertEqual(score_completion("so ... Answer: 3.1769", "3.18", "open"), 1.0)      # 소수 둘째 자리 반올림
        self.assertEqual(score_completion("Answer: 4", "3.18", "open"), 0.0)
        self.assertEqual(score_completion("no answer line", "3.18", "open"), 0.0)
        self.assertEqual(score_completion("Answer: E", "E", "multiple-choice", 6), 1.0)       # 보기 6개
        self.assertEqual(score_completion("Answer: E", "E", "multiple-choice", 4), 0.0)       # 4개면 E는 유효하지 않음
        self.assertEqual(accuracy_reward(["Answer: B"], ["B"]), [1.0])                         # 기본은 SSMR 4지선다
        self.assertEqual(accuracy_reward(["Answer: 7"], ["7"], question_type=["open"], num_options=[0]), [1.0])

    def test_end_to_end_with_fake_parquet(self):
        try:
            import pandas as pd
            import pyarrow  # noqa: F401
        except Exception as e:  # noqa: BLE001  (이 PC의 pandas/pyarrow/numpy 버전이 안 맞으면 건너뛴다)
            self.skipTest(f"pandas/pyarrow를 쓸 수 없다: {type(e).__name__}")
        from PIL import Image
        import io
        import build_general

        buf = io.BytesIO(); Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
        rows = []
        for i in range(30):
            r = self._row(f"Q{i}?\nA. 1\nB. 2", "A" if i % 2 else "B", rate="2/8") if i % 3 else self._row(f"Find {i}.", str(i), rate="2/8")
            r["images"] = [{"bytes": buf.getvalue(), "path": f"x{i}.png"}]
            r["data_source"] = "t"; r["ability"] = "math"; r["prompt"] = "p"; r["gpt5-mini-vision-anno"] = ""; r["q32-vision-anno"] = ""
            rows.append(r)
        d = Path(tempfile.mkdtemp())
        try:
            pd.DataFrame(rows).to_parquet(d / "t.parquet")
        except Exception as e:  # noqa: BLE001  (pyarrow/numpy 버전이 안 맞는 PC에서는 parquet을 만들 수 없다)
            self.skipTest(f"parquet을 쓸 수 없다: {type(e).__name__}")
        old = sys.argv
        sys.argv = ["build_general.py", "--parquet", str(d / "t.parquet"), "--out-dir", str(d / "out"), "--pool-size", "12"]
        try:
            build_general.main()
        finally:
            sys.argv = old
        out = [json.loads(l) for l in (d / "out" / "train.jsonl").read_text().splitlines()]
        self.assertEqual(len(out), 12)
        self.assertTrue(all((d / "out" / r["image_paths"]["image_1"]).exists() for r in out))
        self.assertEqual({r["question_type"] for r in out}, {"multiple-choice", "open"})


class DedupeLogic(unittest.TestCase):
    """MMMU 전 과목 중복 제거의 판정 로직 (HF 다운로드 없이 가짜 MMMU 문항으로)."""

    @staticmethod
    def _png(kind, size=160):
        import io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(img)
        if kind == "circle":
            d.ellipse((30, 30, 130, 130), fill="black")
        elif kind == "rect":
            d.rectangle((20, 60, 140, 100), fill="black")
        else:
            d.polygon([(80, 20), (20, 140), (140, 140)], fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), img

    def test_check_row(self):
        import io

        import dedupe as D

        circle, circle_img = self._png("circle")
        rect, _ = self._png("rect")
        tri, _ = self._png("tri")
        long_q = "A particle moves along a straight line with constant acceleration; find its displacement after four seconds."
        items = [
            {"id": "m1", "split": "validation", "question": "<image 1> Which shape is shown in the figure?", "options": ["a", "b"], "images": [circle]},
            {"id": "m2", "split": "test", "question": "<image 1> What is drawn?", "options": ["c", "d"], "images": [rect]},
            {"id": "m3", "split": "test", "question": long_q, "options": [], "images": []},
        ]
        ref = D.build_reference(items)
        self.assertEqual((ref["items"], len(ref["hash"])), (3, 2))

        d = Path(tempfile.mkdtemp())
        (d / "same.png").write_bytes(circle)
        buf = io.BytesIO()
        circle_img.resize((150, 150)).save(buf, format="JPEG", quality=60)
        (d / "variant.jpg").write_bytes(buf.getvalue())
        (d / "other.png").write_bytes(tri)

        def row(question, img, options=()):
            return {"id": "r", "question": question, "options": list(options), "image_paths": {"image_1": img}}

        self.assertTrue(D.check_row(row("<image 1> zzz", "same.png"), d, ref)[0].startswith("sha1"))
        self.assertTrue(D.check_row(row("<image 1> zzz", "variant.jpg"), d, ref)[0].startswith("phash"))      # 재인코딩·크기 변경
        self.assertIsNone(D.check_row(row("<image 1> unrelated question about triangles", "other.png"), d, ref)[0])
        self.assertEqual(D.check_row(row("<image 1> " + long_q, "other.png"), d, ref)[0], "text_exact")        # 이미지는 달라도 질문이 같음
        near = long_q.replace("four seconds", "4 seconds")
        self.assertEqual(D.check_row(row("<image 1> " + near, "other.png"), d, ref)[0], "text")               # 선택지 같고 글자 유사도 높음
        self.assertIsNone(D.check_row(row("<image 1> Find x.", "other.png"), d, ref)[0])                       # 짧은 질문은 완전 일치로 제거하지 않음


class RftData(unittest.TestCase):
    """RFT 학습 데이터 선택 규칙과 손실 마스크 (모델 없이 확인 가능한 부분)."""

    @staticmethod
    def _samples(spec):
        return [{"text": t, "correct": c, "finish_reason": f} for t, c, f in spec]

    def test_degenerate_detection(self):
        from build_rft import looks_degenerate

        self.assertFalse(looks_degenerate("Step 1: set up the equation.\nStep 2: solve it.\nAnswer: 3"))
        self.assertTrue(looks_degenerate("\n".join(["Let me recheck the calculation again."] * 6) + "\nAnswer: 3"))
        self.assertFalse(looks_degenerate("a\nb\nb\nb\nb\nb\nb"))        # 짧은 줄은 무시

    def test_select_samples_rules(self):
        import random

        from build_rft import select_samples

        good = [(f"solution {i} with enough detail to be unique\nAnswer: A", True, "stop") for i in range(4)]
        wrong = [(f"bad {i}\nAnswer: B", False, "stop") for i in range(3)] + [("loops...", False, "length")]
        rng = random.Random(0)
        picked = select_samples(self._samples(good + wrong), 1, 7, 2, rng)
        self.assertEqual(len(picked), 2)
        self.assertTrue(all("Answer: A" in p for p in picked))                                   # 맞은 응답만
        self.assertEqual(select_samples(self._samples([(g[0], True, "stop") for g in good] * 2), 1, 7, 2, rng), [])     # 8/8은 제외
        self.assertEqual(select_samples(self._samples(wrong * 2), 1, 7, 2, rng), [])                                    # 0/8은 제외
        trunc_only = self._samples([("long right answer\nAnswer: A", True, "length")] + wrong)
        self.assertEqual(select_samples(trunc_only, 1, 7, 2, rng), [])                           # 잘린 응답은 쓰지 않음
        dup = self._samples([("same text here\nAnswer: A", True, "stop")] * 3 + wrong)
        self.assertEqual(len(select_samples(dup, 1, 7, 3, rng)), 1)                               # 같은 응답은 한 번만

    def test_selection_is_not_biased_to_short(self):
        import random

        from build_rft import select_samples

        samples = self._samples([("x" * (50 + 40 * i) + f" {i}\nAnswer: A", True, "stop") for i in range(6)]
                                + [("bad\nAnswer: B", False, "stop")] * 2)
        lengths = []
        for seed in range(200):
            lengths += [len(p) for p in select_samples(samples, 1, 7, 1, random.Random(seed))]
        self.assertGreater(sum(lengths) / len(lengths), 130)       # 가장 짧은 것(약 60자)만 고르면 평균이 훨씬 작다

    def test_build_and_labels(self):
        from build_rft import build
        from train_rft import make_labels
        import random

        d = Path(tempfile.mkdtemp())
        row = {"id": "q1", "question": "<image 1> What is 1+1?", "options": [], "question_type": "open", "answer": "2",
               "image_paths": {"image_1": "images/a.png"}, "category": "Algebra"}
        (d / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        samples = [{"text": f"1+1=2 solution {i}\nAnswer: 2", "correct": i < 3, "finish_reason": "stop"} for i in range(8)]
        (d / "screen_samples.jsonl").write_text(json.dumps({"id": "q1", "samples": samples}) + "\n", encoding="utf-8")
        out, stats = build(("dv", str(d), "train.jsonl", None), random.Random(0), 1, 7, 2)
        self.assertEqual((len(out), stats["prompts_used"]), (2, 1))
        self.assertEqual(out[0]["source"], "dv")
        self.assertEqual(out[0]["data_dir"], str(d))
        self.assertEqual(make_labels([5, 6, 7, 8, 9], 3), [-100, -100, -100, 8, 9])
        with self.assertRaises(ValueError):
            make_labels([5, 6, 7], 3)


class ResponseOnlyLoss(unittest.TestCase):
    """응답 위치만 계산하는 교차 엔트로피가 전체 위치를 계산하는 HF식 손실과 같은 값/기울기를 주는지 (작은 가짜 모델로)."""

    def test_matches_full_logits_loss(self):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"torch를 쓸 수 없다: {type(e).__name__}")
        from train_rft import response_only_ce

        torch.manual_seed(0)
        L, H, V = 12, 8, 50
        hidden = torch.randn(1, L, H, requires_grad=True)
        head = torch.nn.Linear(H, V, bias=False)
        labels = torch.randint(0, V, (1, L))
        labels[0, :7] = -100                                   # 프롬프트 7토큰은 손실 제외
        ref = torch.nn.functional.cross_entropy(head(hidden)[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
        ce, n = response_only_ce(hidden, head, labels)
        self.assertEqual(int(n), 5)                                                 # 위치 7~11의 응답 토큰 5개
        self.assertTrue(torch.allclose(ce / n, ref, atol=1e-6), ((ce / n).item(), ref.item()))
        g_ref, = torch.autograd.grad(ref, hidden, retain_graph=True)
        g_ours, = torch.autograd.grad(ce / n, hidden)
        self.assertTrue(torch.allclose(g_ref, g_ours, atol=1e-6))


class DpoData(unittest.TestCase):
    """DPO 쌍 선택 규칙과 손실 수식 (모델 없이 확인 가능한 부분)."""

    @staticmethod
    def _s(spec):
        return [{"text": t, "correct": c, "finish_reason": f} for t, c, f in spec]

    def test_pair_selection(self):
        import random

        from build_dpo import select_pair

        good = [(f"correct solution number {i} with detail\nAnswer: A", True, "stop") for i in range(3)]
        wrong = [(f"wrong guess {i}\nAnswer: B", False, "stop") for i in range(3)]
        trunc = [(f"perhaps maybe perhaps again {i}", False, "length") for i in range(3)]
        samples = self._s(good + wrong + trunc)
        kinds = []
        for seed in range(400):
            c, r, kind = select_pair(samples, random.Random(seed), p_truncated=0.6)
            self.assertIn("correct solution", c)
            self.assertNotIn("correct solution", r)                               # 좋은 응답과 나쁜 응답은 겹치지 않음
            kinds.append(kind)
        frac = kinds.count("truncated") / len(kinds)
        self.assertTrue(0.5 < frac < 0.7, frac)                                   # 잘린 응답 우선 비율이 p_truncated 근처
        only_trunc = select_pair(self._s(good + trunc), random.Random(0))
        self.assertEqual(only_trunc[2], "truncated")
        only_wrong = select_pair(self._s(good + wrong), random.Random(0))
        self.assertEqual(only_wrong[2], "wrong")
        self.assertIsNone(select_pair(self._s(wrong + trunc), random.Random(0)))                       # 좋은 응답이 없음 (0/8)
        self.assertIsNone(select_pair(self._s([(g[0], True, "stop") for g in good] * 3), random.Random(0)))  # 8/8: 나쁜 응답 없음
        self.assertIsNone(select_pair(self._s([("\n".join(["Let me recheck the calculation again."] * 6) + "\nAnswer: A", True, "stop")] + wrong), random.Random(0)))  # 반복 줄이 있으면 좋은 응답 불가

    def test_build_pairs(self):
        import random

        from build_dpo import build

        d = Path(tempfile.mkdtemp())
        row = {"id": "q1", "question": "<image 1> Q?", "options": [], "question_type": "open", "answer": "2",
               "image_paths": {"image_1": "images/a.png"}, "category": "Algebra"}
        (d / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        samples = [{"text": f"sol {i} text\nAnswer: 2", "correct": i < 3, "finish_reason": "stop"} for i in range(4)] \
            + [{"text": f"spiral {i}", "correct": False, "finish_reason": "length"} for i in range(4)]
        (d / "screen_samples.jsonl").write_text(json.dumps({"id": "q1", "samples": samples}) + "\n", encoding="utf-8")
        out, stats = build(("dv", str(d), "train.jsonl", None), random.Random(0), 1.0, 1, 7, 2)
        self.assertEqual((len(out), stats["prompts_used"]), (2, 1))
        self.assertTrue(all(p["rejected_kind"] == "truncated" and p["chosen"] != p["rejected"] for p in out))

    def test_dpo_loss_math(self):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"torch를 쓸 수 없다: {type(e).__name__}")
        from train_dpo import dpo_loss

        t = lambda v: torch.tensor(float(v), requires_grad=True)
        lc, lr_, rc, rr = t(-10), t(-30), torch.tensor(-10.0), torch.tensor(-30.0)
        loss, margin, ok = dpo_loss(lc, lr_, rc, rr, torch.tensor(0.0), beta=0.1, sft_alpha=0.0)
        self.assertAlmostEqual(loss.item(), 0.6931, places=3)                    # 정책 = 기준이면 log 2, 마진 0
        self.assertEqual(margin.item(), 0.0)
        loss.backward()
        self.assertLess(lc.grad.item(), 0)                                        # 좋은 응답 확률을 올리면 손실이 줄어듦
        self.assertGreater(lr_.grad.item(), 0)                                    # 나쁜 응답 확률을 올리면 손실이 늘어남
        better, _, ok2 = dpo_loss(torch.tensor(-8.0), torch.tensor(-35.0), rc, rr, torch.tensor(0.0), beta=0.1, sft_alpha=0.0)
        self.assertLess(better.item(), loss.item())
        self.assertEqual(ok2.item(), 1.0)
        with_sft, _, _ = dpo_loss(torch.tensor(-10.0), torch.tensor(-30.0), rc, rr, torch.tensor(0.5), beta=0.1, sft_alpha=1.0)
        self.assertAlmostEqual(with_sft.item(), 0.6931 + 0.5, places=3)           # 보조 NLL 항이 더해짐


class MissingImageTag(unittest.TestCase):
    """질문에 '<image 1> '이 없으면 이미지가 조용히 빠진다. 학습/선별/평가 코드는 이를 오류로 알려야 한다."""

    @staticmethod
    def _data(question):
        from PIL import Image

        d = Path(tempfile.mkdtemp())
        (d / "images").mkdir()
        Image.new("RGB", (64, 32), "white").save(d / "images" / "a.png")
        row = {"id": "x_1", "question": question, "options": ["a", "b", "c", "d"], "question_type": "multiple-choice",
               "answer": "A", "image_paths": {"image_1": "images/a.png"}, "category": "KeySignature"}
        (d / "train_subset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return d, row

    def test_tagged_question_keeps_the_image(self):
        from ssmr_client import build_content
        from train_grpo import load_examples

        d, row = self._data("<image 1> 조표는?")
        self.assertIn("image_url", [c["type"] for c in build_content(row, d)])
        self.assertEqual(len(load_examples(d, d / "jpeg", "train_subset.jsonl")[0]["images"]), 1)

    def test_untagged_question_is_rejected(self):
        from ssmr_client import build_content
        from train_grpo import load_examples

        d, row = self._data("조표는?")
        with self.assertRaises(ValueError):
            build_content(row, d)
        with self.assertRaises(SystemExit):
            load_examples(d, d / "jpeg", "train_subset.jsonl")


class GrpoConfigOptions(unittest.TestCase):
    """E1' 옵션(--beta, --mask-truncated, --save-total-limit)이 GRPOConfig로 제대로 전달되는지. trl은 가짜 모듈로 대신한다."""

    @staticmethod
    def _config(**extra):
        import argparse
        import types

        captured = {}

        class FakeConfig:
            def __init__(self, **kw):
                captured.update(kw)

        fake = types.ModuleType("trl")
        fake.GRPOConfig = FakeConfig
        old = sys.modules.get("trl")
        sys.modules["trl"] = fake
        try:
            from train_grpo import build_grpo_config

            args = argparse.Namespace(prompts_per_step=8, micro_batch=1, learning_rate=2e-5, max_steps=60,
                                      max_completion_length=4096, rollout="hf", save_steps=10, vllm_mem=0.35,
                                      beta=0.0, mask_truncated=False, save_total_limit=3)
            for k, v in extra.items():
                setattr(args, k, v)
            build_grpo_config(args, "out")
        finally:
            if old is None:
                sys.modules.pop("trl", None)
            else:
                sys.modules["trl"] = old
        return captured

    def test_defaults_reproduce_e1(self):
        c = self._config()
        self.assertEqual(c["beta"], 0.0)
        self.assertEqual(c["save_total_limit"], 3)
        self.assertNotIn("mask_truncated_completions", c)      # 기본 실행은 E1과 같은 설정(옵션을 아예 넘기지 않는다)
        self.assertEqual(c["max_completion_length"], 4096)

    def test_e1_prime_options_are_passed(self):
        c = self._config(beta=0.04, mask_truncated=True, save_total_limit=10)
        self.assertEqual(c["beta"], 0.04)
        self.assertEqual(c["save_total_limit"], 10)
        self.assertIs(c["mask_truncated_completions"], True)

    def test_pipeline_forwards_extra_and_refuses_finished_run(self):
        env = dict(os.environ, DRY_RUN="1", GRPO_EXTRA="--beta 0.04 --mask-truncated", TAG_NEW="grpo_run2_test",
                   STEPS="60", SAVE_STEPS="10")
        out = subprocess.run(["bash", str(REPO / "lab" / "run_pipeline.sh"), "grpo"], env=env, cwd=REPO,
                             capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if "train_grpo.py" in l][0]
        self.assertIn("--beta 0.04 --mask-truncated", line)
        self.assertIn("results/grpo_run2_test", line)
        pilot = subprocess.run(["bash", str(REPO / "lab" / "run_pipeline.sh"), "pilot"], env=env, cwd=REPO,
                               capture_output=True, text=True).stdout
        self.assertIn("results/grpo_pilot_grpo_run2_test", pilot)          # 파일럿은 설정마다 따로
        done = REPO / "results" / "grpo_done_unittest" / "final_lora"       # results/는 .gitignore 대상
        done.mkdir(parents=True, exist_ok=True)
        try:
            (done / "adapter_model.safetensors").write_bytes(b"x")
            res = subprocess.run(["bash", str(REPO / "lab" / "run_pipeline.sh"), "grpo"], cwd=REPO, capture_output=True,
                                 text=True, env=dict(env, TAG_NEW="grpo_done_unittest"))
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("이미 학습이 끝났다", res.stdout)
            self.assertNotIn("train_grpo.py", "\n".join(l for l in res.stdout.splitlines() if l.startswith("+ python music_ft")))
        finally:
            import shutil

            shutil.rmtree(REPO / "results" / "grpo_done_unittest", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
