"""lab/check_vllm.py 테스트 (서버를 실제로 띄우지 않는다)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_vllm as C  # noqa: E402

UVA_LOG = "(EngineCore pid=16210) INFO Using V2 Model Runner\n(EngineCore) RuntimeError: UVA is not available\n"


class Diagnose(unittest.TestCase):
    def test_uva_on_wsl(self):
        hints = C.diagnose(UVA_LOG)
        self.assertEqual(len(hints), 1)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", hints[0])
        self.assertIn("WSL2", hints[0])

    def test_oom_and_free_memory(self):
        hints = C.diagnose("torch.OutOfMemoryError: CUDA out of memory\nValueError: Free memory on device (21.0/24) less than desired")
        self.assertEqual(len(hints), 2)
        self.assertTrue(all("Windows" in h for h in hints))

    def test_cudnn_and_unknown(self):
        self.assertIn("env_triage", C.diagnose("CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH")[0])
        self.assertEqual(C.diagnose("something unrelated"), [])


class Main(unittest.TestCase):
    def run_main(self, start, probe=None):
        d = tempfile.mkdtemp()
        stopped = []
        rc = C.main(["--log", f"{d}/check.log"], start=start, stop=lambda s: stopped.append(s),
                    probe=probe or (lambda url: {"data": [{"id": "qwen3-vl-4b"}]}))
        return rc, stopped, d

    def test_success_stops_the_server(self):
        calls = []
        rc, stopped, _ = self.run_main(lambda cmd, port, log: calls.append(cmd) or "SERVER")
        self.assertEqual(rc, 0)
        self.assertEqual(stopped, ["SERVER"])
        cmd = calls[0]
        self.assertIn("--max-model-len", cmd)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "qwen3-vl-4b")     # 평가와 같은 이름
        self.assertEqual(cmd[cmd.index("--seed") + 1], "42")

    def test_failure_reports_cause_from_log(self):
        def start(cmd, port, log):
            Path(log).write_text(UVA_LOG)
            raise RuntimeError("vLLM 서버가 종료됐습니다")

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, stopped, _ = self.run_main(start)
        self.assertEqual(rc, 1)
        self.assertEqual(stopped, [])                                       # 뜨지 않았으니 정지할 것도 없다
        self.assertIn("UVA", buf.getvalue())
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", buf.getvalue())

    def test_server_is_stopped_even_if_probe_fails(self):
        def bad_probe(url):
            raise ConnectionError("no")

        with self.assertRaises(ConnectionError):
            self.run_main(lambda cmd, port, log: "S", probe=bad_probe)

    def test_launch_env_disables_v2_runner(self):
        # 평가와 같은 환경변수를 쓰는지: UVA 실패를 막는 핵심
        import eval_repeat as E

        self.assertEqual(E.server_env()["VLLM_USE_V2_MODEL_RUNNER"], "0")


if __name__ == "__main__":
    unittest.main()
