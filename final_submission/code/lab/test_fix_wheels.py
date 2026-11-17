"""lab/fix_nvidia_wheels.py 테스트 (pip을 실제로 실행하지 않고 가짜 runner로 순서와 인자를 확인)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix_nvidia_wheels as F  # noqa: E402

# 2026-10-28 이 실습실 mmdl 환경에서 env_triage가 보고한 상태
LAB = {"torch": "2.13.0+cu129", "vllm": "0.27.1+cu129",
       "nvidia-cudnn-cu12": "9.20.0.48", "nvidia-cudnn-cu13": "9.20.0.48",
       "nvidia-nccl-cu12": "2.29.7", "nvidia-nccl-cu13": "2.29.7",
       "nvidia-cusparselt-cu12": "0.8.1", "nvidia-cusparselt-cu13": "0.8.1",
       "nvidia-nvshmem-cu12": "3.4.5", "nvidia-nvshmem-cu13": "3.4.5",
       "nvidia-cutlass-dsl-libs-cu12": "4.6.0", "nvidia-cutlass-dsl-libs-cu13": "4.6.0",
       "nvidia-cublas-cu12": "12.9.1.4", "cuda-python": "13.4.1"}


class Plan(unittest.TestCase):
    def test_lab_state(self):
        p = F.plan(LAB)
        self.assertEqual(sorted(n for n, _ in p["remove"]), sorted(n for n in LAB if n.endswith("-cu13")))
        self.assertEqual(sorted(n for n, _ in p["reinstall"]), sorted(n for n in LAB if n.endswith("-cu12") and n != "nvidia-cublas-cu12"))
        self.assertTrue(all(n.endswith("-cu12") for n, _ in p["reinstall"]))
        self.assertNotIn("nvidia-cublas-cu12", [n for n, _ in p["reinstall"]])      # 섞이지 않은 라이브러리는 건드리지 않는다
        self.assertEqual(p["leave"], [])

    def test_colab_state_with_different_versions(self):
        p = F.plan({"nvidia-cudnn-cu12": "9.27.0.42", "nvidia-cudnn-cu13": "9.20.0.48"}, keep="13")   # Colab에서는 cu13을 남겼다
        self.assertEqual(p["remove"], [("nvidia-cudnn-cu12", "9.27.0.42")])
        self.assertEqual(p["reinstall"], [("nvidia-cudnn-cu13", "9.20.0.48")])

    def test_cu13_only_libs_are_left_alone(self):
        p = F.plan({"nvidia-foo-cu13": "1.0", "nvidia-bar-cu12": "2.0", "nvidia-bar-cu13": "2.0"})
        self.assertEqual([n for n, _ in p["remove"]], ["nvidia-bar-cu13"])           # foo는 짝이 없어 대상이 아님

    def test_keep_side_missing_is_reported_not_changed(self):
        p = F.plan({"nvidia-x-cu13": "1", "nvidia-x-cu11": "1"}, keep="12")
        self.assertEqual(p["remove"], [])
        self.assertTrue(p["leave"] and "cu12 빌드가 없어" in p["leave"][0][1])

    def test_cuda_major_of_torch(self):
        self.assertEqual(F.cuda_major_of_torch("2.13.0+cu129"), "12")
        self.assertEqual(F.cuda_major_of_torch("2.13.0+cu126"), "12")
        self.assertEqual(F.cuda_major_of_torch("2.13.0+cu130"), "13")
        self.assertIsNone(F.cuda_major_of_torch("2.13.0"))


class Main(unittest.TestCase):
    def run_main(self, argv, installed):
        calls = []
        d = tempfile.mkdtemp()
        rc = F.main(argv + ["--log-dir", d], installed=installed, runner=lambda cmd: calls.append(cmd))
        return rc, calls, d

    def test_default_is_plan_only(self):
        rc, calls, d = self.run_main([], LAB)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])                                        # 아무것도 실행하지 않는다
        self.assertEqual(list(__import__("pathlib").Path(d).iterdir()), []) # 로그도 만들지 않는다

    def test_apply_order_uninstall_then_reinstall_cu12_same_versions(self):
        rc, calls, d = self.run_main(["--apply"], LAB)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("uninstall", calls[0])
        self.assertTrue(all(a.endswith("-cu13") for a in calls[0] if a.startswith("nvidia-")))
        self.assertIn("--force-reinstall", calls[1])
        self.assertIn("--no-deps", calls[1])
        self.assertIn("nvidia-cudnn-cu12==9.20.0.48", calls[1])               # 같은 버전으로 복구
        self.assertTrue(any(__import__("pathlib").Path(d).glob("pip_freeze_before_nvidia_fix_*.txt")))   # 되돌릴 기준 저장

    def test_refuses_when_torch_is_cu13_build(self):
        installed = dict(LAB, torch="2.13.0+cu130")
        rc, calls, _ = self.run_main(["--apply"], installed)
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])

    def test_nothing_to_do(self):
        rc, calls, _ = self.run_main(["--apply"], {"torch": "2.13.0+cu129", "nvidia-cudnn-cu12": "9.20.0.48"})
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
