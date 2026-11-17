"""conda/venv 활성화 도우미(lab/activate_env.sh) 테스트.

실습실 mmdl 환경의 activate.d 스크립트(~cuda-nvcc_activate.sh)가 설정되지 않은 NVCC_PREPEND_FLAGS를 참조해서,
set -u 모드의 setup_env.sh가 `conda activate`에서 "unbound variable"로 중단됐다. 그 상황을 가짜 conda 환경으로 재현한다.

  python -m unittest lab.test_env_activate -v
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def fake_conda():
    base = Path(tempfile.mkdtemp())
    (base / "etc" / "profile.d").mkdir(parents=True)
    (base / "bin").mkdir()
    # conda 셸 함수: activate.d의 스크립트를 순서대로 source한다 (실제 conda와 같은 방식)
    (base / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() {\n  if [[ "$1" == "activate" ]]; then\n'
        '    for f in "$FAKE_CONDA_BASE"/envs/"$2"/etc/conda/activate.d/*.sh; do source "$f" || return $?; done\n'
        '    export CONDA_DEFAULT_ENV="$2"\n  fi\n}\n')
    # `conda info --base`는 실행 파일로 호출된다
    exe = base / "bin" / "conda"
    exe.write_text('#!/bin/bash\nif [[ "$1 $2" == "info --base" ]]; then echo "$FAKE_CONDA_BASE"; fi\n')
    exe.chmod(0o755)
    d = base / "envs" / "fakeenv" / "etc" / "conda" / "activate.d"
    d.mkdir(parents=True)
    # 실제 파일명(~로 시작)과 같은 문제: 설정되지 않은 변수를 기본값 없이 참조한다
    (d / "~cuda-nvcc_activate.sh").write_text('export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin /fake/gcc"\n')
    return base


def run(script, base, extra_env=None):
    env = {"PATH": f"{base}/bin:/usr/bin:/bin", "FAKE_CONDA_BASE": str(base), "HOME": str(base)}
    env.update(extra_env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd=REPO)


class ActivateHelper(unittest.TestCase):
    def test_naive_activation_fails_under_set_u(self):
        # 이번 실패의 재현: 도우미 없이 set -u에서 활성화
        base = fake_conda()
        out = run('set -euo pipefail\nsource "$(conda info --base)/etc/profile.d/conda.sh"\nconda activate fakeenv\necho OK', base)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("unbound variable", out.stderr)

    def test_helper_activates_under_set_u_and_restores_it(self):
        base = fake_conda()
        out = run('set -euo pipefail\nsource lab/activate_env.sh\nlab_activate_conda fakeenv\n'
                  'echo "FLAGS=$NVCC_PREPEND_FLAGS"\necho "ENV=$CONDA_DEFAULT_ENV"\necho "OPTS=$-"', base)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("FLAGS= -ccbin /fake/gcc", out.stdout)
        self.assertIn("ENV=fakeenv", out.stdout)
        self.assertRegex(out.stdout, r"OPTS=\w*u\w*")            # set -u가 다시 켜져 있어야 한다
        # 활성화 이후에는 다시 엄격하다: 미설정 변수를 참조하면 실패
        again = run('set -euo pipefail\nsource lab/activate_env.sh\nlab_activate_conda fakeenv\necho "$NOT_DEFINED_ANYWHERE"', base)
        self.assertNotEqual(again.returncode, 0)

    def test_helper_does_not_enable_set_u_if_it_was_off(self):
        base = fake_conda()
        out = run('set -eo pipefail\nsource lab/activate_env.sh\nlab_activate_conda fakeenv\necho "OPTS=$-"', base)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotRegex(out.stdout, r"OPTS=\w*u\w*")

    def test_helper_reports_activation_failure(self):
        base = fake_conda()
        (base / "envs" / "fakeenv" / "etc" / "conda" / "activate.d" / "~cuda-nvcc_activate.sh").write_text("return 7\n")
        out = run('set -uo pipefail\nsource lab/activate_env.sh\nlab_activate_conda fakeenv; echo "RC=$?"', base)
        self.assertIn("RC=7", out.stdout)                           # 실패를 삼키지 않는다

    def test_venv_activation_under_set_u(self):
        base = Path(tempfile.mkdtemp())
        (base / "bin").mkdir()
        (base / "bin" / "activate").write_text('export VENV_FLAG="${UNSET_VENV_VAR} set"\n')    # 미설정 변수 참조
        out = run(f'set -euo pipefail\nsource lab/activate_env.sh\nlab_activate_venv "{base}"\necho "FLAG=$VENV_FLAG"', base)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("FLAG= set", out.stdout)

    def test_scripts_use_the_helper(self):
        for name in ("setup_env.sh", "run_pipeline.sh"):
            text = (REPO / "lab" / name).read_text()
            self.assertIn("lab_activate_conda", text, name)
            self.assertNotIn("conda activate", text.replace("tmux 안에서 conda activate는 하지 말고", ""), name)


if __name__ == "__main__":
    unittest.main()
