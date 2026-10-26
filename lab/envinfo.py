"""실행 환경 기록과 .env 로더. 교수님 제출 양식이 요구하는 "HW 인프라와 툴 패키지 버전"을 결과 폴더에 남긴다."""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def load_env_file(path):
    """.env의 KEY=VALUE를 환경변수에 넣는다 (이미 설정된 값은 유지). HF_TOKEN용."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return out.stdout.strip() if out.returncode == 0 else f"(exit {out.returncode}) {out.stderr.strip()[:300]}"
    except Exception as e:  # noqa: BLE001
        return f"(실행 실패: {e!r})"


def snapshot(out_dir, extra=None):
    """out_dir/env_snapshot.json에 python, OS, GPU/드라이버, torch/CUDA, pip freeze를 저장한다. 실패해도 예외를 내지 않는다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvidia_smi_cuda": next((l.strip() for l in _run(["nvidia-smi"]).splitlines() if "CUDA Version" in l), None),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    try:
        import torch

        info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"(torch import 실패: {e!r})"
    if extra:
        info.update(extra)
    (out_dir / "env_snapshot.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
