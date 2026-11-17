"""NVIDIA wheel 정리: 같은 라이브러리의 cu12/cu13이 함께 설치된 경우, 쓰지 않는 쪽(기본 cu13)을 지우고 남길 쪽(cu12)을 복구한다.

  python lab/fix_nvidia_wheels.py            # 계획만 출력 (아무것도 바꾸지 않음)
  python lab/fix_nvidia_wheels.py --apply    # 실행

주의: cu12와 cu13 wheel은 **같은 경로**(예: nvidia/cudnn/lib/libcudnn.so.9)에 파일을 설치한다. cu13만 지우면 cu12가 쓰는 파일까지
삭제되므로, 지운 뒤 cu12를 같은 버전으로 --force-reinstall --no-deps 해서 파일을 되살린다 (Colab에서 cuDNN 불일치를 고칠 때 쓴 방법).
torch가 남길 CUDA 메이저(기본 12)로 빌드된 경우에만 실행한다. 실행 전 pip freeze를 logs/에 저장한다.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAIR = re.compile(r"nvidia-(.+)-cu(\d+)")


def cuda_major_of_torch(version):
    """'2.13.0+cu129' -> '12', '2.13.0+cu130' -> '13', 로컬 태그가 없으면 None"""
    m = re.search(r"\+cu(\d+)", str(version))
    return m.group(1)[:2] if m else None


def plan(installed, keep="12"):
    """installed: {소문자 패키지명: 버전}. 반환: remove(지울 것), reinstall(복구할 것), leave(건드리지 않는 것과 이유)"""
    by_lib = {}
    for name, ver in installed.items():
        m = PAIR.fullmatch(name)
        if m:
            by_lib.setdefault(m.group(1), {})[m.group(2)] = (name, ver)
    remove, reinstall, leave = [], [], []
    for lib, variants in sorted(by_lib.items()):
        if len(variants) < 2:
            continue                                   # 섞이지 않은 라이브러리는 대상이 아니다
        if keep not in variants:
            leave.append((f"nvidia-{lib}", f"cu{keep} 빌드가 없어서 건드리지 않는다 (있는 것: {sorted('cu' + k for k in variants)})"))
            continue
        for k, (name, ver) in sorted(variants.items()):
            if k != keep:
                remove.append((name, ver))
        reinstall.append(variants[keep])
    return {"remove": remove, "reinstall": reinstall, "leave": leave}


def installed_packages():
    out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)
    return {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(out)}


def main(argv=None, installed=None, runner=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep", default="12", help="남길 CUDA 메이저 (torch 빌드와 같아야 한다)")
    parser.add_argument("--log-dir", default=str(REPO / "logs"))
    args = parser.parse_args(argv)

    installed = installed if installed is not None else installed_packages()
    runner = runner or (lambda cmd: subprocess.check_call(cmd))
    p = plan(installed, args.keep)

    torch_major = cuda_major_of_torch(installed.get("torch", ""))
    print(f"torch {installed.get('torch', '(미설치)')} -> CUDA {torch_major}, 남길 계열: cu{args.keep}")
    for name, ver in p["remove"]:
        print(f"  지움      {name}=={ver}")
    for name, ver in p["reinstall"]:
        print(f"  복구      {name}=={ver} (--force-reinstall --no-deps: 같은 경로의 파일을 되살린다)")
    for name, why in p["leave"]:
        print(f"  건드리지 않음  {name}: {why}")
    if not p["remove"]:
        print("정리할 것이 없다 (cu12/cu13이 함께 설치된 NVIDIA 패키지가 없다).")
        return 0

    if torch_major != args.keep:
        print(f"중단: torch가 CUDA {torch_major} 빌드라서 cu{args.keep}만 남기면 깨진다. --keep을 torch와 같게 지정하세요.", file=sys.stderr)
        return 2
    if not args.apply:
        print("\n계획만 출력했다. 실행하려면 --apply (실행 전 pip freeze를 logs/에 저장한다).")
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    freeze = log_dir / f"pip_freeze_before_nvidia_fix_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    freeze.write_text("\n".join(f"{n}=={v}" for n, v in sorted(installed.items())) + "\n")
    print(f"\n되돌릴 기준 저장: {freeze}")

    py = [sys.executable, "-m", "pip"]
    runner(py + ["uninstall", "-y"] + [n for n, _ in p["remove"]])
    runner(py + ["install", "--force-reinstall", "--no-deps"] + [f"{n}=={v}" for n, v in p["reinstall"]])
    print("\n완료. 다음: python lab/env_triage.py (NVIDIA wheel 항목이 OK로 바뀌어야 한다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
