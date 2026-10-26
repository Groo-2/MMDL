"""CUDA·패키지 버전 점검 (triage). 에러가 반복되면 코드를 고치기 전에 이것부터 실행한다.

  python lab/env_triage.py                    # 점검 + logs/env_triage_<시각>.txt 저장
  python lab/env_triage.py --after-failure    # 파이프라인 단계 실패 직후 자동 호출용 (안내 문구 포함)
  python lab/env_triage.py --allow-no-gpu     # GPU 없는 PC(Mac 등)에서 로직만 시험할 때

점검 항목
  1) 드라이버가 지원하는 CUDA >= torch가 쓰는 CUDA
  2) nvidia-*-cu12 / -cu13 wheel이 섞여 있는지 (Colab 사고 원인: cuDNN 하위 라이브러리 버전 불일치)
  3) 핵심 패키지 버전(torch, vllm, trl, transformers)이 기대값과 lock 파일에 맞는지
  4) pip check
  5) bf16 matmul, conv3d(Qwen3-VL 비전 패치 임베딩과 같은 형태) 스모크
종료 코드: FAIL이 하나라도 있으면 1.
"""

import argparse
import importlib.metadata as md
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# vLLM 0.27.1 wheel 메타데이터가 요구하는 값 (torch==2.13.0, transformers>=5.5.3). trl은 우리가 검증한 버전.
EXPECTED_EXACT = {"torch": "2.13.0", "vllm": "0.27.1", "trl": "1.16.0"}
EXPECTED_MIN = {"transformers": "5.5.3"}
KEY_PACKAGES = ["torch", "torchvision", "torchaudio", "vllm", "trl", "transformers", "peft", "accelerate", "datasets", "numpy"]

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
ICON = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭️ "}


# ------------------------------------------------------------
# 순수 함수 (모의 입력으로 테스트 가능)
# ------------------------------------------------------------

def parse_nvidia_smi(text):
    """`nvidia-smi` 전체 출력에서 드라이버 버전과 지원 CUDA 버전을 뽑는다."""
    driver = re.search(r"Driver Version:\s*([\d.]+)", text)
    cuda = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return {
        "driver": driver.group(1) if driver else None,
        "cuda": (int(cuda.group(1)), int(cuda.group(2))) if cuda else None,
    }


def base_version(version):
    """'2.13.0+cu129' -> '2.13.0'"""
    return str(version).split("+")[0]


def version_tuple(version):
    return tuple(int(x) for x in re.findall(r"\d+", base_version(version))[:3])


def driver_vs_torch(driver_cuda, torch_cuda):
    """드라이버가 지원하는 CUDA가 torch의 CUDA 이상이어야 한다. 반환: (level, 메시지)"""
    if driver_cuda is None:
        return FAIL, "nvidia-smi에서 드라이버의 CUDA 버전을 읽지 못했다 (드라이버 미설치?)"
    if not torch_cuda:
        return FAIL, "설치된 torch가 CUDA 빌드가 아니다 (CPU 빌드)"
    t = tuple(int(x) for x in torch_cuda.split(".")[:2])
    if driver_cuda < t:
        return FAIL, (f"드라이버가 지원하는 CUDA {driver_cuda[0]}.{driver_cuda[1]} < torch CUDA {torch_cuda}. "
                      "드라이버를 올리거나 더 낮은 CUDA용 torch를 설치해야 한다 (RTX 4090은 forward-compat 불가)")
    return OK, f"드라이버 CUDA {driver_cuda[0]}.{driver_cuda[1]} >= torch CUDA {torch_cuda}"


def find_nvidia_wheel_mix(installed, torch_cuda):
    """nvidia-<lib>-cuNN wheel 중 같은 라이브러리가 두 CUDA 계열로 섞였거나, torch와 다른 계열이 있으면 문제 목록을 돌려준다."""
    by_lib = {}
    for name, ver in installed.items():
        m = re.fullmatch(r"nvidia-(.+)-cu(\d+)", name)
        if m:
            by_lib.setdefault(m.group(1), {})[m.group(2)] = ver
    problems = []
    for lib, variants in sorted(by_lib.items()):
        if len(variants) > 1:
            shown = ", ".join(f"cu{k}={v}" for k, v in sorted(variants.items()))
            problems.append(f"{lib}: {shown} (같은 폴더에 섞여 설치되면 하위 라이브러리 버전이 어긋남)")
    if torch_cuda:
        want = torch_cuda.split(".")[0]
        for lib, variants in sorted(by_lib.items()):
            if want not in variants:
                problems.append(f"{lib}: torch는 CUDA {torch_cuda}인데 cu{want} 빌드가 없고 {sorted('cu' + k for k in variants)}만 있다")
    return problems


def check_versions(installed):
    """핵심 패키지 버전을 기대값과 비교한다. 반환: [(level, 메시지)]"""
    out = []
    for pkg, want in EXPECTED_EXACT.items():
        have = installed.get(pkg)
        if have is None:
            out.append((FAIL, f"{pkg} 미설치 (기대 {want})"))
        elif base_version(have) != want:
            out.append((FAIL, f"{pkg} {have} (기대 {want})"))
        else:
            out.append((OK, f"{pkg} {have}"))
    for pkg, want in EXPECTED_MIN.items():
        have = installed.get(pkg)
        if have is None:
            out.append((FAIL, f"{pkg} 미설치 (기대 >= {want})"))
        elif version_tuple(have) < version_tuple(want):
            out.append((FAIL, f"{pkg} {have} (vLLM 0.27.1은 >= {want}을 요구)"))
        else:
            out.append((OK, f"{pkg} {have} (>= {want})"))
    return out


def compare_lock(installed, lock_text):
    """setup 시점에 저장한 lock(pip freeze)과 현재 핵심 패키지 버전을 비교한다. 다르면 문장 목록."""
    locked = {}
    for line in lock_text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line.strip())
        if m:
            locked[m.group(1).lower().replace("_", "-")] = m.group(2)
    diffs = []
    for pkg in KEY_PACKAGES:
        if pkg in locked and pkg in installed and locked[pkg] != installed[pkg]:
            diffs.append(f"{pkg}: lock {locked[pkg]} -> 현재 {installed[pkg]}")
        elif pkg in locked and pkg not in installed:
            diffs.append(f"{pkg}: lock에는 {locked[pkg]}가 있으나 현재 미설치")
    return diffs


def installed_packages():
    out = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name:
            out[name] = dist.version
    return out


# ------------------------------------------------------------
# 실제 점검
# ------------------------------------------------------------

def run_checks(allow_no_gpu=False, lock_path=None):
    results = []
    add = lambda level, title, detail="": results.append((level, title, detail))  # noqa: E731

    # 1) 드라이버 vs torch CUDA
    smi = ""
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        pass
    info = parse_nvidia_smi(smi)

    torch_cuda, torch_ver, cuda_ok = None, None, False
    try:
        import torch

        torch_ver, torch_cuda = torch.__version__, torch.version.cuda
        cuda_ok = torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        add(FAIL, "torch import", repr(e))
        torch = None

    if not smi:
        add(SKIP if allow_no_gpu else FAIL, "드라이버/GPU", "nvidia-smi가 없거나 실행되지 않는다")
    elif torch is not None:
        level, msg = driver_vs_torch(info["cuda"], torch_cuda)
        add(level, "드라이버 CUDA vs torch CUDA", f"{msg} | 드라이버 {info['driver']}, torch {torch_ver}")

    # 2) cu12/cu13 혼재
    pkgs = installed_packages()
    problems = find_nvidia_wheel_mix(pkgs, torch_cuda)
    if problems:
        add(FAIL, "NVIDIA wheel 혼재/불일치", "\n      ".join(problems))
    else:
        add(OK, "NVIDIA wheel 혼재 없음", f"nvidia-* 패키지 {sum(1 for n in pkgs if n.startswith('nvidia-'))}개")

    # 3) 버전
    for level, msg in check_versions(pkgs):
        add(level, "버전", msg)
    lock = Path(lock_path) if lock_path else REPO / "requirements_lab.lock.txt"
    if lock.exists():
        diffs = compare_lock(pkgs, lock.read_text())
        add(FAIL if diffs else OK, "lock 파일과 비교", "\n      ".join(diffs) if diffs else f"{lock.name}와 일치")
    else:
        add(WARN, "lock 파일 없음", f"{lock} (setup_env.sh가 만든다)")

    # 4) pip check
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=300)
        lines = [l for l in out.stdout.splitlines() if l.strip() and "No broken requirements" not in l]
        key_lines = [l for l in lines if any(k in l.lower() for k in KEY_PACKAGES)]
        if key_lines:
            add(FAIL, "pip check (핵심 패키지)", "\n      ".join(key_lines[:10]))
        elif lines:
            add(WARN, "pip check (기타 패키지)", "\n      ".join(lines[:5]))
        else:
            add(OK, "pip check", "깨진 의존성 없음")
    except Exception as e:  # noqa: BLE001
        add(WARN, "pip check", f"실행 실패: {e!r}")

    # 5) GPU 스모크
    if torch is not None and cuda_ok:
        try:
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            assert torch.isfinite(a @ a).all()
            add(OK, "bf16 matmul")
        except Exception as e:  # noqa: BLE001
            add(FAIL, "bf16 matmul", repr(e)[:200])
        try:
            x = torch.randn(2, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(3, 8, (2, 16, 16), (2, 16, 16)).cuda().to(torch.bfloat16)
            conv(x)
            add(OK, "conv3d (cuDNN)", f"cuDNN {torch.backends.cudnn.version()}")
        except Exception as e:  # noqa: BLE001
            add(FAIL, "conv3d (cuDNN)", f"{repr(e)[:200]}\n      -> 위 'NVIDIA wheel' 항목을 먼저 확인 (cu12/cu13 혼재가 원인이었음)")
        props = torch.cuda.get_device_properties(0)
        add(FAIL if props.major < 8 else OK, "GPU",
            f"{props.name}, {props.total_memory / 1024 ** 3:.1f} GB, compute capability {props.major}.{props.minor}"
            + (" (bf16 미지원)" if props.major < 8 else ""))
    else:
        add(SKIP if allow_no_gpu else FAIL, "GPU 스모크", "CUDA GPU를 쓸 수 없다 (torch.cuda.is_available() == False)")
    return results


def render(results, after_failure=False):
    lines = []
    if after_failure:
        lines += ["=" * 70,
                  "단계가 실패했다. 코드를 고치기 전에 아래 CUDA/패키지 점검 결과부터 확인하세요.",
                  "=" * 70]
    for level, title, detail in results:
        lines.append(f"{ICON[level]} [{level}] {title}" + (f": {detail}" if detail else ""))
    fails = sum(1 for r in results if r[0] == FAIL)
    lines.append("")
    lines.append(f"결과: FAIL {fails}개" + (" -> 위 FAIL 항목을 먼저 해결하세요." if fails else " -> 환경은 정상으로 보입니다. 에러 로그 자체를 확인하세요."))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--after-failure", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--lock", default=None)
    parser.add_argument("--log-dir", default=str(REPO / "logs"))
    args = parser.parse_args()

    results = run_checks(allow_no_gpu=args.allow_no_gpu, lock_path=args.lock)
    text = render(results, after_failure=args.after_failure)
    print(text)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"env_triage_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    log.write_text(text + "\n")
    print(f"\n로그 저장: {log}")
    sys.exit(1 if any(r[0] == FAIL for r in results) else 0)


if __name__ == "__main__":
    main()
