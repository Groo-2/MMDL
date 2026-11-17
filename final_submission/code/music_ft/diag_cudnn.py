"""cuDNN 불일치(CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH) 진단. GPU 런타임에서 약 1분이면 끝난다.

Qwen3-VL 비전 패치 임베딩(Conv3d)을 bf16으로 한 번 돌려 보고, 어떤 cuDNN 파일이 로드됐는지 출력한다.
환경을 바꿔 가며(기본 / pip cuDNN 경로를 앞에) 같은 시험을 별도 프로세스로 반복해서
어느 쪽에서 통과하는지 알려 준다. (LD_LIBRARY_PATH를 지우는 시험은 GPU 드라이버 경로까지 지워서 뺐다.) 모델 구현은 건드리지 않는다.

  python diag_cudnn.py
"""

import glob
import os
import subprocess
import sys

PROBE = r'''
import torch
x = torch.randn(2, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
conv = torch.nn.Conv3d(3, 8, (2, 16, 16), (2, 16, 16)).cuda().to(torch.bfloat16)
try:
    conv(x)
    result = "OK"
except RuntimeError as e:
    result = "FAIL " + str(e)[:110]
print("결과:", result)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
libs = sorted({line.split()[-1] for line in open("/proc/self/maps") if "cudnn" in line})
import os
for lib in libs:
    print("  로드됨:", lib, "->", os.path.basename(os.path.realpath(lib)))
'''


def pip_cudnn_dirs():
    dirs = []
    for base in sys.path:
        dirs += glob.glob(os.path.join(base, "nvidia", "cudnn*", "lib"))
        dirs += glob.glob(os.path.join(base, "nvidia", "cudnn", "lib"))
    return sorted(set(dirs))


def run(label, env):
    print(f"\n=== {label} ===")
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip()[-400:])
    return "결과: OK" in out.stdout


def main():
    print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
    print("pip의 cuDNN 폴더:", pip_cudnn_dirs() or "없음")
    print("\n[pip 패키지 중 cudnn]")
    subprocess.run("pip list 2>/dev/null | grep -i cudnn", shell=True)
    print("\n[시스템 cuDNN (ldconfig)]")
    subprocess.run("ldconfig -p | grep -i cudnn | head -8", shell=True)

    results = {}

    results["기본"] = run("1) 기본 환경", dict(os.environ))

    env = dict(os.environ)
    dirs = pip_cudnn_dirs()
    if dirs:
        env["LD_LIBRARY_PATH"] = ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
        results["pip cuDNN 우선"] = run("2) pip cuDNN 폴더를 LD_LIBRARY_PATH 맨 앞에", env)
        print("   사용한 LD_LIBRARY_PATH =", env["LD_LIBRARY_PATH"])

    print("\n=== 요약 ===")
    for k, v in results.items():
        print(f"  {k}: {'통과' if v else '실패'}")
    if not results["기본"]:
        passed = [k for k, v in results.items() if v]
        print("→ 통과한 설정:", passed or "없음 (이 경우 로그 전체를 보내 주세요)")


if __name__ == "__main__":
    main()
