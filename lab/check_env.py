"""설치 직후 환경 점검 (GPU 필요, 약 20초). 실패하면 이유를 알려 주고 종료 코드 1로 끝난다.

  python lab/check_env.py
"""

import importlib.metadata as md
import sys


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "미설치"


def main():
    import torch

    ok = True
    print(f"torch {torch.__version__} | torch가 쓰는 CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    for pkg in ("vllm", "trl", "peft", "transformers", "datasets", "nvidia-cudnn-cu12", "nvidia-cudnn-cu13"):
        print(f"  {pkg:<22}{version(pkg)}")

    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() == False (드라이버/torch 빌드 불일치)")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory / 1024 ** 3:.1f} GB | compute capability {props.major}.{props.minor}")
    if props.major < 8:
        print("❌ bf16을 지원하지 않는 GPU")
        ok = False

    # 1) Qwen3-VL 비전 패치 임베딩과 같은 형태의 conv3d (Colab에서 cuDNN 라이브러리가 섞여 실패했던 부분)
    try:
        x = torch.randn(2, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(3, 8, (2, 16, 16), (2, 16, 16)).cuda().to(torch.bfloat16)
        conv(x)
        print("✅ conv3d (bf16, cuDNN) 정상")
    except RuntimeError as e:
        print(f"❌ conv3d 실패: {str(e)[:150]}")
        print("   nvidia-cudnn-cu12 / -cu13이 둘 다 설치돼 있으면 파일이 섞일 수 있다 (위 목록 확인).")
        ok = False

    # 2) bf16 행렬곱
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(a @ a).all()
    print("✅ bf16 matmul 정상")

    # 3) vLLM / TRL import
    try:
        import vllm  # noqa: F401
        import trl  # noqa: F401
        print("✅ vllm, trl import 정상")
    except Exception as e:  # noqa: BLE001
        print(f"❌ import 실패: {e!r}")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
