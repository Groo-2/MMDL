"""vLLM 서버가 이 환경에서 뜨는지 빠르게 확인한다 (약 2~4분). 평가/선별/hold-out이 쓰는 것과 같은 기동 코드와 환경변수를 쓴다.

  python lab/check_vllm.py
  bash lab/run_pipeline.sh vllm_check

직접 `vllm serve ...`를 치지 말고 이것을 쓰세요. 직접 치면 우리가 항상 붙이는 환경변수(VLLM_USE_V2_MODEL_RUNNER=0 등)가 빠져서
다른 이유로 실패한다 (2026-10-28: WSL2에서 "UVA is not available").
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 로그에서 찾은 문구 -> 원인과 대응
HINTS = [
    ("UVA is not available",
     "vLLM V2 model runner는 UVA(호스트 메모리 직접 주소 지정)가 필요한데 WSL2는 지원하지 않는다. "
     "VLLM_USE_V2_MODEL_RUNNER=0이 필요하다 (이 도구와 모든 평가 단계는 자동 설정). 직접 vllm serve를 칠 때는 명령 앞에 붙이세요."),
    ("out of memory",
     "GPU 메모리 부족. Windows 앱(브라우저, 영상, 게임)이 VRAM을 쓰고 있을 수 있다. 닫거나 --gpu-mem-util을 낮추세요."),
    ("Free memory on device",
     "시작 시 GPU 여유가 요구량(전체 x gpu-memory-utilization)보다 적다. Windows 앱이 쓰는 VRAM을 닫거나 --gpu-mem-util을 낮추세요."),
    ("nvcc",
     "nvcc를 찾지 못했다. FlashInfer 샘플러 JIT 때문일 수 있다 (VLLM_USE_FLASHINFER_SAMPLER=0이 기본이므로 환경변수를 덮어쓰지 않았는지 확인)."),
    ("CUDNN_STATUS", "cuDNN 불일치. python lab/env_triage.py의 NVIDIA wheel 항목을 먼저 확인하세요."),
    ("no kernel image", "이 GPU 아키텍처용 커널이 없다. torch/vLLM 빌드(cu129/cu130)와 드라이버를 python lab/env_triage.py로 확인하세요."),
]


def diagnose(log_text):
    """로그 텍스트에서 알려진 원인을 찾아 설명 목록을 돌려준다."""
    low = log_text.lower()
    return [msg for key, msg in HINTS if key.lower() in low]


def main(argv=None, start=None, stop=None, probe=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", default=str(REPO / "logs" / "check_vllm.log"))
    args = parser.parse_args(argv)

    if start is None or stop is None:
        from eval_repeat import start_server, stop_server    # 평가와 같은 기동 코드/환경변수 (server_env)

        start, stop = start or start_server, stop or stop_server
    if probe is None:
        import requests

        probe = lambda url: requests.get(url, timeout=10).json()   # noqa: E731

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["vllm", "serve", args.model, "--revision", args.revision, "--served-model-name", "qwen3-vl-4b", "--dtype", "bfloat16",
           "--seed", "42", "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", str(args.gpu_mem_util),
           "--port", str(args.port)]
    t0 = time.time()
    try:
        server = start(cmd, args.port, log)
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ vLLM 서버가 뜨지 않았다: {e}")
        text = log.read_text(errors="ignore") if log.exists() else ""
        hints = diagnose(text)
        for h in hints:
            print(f"   -> 원인 추정: {h}")
        if not hints:
            print("   -> 알려진 원인을 찾지 못했다. python lab/env_triage.py 결과와 아래 로그를 같이 보내 주세요:")
        print(f"   로그: {log}")
        return 1
    try:
        models = probe(f"http://localhost:{args.port}/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"\n✅ vLLM 서버 정상 기동 ({time.time() - t0:.0f}초). 서빙 중인 모델: {ids}")
    finally:
        stop(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
