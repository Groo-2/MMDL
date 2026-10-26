"""긴 작업을 시작하기 전 점검. 몇 시간 돌린 뒤에야 알게 되는 문제(디스크, GPU 점유, 포트, 토큰, 파일)를 먼저 잡는다.

  python lab/preflight.py --stage eval      # baseline/학습 모델 평가 (vLLM 서버 + Llama judge)
  python lab/preflight.py --stage screen    # SSMR 통과율 측정 (vLLM 서버)
  python lab/preflight.py --stage train     # GRPO 학습
  python lab/preflight.py --stage holdout   # SSMR hold-out 평가
  옵션: --offline (Llama 접근 권한의 네트워크 확인 생략)
종료 코드: FAIL이 있으면 1.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from envinfo import load_env_file  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"
ICON = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}
JUDGE_REPO = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def check_disk(path, need_gb):
    path = Path(path)
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), Path("/"))
    free = shutil.disk_usage(probe).free / 1024 ** 3
    if free < need_gb:
        return FAIL, f"디스크 여유 {free:.0f}GB < 필요 {need_gb}GB ({probe})"
    return OK, f"디스크 여유 {free:.0f}GB (필요 {need_gb}GB) ({probe})"


def check_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
    return (FAIL, f"포트 {port}를 이미 다른 프로세스가 쓰고 있다 (이전 vLLM 서버? `pkill -f 'vllm serve'`)") if in_use \
        else (OK, f"포트 {port} 사용 가능")


def parse_gpu_memory(csv_text):
    """`nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits` 첫 줄 -> (free_gb, total_gb)"""
    first = csv_text.strip().splitlines()[0]
    free, total = (float(x) for x in first.split(","))
    return free / 1024, total / 1024


def check_gpu_free(csv_text, need_gb):
    try:
        free, total = parse_gpu_memory(csv_text)
    except Exception:  # noqa: BLE001
        return FAIL, "GPU 메모리를 읽지 못했다 (nvidia-smi 출력 형식 오류)"
    if free < need_gb:
        return FAIL, f"GPU 여유 메모리 {free:.1f}GB < 필요 {need_gb}GB (전체 {total:.1f}GB). 다른 프로세스가 GPU를 쓰고 있다"
    return OK, f"GPU 여유 메모리 {free:.1f}GB / {total:.1f}GB (필요 {need_gb}GB)"


def check_files(paths):
    out = []
    for p in paths:
        p = Path(p)
        out.append((OK, f"파일 있음: {p}") if p.exists() else (FAIL, f"파일 없음: {p}"))
    return out


def check_hf_token(need_judge):
    token = os.environ.get("HF_TOKEN")
    if token:
        return OK, "HF_TOKEN 설정됨"
    return (FAIL if need_judge else WARN), "HF_TOKEN이 없다 (.env 또는 환경변수). Llama judge는 gated 모델이라 필요하다 (judge 없이 돌리려면 --no-judge)"


def check_llama_access(offline):
    token = os.environ.get("HF_TOKEN")
    if offline:
        return WARN, "Llama 접근 권한 확인 생략 (--offline)"
    if not token:
        return FAIL, "토큰이 없어 Llama 접근 권한을 확인할 수 없다"
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(JUDGE_REPO, "config.json", revision=JUDGE_REVISION, token=token)
        return OK, "Llama-3.1-8B-Instruct 접근 가능 (라이선스 동의 확인됨)"
    except Exception as e:  # noqa: BLE001
        return FAIL, f"Llama 접근 실패: {type(e).__name__}. https://huggingface.co/{JUDGE_REPO} 에서 라이선스에 동의했는지, 토큰 권한을 확인하세요"


def stage_checks(stage, offline=False, data_dir=None):
    data_dir = Path(data_dir) if data_dir else REPO / "data" / "ssmr"
    nvsmi = ""
    try:
        nvsmi = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        pass
    res = []
    if stage in ("eval", "screen", "train", "holdout"):
        res.append(check_gpu_free(nvsmi, 20) if nvsmi.strip() else (FAIL, "nvidia-smi를 실행할 수 없다 (GPU/드라이버 없음)"))
    if stage in ("eval", "screen", "holdout"):
        res.append(check_port_free(8000))
    if stage == "eval":
        res.append(check_port_free(8001))
        res.append(check_disk(REPO, 60))
        res.append(check_hf_token(True))
        res.append(check_llama_access(offline))
    if stage in ("screen", "train", "holdout"):
        res.append(check_disk(REPO, 20))
    if stage == "screen":
        res += check_files([data_dir / "train.dedup.jsonl"])
    if stage == "train":
        res += check_files([data_dir / "train_subset.jsonl"])
    if stage == "holdout":
        res += check_files([data_dir / "holdout.jsonl"])
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["eval", "screen", "train", "holdout"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    load_env_file(REPO / ".env")
    results = stage_checks(args.stage, args.offline, args.data_dir)
    for level, msg in results:
        print(f"{ICON[level]} [{level}] {msg}")
    fails = sum(1 for r in results if r[0] == FAIL)
    print(f"\n결과: FAIL {fails}개" + (" -> 해결 후 다시 실행하세요" if fails else " -> 시작해도 됩니다"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
