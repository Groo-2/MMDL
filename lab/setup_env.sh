#!/usr/bin/env bash
# 실습실(RTX 4090, CUDA 12.x 드라이버) 환경 설치. 저장소 루트에서 실행한다.
#
#   bash lab/setup_env.sh              # .venv_lab 생성
#   PYTHON=python3.12 bash lab/setup_env.sh
#
# 드라이버가 지원하는 CUDA 버전(nvidia-smi 맨 위 "CUDA Version")에 맞춰 vLLM 0.27.1 빌드를 고른다.
#   13.x 이상: PyPI 기본 wheel (CUDA 13용 torch)
#   12.x     : vllm-0.27.1+cu129 wheel (GitHub 릴리스) + CUDA 12.9용 torch 2.13.0
#              드라이버가 12.9보다 낮아도(예: 12.6) CUDA 마이너 버전 호환으로 실행된다.
#              이 실습실 PC(WSL2, 드라이버 560.94 = CUDA 12.6)에서 2026-10-14, 10-20에 이 조합으로 평가를 실행한 이력이 있다.
#   12 미만  : 지원하지 않는다 → 드라이버(WSL이면 Windows 드라이버)를 올려야 한다
#
# 이미 검증된 conda 환경을 복제해서 쓰려면 (이 PC의 mmdl 환경: torch 2.13.0+cu129, vllm 0.27.1+cu129):
#   conda create --clone mmdl -n mmdl_lab
#   CONDA_ENV=mmdl_lab bash lab/setup_env.sh      # torch/vLLM이 이미 맞으면 다시 설치하지 않고 trl, peft 등만 설치
# DRY_RUN=1 이면 어떤 경로를 고르는지만 출력하고 끝낸다.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv_lab}"
CONDA_ENV="${CONDA_ENV:-}"
VLLM_VERSION="0.27.1"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi가 없다. GPU 드라이버가 설치된 PC에서 실행하세요 (WSL이면 Windows에 NVIDIA 드라이버가 있어야 한다)." >&2; exit 1
fi

CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
CUDA_MAJOR="${CUDA_VER%%.*}"
CUDA_MINOR="${CUDA_VER##*.}"
echo "드라이버 CUDA 버전: ${CUDA_VER} | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

if [[ -z "$CUDA_MAJOR" ]]; then
  echo "nvidia-smi에서 CUDA 버전을 읽지 못했다." >&2; exit 1
elif [[ "$CUDA_MAJOR" -ge 13 ]]; then
  PLAN="cu130"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
  PLAN="cu129"
else
  echo "드라이버의 CUDA ${CUDA_VER}는 지원하지 않는다 (12 이상 필요). 드라이버를 올려야 한다 (WSL이면 Windows의 NVIDIA 드라이버)." >&2
  exit 2
fi

if [[ "$PLAN" == "cu129" && "$CUDA_MINOR" -lt 9 ]]; then
  echo ">>> 드라이버 CUDA ${CUDA_VER} < 12.9: CUDA 마이너 버전 호환으로 cu129 wheel을 쓴다."
  echo "    (이 PC에서 드라이버 560.94로 vllm 0.27.1+cu129 평가를 실행한 이력이 있다. 문제가 생기면 코드보다 먼저 'python lab/env_triage.py'를 실행하세요.)"
fi
echo ">>> 설치 경로: ${PLAN}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] 경로 ${PLAN} 선택, 설치하지 않고 종료"; exit 0
fi

if [[ -n "$CONDA_ENV" ]]; then
  # shellcheck disable=SC1091
  source lab/activate_env.sh
  lab_activate_conda "$CONDA_ENV"
  echo ">>> conda 환경 사용: $CONDA_ENV ($(python --version 2>&1))"
else
  "$PYTHON" -m venv "$VENV"
  # shellcheck disable=SC1091
  source lab/activate_env.sh
  lab_activate_venv "$VENV"
  pip install -q --upgrade pip
fi

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11 이상이 필요하다 (TRL 1.16). 현재 {sys.version.split()[0]}. PYTHON=python3.12 또는 다른 conda 환경을 지정하세요.")
PY

stack_ok() {
  python - <<'PY'
import importlib.metadata as md, sys
try:
    ok = md.version("torch").startswith("2.13.0") and md.version("vllm").startswith("0.27.1")
except md.PackageNotFoundError:
    ok = False
sys.exit(0 if ok else 1)
PY
}

if stack_ok; then
  echo ">>> torch 2.13.0과 vLLM 0.27.1이 이미 설치되어 있어 다시 설치하지 않는다"
elif [[ "$PLAN" == "cu130" ]]; then
  echo ">>> CUDA 13 경로: PyPI 기본 wheel"
  pip install "vllm==${VLLM_VERSION}"
else
  echo ">>> CUDA 12 경로: cu129 빌드"
  # vllm wheel이 요구하는 torch==2.13.0 / torchvision==0.28.0 / torchaudio==2.11.0 을 CUDA 12.9 빌드로 먼저 깐다.
  # 이렇게 하면 vLLM 설치 때 PyPI의 CUDA 13 빌드 torch로 바뀌지 않는다.
  pip install "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" \
      --index-url https://download.pytorch.org/whl/cu129
  pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cu129
fi

# 학습(GRPO) + 데이터 변환 + 평가 보조
pip install "trl==1.16.0" peft "datasets>=4.7.0" huggingface_hub Pillow pyarrow pandas requests numpy

# Colab에서 PEFT와 충돌했던 낡은 torchao가 깔려 있으면 제거 (우리는 쓰지 않는다)
pip uninstall -y -q torchao 2>/dev/null || true

# vLLM/trl 설치가 torch를 다른 CUDA 빌드로 바꾸지 않았는지 확인한다 (바뀌었다면 드라이버와 맞지 않을 수 있다)
python - "$CUDA_MAJOR" "$CUDA_MINOR" <<'PY'
import sys, torch
major, minor = int(sys.argv[1]), int(sys.argv[2])
cuda = torch.version.cuda or "없음"
print(f"설치된 torch {torch.__version__} (CUDA {cuda})")
if major == 12 and not cuda.startswith("12."):
    sys.exit(f"torch가 CUDA {cuda} 빌드로 설치됐다 (기대 12.9). vLLM 설치가 torch를 바꿨을 수 있다. torch를 cu129 index에서 다시 설치하세요.")
if major >= 13 and not cuda.startswith("13"):
    sys.exit(f"torch가 CUDA {cuda} 빌드로 설치됐다 (기대 13.x).")
PY

if ! command -v nvcc >/dev/null; then
  echo ">>> nvcc가 없다: FlashInfer 샘플러의 실행 중 컴파일(JIT)을 쓸 수 없어서, 평가 서버는 VLLM_USE_FLASHINFER_SAMPLER=0으로 기본 실행한다 (lab/eval_repeat.py)."
fi

pip freeze > requirements_lab.lock.txt
echo ">>> 설치 완료. lock 파일: requirements_lab.lock.txt (보고서의 패키지 버전 기록으로 쓴다)"
python lab/env_triage.py
echo ">>> 다음: HF_TOKEN을 .env에 넣고 'bash lab/run_pipeline.sh env smoke'"
