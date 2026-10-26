#!/usr/bin/env bash
# 실습실(RTX 4090, CUDA 12.x 드라이버) 환경 설치. 저장소 루트에서 실행한다.
#
#   bash lab/setup_env.sh              # .venv_lab 생성
#   PYTHON=python3.12 bash lab/setup_env.sh
#
# 드라이버가 지원하는 CUDA 버전(nvidia-smi 맨 위 "CUDA Version")에 맞춰 vLLM 0.27.1 빌드를 고른다.
#   13.x 이상  : PyPI 기본 wheel (CUDA 13용 torch)
#   12.9 ~ 12.x: vllm-0.27.1+cu129 wheel (GitHub 릴리스) + CUDA 12.9용 torch 2.13.0
#   12.9 미만  : vLLM 0.27.1에는 맞는 빌드가 없다 → 드라이버를 올려야 한다 (4090은 CUDA forward-compat 불가)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv_lab}"
VLLM_VERSION="0.27.1"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi가 없다. GPU 드라이버가 설치된 PC에서 실행하세요." >&2; exit 1
fi

CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
CUDA_MAJOR="${CUDA_VER%%.*}"
CUDA_MINOR="${CUDA_VER##*.}"
echo "드라이버 CUDA 버전: ${CUDA_VER} | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11 이상이 필요하다 (TRL 1.16). 현재 {sys.version.split()[0]}. PYTHON=python3.12 처럼 지정하세요.")
PY

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip

if [[ "$CUDA_MAJOR" -ge 13 ]]; then
  echo ">>> CUDA 13 경로: PyPI 기본 wheel"
  pip install "vllm==${VLLM_VERSION}"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 9 ]]; then
  echo ">>> CUDA 12.9 경로: cu129 빌드"
  # vllm wheel이 요구하는 torch==2.13.0 / torchvision==0.28.0 / torchaudio==2.11.0 을 CUDA 12.9 빌드로 먼저 깐다.
  # 이렇게 하면 vLLM 설치 때 PyPI의 CUDA 13 빌드 torch로 바뀌지 않는다.
  pip install "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" \
      --index-url https://download.pytorch.org/whl/cu129
  pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cu129
else
  echo "드라이버의 CUDA ${CUDA_VER}는 vLLM ${VLLM_VERSION}의 어떤 빌드(CUDA 12.9 / 13)와도 맞지 않는다." >&2
  echo "→ NVIDIA 드라이버를 575 이상(CUDA 12.9 이상)으로 올리거나, 다른 vLLM 버전을 쓰도록 요청하세요." >&2
  exit 2
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
if major == 12 and not cuda.startswith("12.9"):
    sys.exit(f"torch가 CUDA {cuda} 빌드로 설치됐다 (기대 12.9). vLLM 설치가 torch를 바꿨을 수 있다. torch를 cu129 index에서 다시 설치하세요.")
if major >= 13 and not cuda.startswith("13"):
    sys.exit(f"torch가 CUDA {cuda} 빌드로 설치됐다 (기대 13.x).")
PY

pip freeze > requirements_lab.lock.txt
echo ">>> 설치 완료. lock 파일: requirements_lab.lock.txt (보고서의 패키지 버전 기록으로 쓴다)"
python lab/env_triage.py
echo ">>> 다음: HF_TOKEN을 .env에 넣고 'bash lab/run_pipeline.sh env smoke'"
