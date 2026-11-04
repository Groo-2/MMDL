# Colab A100 40GB에서 학습만 돌리기 (실습실 24GB에서 메모리가 모자랄 때의 대안)

목적: 실습실(RTX 4090 24GB)의 `stress` 시험에서 메모리가 부족할 때, **같은 코드, 같은 패키지 버전**으로 Colab A100 40GB에서 학습(E1')을 돌린다.
**평가는 실습실에서 한다**(학습만 Colab). 평가 환경이 바뀌면 점수 비교가 깨지기 때문이다. 아직 Colab에서 한 번도 돌려 보지 않았다(미검증).

## 먼저 알아 둘 것

- **컴퓨트 유닛이 모자라지 않은지 확인.** 이전에 Colab을 떠난 이유가 유닛 부족이었다(100유닛이 A100 약 8.5시간). E1' 60스텝은 응답이 길게 유지되면 10~20시간으로 추정되어
  유닛이 모자랄 수 있다. 먼저 `stress`와 `pilot`(합쳐 1시간 안팎 추정)만 Colab에서 돌려 메모리와 스텝 시간을 재고, 그 값으로 본 학습을 할지 정한다.
- **GPU가 다르다**(4090 -> A100). seed는 같지만 숫자가 비트 단위로 같지는 않다. 보고서에 학습 하드웨어를 적는다.
- **세션이 끊길 수 있다.** 체크포인트를 Drive에 두고(아래 3번) 같은 명령으로 이어서 돌린다. Colab에는 tmux가 없다.
- 학습에는 Llama judge와 `HF_TOKEN`이 필요 없다. Qwen 모델은 공개다.

## 0. 런타임

Colab에서 런타임 유형을 **A100 GPU**로 바꾼다(High-RAM 권장).

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!nvidia-smi | head -4        # 맨 위 "CUDA Version" 확인 (13.x여도 cu129 wheel이 동작한다)
!python3 --version           # 3.11 이상 (실습실은 3.12)
```

## 1. 코드와 데이터 준비

실습실에서 학습에 필요한 문항 파일과 이미지만 묶는다(1,120문항 분량, 전체 data/ssmr보다 훨씬 작다). 실습실에서:
```bash
cd ~/mmdl/MMDL_ssmr
python - <<'EOF'
import json, tarfile
rows = [json.loads(l) for l in open("data/ssmr/train_subset.jsonl", encoding="utf-8")]
paths = set()
for r in rows: paths.update(r["image_paths"].values())
with tarfile.open("ssmr_subset.tgz", "w:gz") as t:
    t.add("data/ssmr/train_subset.jsonl")
    for p in sorted(paths): t.add("data/ssmr/" + p)
print(len(rows), "문항,", len(paths), "이미지")
EOF
ls -lh ssmr_subset.tgz
```
이 파일(`ssmr_subset.tgz`)과 실습실의 `requirements_lab.lock.txt`를 Google Drive의 `MyDrive/mmdl/`에 올린다.

Colab에서:
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/mmdl/results
!git clone -b ssmr https://github.com/Groo-2/MMDL.git /content/MMDL
%cd /content/MMDL
!tar xzf /content/drive/MyDrive/mmdl/ssmr_subset.tgz
!cp /content/drive/MyDrive/mmdl/requirements_lab.lock.txt .
!ls data/ssmr | head
```

## 2. 실습실과 같은 환경 만들기

`FORCE_PLAN=cu129`가 핵심이다. Colab 드라이버가 CUDA 13이어도 실습실과 같은 `torch 2.13.0+cu129`, `vllm 0.27.1+cu129`를 설치한다.
`lab/lab_pins.txt`와 `LOCK_FILE`이 transformers 5.19.0, TRL 1.16.0, PEFT 0.21.2 등을 실습실 버전에 고정한다.
새 가상환경(`.venv_lab`)에 설치하므로 Colab에 미리 깔린 torch나 NVIDIA 패키지와 섞이지 않는다
(이전 Colab 사고의 원인이 cu12/cu13 NVIDIA 패키지 혼재였다).

```python
!FORCE_PLAN=cu129 LOCK_FILE=requirements_lab.lock.txt bash lab/setup_env.sh 2>&1 | tail -40
```
끝나면 실습실 값과 비교한다(아래가 모두 같아야 한다).
```python
!.venv_lab/bin/pip list 2>/dev/null | grep -i -E "^(torch|torchvision|torchaudio|vllm|transformers|trl|peft|accelerate|datasets|numpy) "
```
기대값: torch 2.13.0+cu129, torchvision 0.28.0+cu129, torchaudio 2.11.0+cu129, vllm 0.27.1+cu129, transformers 5.19.0, trl 1.16.0, peft 0.21.2,
accelerate 1.15.0, datasets 5.1.0, numpy 2.3.5. 다르면 학습을 시작하지 말고 출력을 보여 준다.
`python3 -m venv`가 실패하면 `!pip install virtualenv` 후 `VENV` 대신 virtualenv를 쓰거나 알려 달라.

## 3. 결과가 사라지지 않게 Drive에 연결

```python
!rm -rf results && ln -s /content/drive/MyDrive/mmdl/results results && ls -la results | head -3
```
체크포인트와 최종 어댑터가 Drive에 바로 저장된다(쓰기가 느릴 수 있지만 세션 끊김에서 살아남는다).

## 4. 시험과 학습 (실습실과 같은 명령, 값만 24GB 제약이 없어짐)

실습실과 **같은 설정**으로 돌린다. A100 40GB는 메모리가 충분하므로 같은 코드와 같은 값(`--max-completion-length 4096` 등)을 쓴다.
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.8"
os.environ["GRPO_EXTRA"] = "--beta 0.04 --mask-truncated --save-total-limit 12"
os.environ["MAX_COMPLETION"] = "4096"
os.environ["TAG_NEW"] = "grpo_run2"
os.environ["PROMPTS_PER_STEP"] = "1"
!bash lab/run_pipeline.sh stress 2>&1 | tail -30
```
스텝 시간, `GPU 최대 메모리`를 확인한 뒤 파일럿(`PROMPTS_PER_STEP`를 8로 되돌리고 `pilot`), 본 학습(`STEPS=60 SAVE_STEPS=10`, `grpo`) 순서로 간다.
```python
os.environ["PROMPTS_PER_STEP"] = "8"; os.environ["STEPS"] = "60"; os.environ["SAVE_STEPS"] = "10"
!bash lab/run_pipeline.sh pilot 2>&1 | tail -30
!nohup bash lab/run_pipeline.sh grpo > /content/grpo_run2.log 2>&1 &
```
진행 확인: `!tail -5 /content/grpo_run2.log`, `!grep -o "{'loss'.*" /content/grpo_run2.log | tail -2`.
세션이 끊기면 0~3번을 다시 하고 같은 `grpo` 명령을 실행하면 Drive의 마지막 체크포인트부터 이어진다(자동 재시도와 `--resume`).

## 5. 평가는 실습실에서

학습이 끝나면 `results/grpo_run2/final_lora`(어댑터)를 Drive에서 받아 실습실의 `results/grpo_run2/`에 넣고,
`TAG_NEW=grpo_run2 bash lab/run_pipeline.sh merge holdout_new music4 full2`로 평가한다. 평가 조건은 동결이다.

## 한계

- 이 문서의 절차는 Colab에서 실행해 본 적이 없다. 특히 `setup_env.sh`의 `FORCE_PLAN`/`LOCK_FILE` 경로(Mac에서 경로 선택만 테스트했다), Colab의 `python3 -m venv`,
  Drive 심볼릭 링크 위 체크포인트 저장은 처음 시도하는 것이다.
- A100에서의 스텝 시간은 모른다(실습실 4090보다 빠를 것으로 기대하지만 측정 전이다).
