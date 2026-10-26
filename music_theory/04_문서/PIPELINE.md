# 음악 이론 학습과 baseline 평가 실행 순서

명령은 `MMDL/` 저장소 최상위에서 실행합니다.
사용 브랜치는 `music_theory`입니다.
`baseline/`과 `music_theory/`를 같은 위치에 두고, 새 결과는 `runs/` 아래에 저장합니다.

## 1. 실행 환경 구성

| 환경 | 용도 | 주요 버전 |
| --- | --- | --- |
| `theory-train` | 데이터 검사, SFT, LoRA 병합 | Transformers 4.57.1, PEFT 0.17.1 |
| `theory-eval` | vLLM 추론, judge, 최종 MMMU 평가 | vLLM 0.27.1, Transformers 5.5.3 이상 |

두 환경의 requirements를 함께 설치하지 않습니다.
공용 서버에서는 본인의 프로젝트 폴더와 배정받은 GPU만 사용하고, 시스템 드라이버와 CUDA는 변경하지 않습니다.
긴 작업은 기존 tmux 세션 안에서 실행합니다.
아래 GPU 번호 `0`은 실제 배정받은 번호로 바꿉니다.
실행 전 `nvidia-smi`로 배정 GPU의 작업 여부를 확인하고, 다른 계산 작업이 있으면 시작하지 않습니다.

학습 환경을 생성하고 활성화합니다.

```bash
conda create -p "$HOME/qwen3_andrew/envs/theory-train" python=3.12 -y
conda activate "$HOME/qwen3_andrew/envs/theory-train"
```

먼저 서버에서 승인된 GPU용 PyTorch와 torchvision 조합을 이 환경에 설치합니다.
기존에 정상 작동하는 프로젝트용 PyTorch/CUDA 배포판이 있다면 그 설치 구성을 사용합니다.
그다음 학습 requirements를 설치하고 GPU와 bfloat16 지원을 확인합니다.

```bash
python -m pip install -r "music_theory/01_학습용/requirements-train.txt"
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m pip freeze > theory-train.lock.txt
```

평가 환경은 별도로 생성합니다.

```bash
conda deactivate
conda create -p "$HOME/qwen3_andrew/envs/theory-eval" python=3.12 -y
conda activate "$HOME/qwen3_andrew/envs/theory-eval"
python -m pip install -r "music_theory/02_평가용/requirements-eval.txt"
CUDA_VISIBLE_DEVICES=0 python -c "import importlib.metadata as m, torch; print(m.version('vllm'), m.version('transformers'), torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m pip freeze > theory-eval.lock.txt
```

평가 환경에서도 CUDA 사용과 bfloat16 지원이 확인된 상태에서 진행합니다.
드라이버 호환 오류가 나면 시스템 구성을 바꾸지 않고 관리자에게 승인된 실행 환경을 확인합니다.

judge에는 `meta-llama/Llama-3.1-8B-Instruct` 접근 권한이 필요합니다.
Hugging Face에서 해당 모델의 이용 조건에 동의하고, 평가 환경의 로그인 또는 기존 `HF_TOKEN` 환경변수로 인증합니다.
토큰 값은 실행 파일, 문서나 Git에 기록하지 않습니다.

## 2. 데이터 검증과 문항 검수

학습 환경으로 전환하고 데이터 검증용 라이브러리를 설치합니다.

```bash
conda activate "$HOME/qwen3_andrew/envs/theory-train"
python -m pip install -r "music_theory/03_기타/설치파일/requirements-data.txt"
python -m pip install -r "music_theory/03_기타/설치파일/requirements-tokenizer.txt"
python -m unittest discover -s "music_theory/03_기타/도구" -p 'test_*.py' -v
python "music_theory/03_기타/도구/validate.py"
python "music_theory/03_기타/도구/check_tokenizer.py"
```

검사 항목은 이론 정답, 문제군 분리, baseline 프롬프트 일치, 마지막 `Answer:` 형식,
Qwen chat template와 assistant 응답에 대한 손실 마스킹입니다.
문항 검수는 다음 파일을 사용합니다.

- 학습 문항: `music_theory/03_기타/검수자료/training_review.html`
- 최종 평가 문항: `music_theory/02_평가용/검수자료/private_test_review.html`
- 검수 이력: `music_theory/03_기타/검수자료/human_review.json`

## 3. 원본 모델의 개발셋 평가

학습 전 원본 모델의 결과부터 저장합니다.
기본모델과 revision은 코드에서 팀 baseline으로 고정합니다.

```bash
conda activate "$HOME/qwen3_andrew/envs/theory-eval"
python "music_theory/02_평가용/run_eval.py" \
  --questions "music_theory/02_평가용/데이터/dev_questions.jsonl" \
  --key "music_theory/02_평가용/데이터/dev_answer_key.jsonl" \
  --output-dir runs/dev_base --dry-run
```

입력 검사가 통과하면 GPU를 지정해 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python "music_theory/02_평가용/run_eval.py" \
  --questions "music_theory/02_평가용/데이터/dev_questions.jsonl" \
  --key "music_theory/02_평가용/데이터/dev_answer_key.jsonl" \
  --output-dir runs/dev_base
```

`run_eval.py`는 평가 모델 서버를 실행해 답변을 생성한 뒤, judge 서버로 필요한 응답만 처리하고 채점합니다.
평가 모델의 기본 포트는 `18000`, judge 포트는 그다음 번호 `18001`입니다.
`--port`를 바꾸면 judge는 지정한 번호에 1을 더한 포트를 사용합니다. 두 포트 중 하나라도 사용 중이면 실행을 거부합니다.
작업이 끝나거나 중단되면 이 실행이 생성한 서버 프로세스만 정리합니다.

생성 조건은 temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0,
presence_penalty 1.5, seed 42, 최대 생성 32,768토큰, 문맥 한도 40,960토큰과 bfloat16입니다.
채점은 baseline의 규칙 파서와 judge를 재사용합니다.

## 4. LoRA 학습

학습 환경에서 입력과 검증 기록을 먼저 확인합니다.

```bash
conda activate "$HOME/qwen3_andrew/envs/theory-train"
python "music_theory/01_학습용/train_lora.py" \
  --output runs/theory_v22 --dry-run
```

실제 학습은 배정받은 GPU 한 장에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python "music_theory/01_학습용/train_lora.py" \
  --output runs/theory_v22 \
  --epochs 2 --lr 5e-5 --rank 16 --grad-accum 16 --seed 42
```

기본 원본은 `Qwen/Qwen3-VL-4B-Instruct`의
`ebb281ec70b05090aa6165b016eac8ec08e71b17` revision입니다.
Hugging Face 캐시 위치는 `HF_HOME`으로 지정할 수 있고, 준비된 캐시로 실행할 때는 `--local-files-only`를 사용합니다.

언어모델 attention의 q/k/v/o projection만 학습합니다.
기본 설정은 rank 16, alpha 32, dropout 0.05, 학습률 5e-5, micro batch 1,
gradient accumulation 16, 2 epochs와 학습 길이 1,536입니다.
학습 길이는 입력과 정답을 합친 학습 샘플 한도이며, 평가의 최대 생성 길이와는 별개입니다.

개발 손실로 체크포인트를 선택하고, 결과를 `runs/theory_v22/adapter/`와
`runs/theory_v22/run_manifest.json`에 저장합니다.

## 5. baseline 평가용 전체 모델 병합

같은 학습 환경에서 provenance를 검사하고 LoRA를 고정 revision 원본에 병합합니다.

```bash
python "music_theory/01_학습용/merge_lora.py" \
  --adapter runs/theory_v22/adapter \
  --output runs/theory_v22/merged --dry-run

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python "music_theory/01_학습용/merge_lora.py" \
  --adapter runs/theory_v22/adapter \
  --output runs/theory_v22/merged
```

병합 결과에는 전체 모델 가중치, 모델 설정, tokenizer와 멀티모달 processor를 함께 저장합니다.
원본의 설정과 처리 파일은 고정 revision에서 그대로 복사해 바이트 단위로 검증합니다.
`merge_manifest.json`에 원본 revision과 어댑터 및 병합 파일의 검증 정보를 기록합니다.
기존 baseline 스크립트의 `--model_path`에는 어댑터 폴더 대신 이 병합 폴더를 사용합니다.

## 6. 개발셋 전후 비교

평가 환경에서 병합 모델을 같은 조건으로 실행합니다.

```bash
conda activate "$HOME/qwen3_andrew/envs/theory-eval"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python "music_theory/02_평가용/run_eval.py" \
  --model runs/theory_v22/merged \
  --questions "music_theory/02_평가용/데이터/dev_questions.jsonl" \
  --key "music_theory/02_평가용/데이터/dev_answer_key.jsonl" \
  --output-dir runs/dev_lora

python "music_theory/02_평가용/score.py" \
  --predictions runs/dev_lora/predictions.jsonl \
  --judge-results runs/dev_lora/judge_results.jsonl \
  --baseline runs/dev_base/predictions.jsonl \
  --baseline-judge-results runs/dev_base/judge_results.jsonl \
  --key "music_theory/02_평가용/데이터/dev_answer_key.jsonl" \
  --output runs/dev_comparison.json
```

`score.py`는 생성 조건, baseline 연결 기준과 원본 모델 provenance가 같은지 확인한 뒤 비교합니다.
원본과 병합 모델의 경로가 다른 것은 허용합니다.
정확도, 파싱 실패율, 잘림 비율, 개선 및 악화 문항 수, 문제군 단위 bootstrap 구간과 McNemar 통계를 기록합니다.
잘림 여부는 서버의 `finish_reason`을 기준으로 판단합니다.

## 7. 최종 평가

개발셋으로 설정과 체크포인트를 확정한 뒤, 위 평가 명령의 질문과 정답 경로를
`test_questions.jsonl`, `test_answer_key.jsonl`로 바꿔 원본과 병합 모델을 각각 새 출력 폴더에 평가합니다.
개발 결과나 이전 v2.1의 greedy 평가 파일과 섞지 않습니다.

이후 기존 MMMU 파이프라인으로 원본과 병합 모델을 같은 조건에서 평가합니다.
아래 명령은 `MMDL/baseline/`에서 실행합니다.

```bash
cd baseline
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root /path/to/mmmu_cache \
  --output_dir ../runs/mmmu_base

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 bash scripts/run_mmmu_eval.sh \
  --model_path ../runs/theory_v22/merged \
  --data_root /path/to/mmmu_cache \
  --output_dir ../runs/mmmu_lora
cd ..
```

두 명령의 GPU 번호 `0`은 실제 배정받은 번호로 바꿉니다.
Music만 먼저 확인하려면 원본과 병합 모델 양쪽에 같은 `--subject Music`을 추가합니다.
최종 전체 평가는 같은 조건으로 900문항을 실행합니다.

## 8. 새 버전 생성과 배포

새 문항 버전은 원본 폴더를 덮어쓰지 않고 생성합니다.
명령은 다시 `MMDL/` 최상위에서 실행합니다.

```bash
python "music_theory/03_기타/도구/generate.py" \
  --output "music_theory/03_기타/생성버전/v2_next"
python "music_theory/03_기타/도구/validate.py" \
  --release "music_theory/03_기타/생성버전/v2_next"
python "music_theory/03_기타/도구/check_tokenizer.py" \
  --release "music_theory/03_기타/생성버전/v2_next"
```

공개 평가 자료와의 중복 검사를 실행합니다. `--pilot-zip`에는 팀 pilot20 원본 ZIP 경로를 지정합니다.

```bash
python "music_theory/03_기타/도구/fetch_exclusions.py" \
  --pilot-zip /path/to/TalkFile_pilot20.zip
python "music_theory/03_기타/도구/screen.py" \
  --release "music_theory/03_기타/생성버전/v2_next"
```

현재 확정된 버전은 다음 명령으로 역할별 ZIP을 생성합니다.

```bash
python "music_theory/03_기타/도구/package_release.py" \
  --output-dir "music_theory/03_기타/배포결과/v2_2"
```

배포 파일은 학습용, 평가용과 전체 파이프라인으로 구분합니다.
ZIP에는 실행에 필요한 `baseline/`도 함께 포함하며, 압축을 푼 뒤 두 폴더의 위치를 유지합니다.

## 9. 실행 기록

학습 및 평가 환경의 lock 파일, 데이터 체크섬, 원본 revision, 학습 설정,
선택한 체크포인트, 병합 manifest와 전후 평가 결과를 함께 보관합니다.
자동 검증과 모의 HTTP 검증 결과는 실제 GPU 학습 및 성능 실험 결과와 구분해 기록합니다.
