# 팀 실행 순서

아래 명령은 최상위 README.md가 있는 폴더에서 실행합니다.

## 1. 어디서 무엇을 하는지

| 폴더 | 역할 |
| --- | --- |
| `01_학습용/` | 학습 코드와 학습/개발 데이터 |
| `02_평가용/` | 답변 생성, 채점, 평가 데이터, 최종 평가 검수 |
| `03_기타/도구/` | 생성기, 검증기, 토크나이저 검사, 배포 도구 |
| `03_기타/검증기록/` | 데이터 버전, 검사 결과, 생성 근거 |
| `03_기타/검수자료/` | 학습 문항 검수 화면과 검수 이력 |
| `04_문서/` | 사용 안내와 연구 근거 |

실행 시 전체 폴더 구조를 유지합니다. 최종 평가 정답과 생성 근거는 평가 담당자가 관리합니다.

## 2. 검증과 문항 검수

데이터 검증은 Mac 또는 Ubuntu의 Python 3.11 이상에서 실행합니다.

```bash
python -m pip install -r "03_기타/설치파일/requirements-data.txt"
python -m pip install -r "03_기타/설치파일/requirements-tokenizer.txt"
python -m unittest discover -s "03_기타/도구" -p test_pipeline.py -v
python "03_기타/도구/validate.py"
python "03_기타/도구/check_tokenizer.py" --model Qwen/Qwen3-VL-4B-Instruct
```

팀 baseline의 모델 경로나 commit을 알고 있으면 `--model`, `--revision`을 그 값으로 지정합니다.
토크나이저 검사에서는 입력 길이와 assistant 답변 부분의 학습 처리를 확인합니다.

- 학습 문항 검수: `03_기타/검수자료/training_review.html`
- 최종 평가 검수: `02_평가용/검수자료/private_test_review.html`
- 검수 이력: `03_기타/검수자료/human_review.json`

## 3. 학습 실행

서버의 프로젝트 전용 conda 환경에서, 정상 동작하는 PyTorch/CUDA 환경을 사용합니다.
필요한 학습 라이브러리를 설치합니다.

```bash
python -m pip install -r "01_학습용/requirements-train.txt"
```

`/path/to/team_baseline`을 실제 팀 baseline 체크포인트 경로로 바꾸고 입력을 검사합니다.

```bash
python "01_학습용/train_lora.py" \
  --model /path/to/team_baseline --output runs/theory_v2 --dry-run
```

dry-run은 데이터 파일, 체크섬과 검증 기록을 확인합니다. 실제 모델 호환성은 모델 로드 후 확인됩니다.
현재 배정받은 GPU 한 장을 노출하고 기존 tmux 세션 안에서 학습합니다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python "01_학습용/train_lora.py" \
  --model /path/to/team_baseline \
  --output runs/theory_v2 \
  --epochs 2 --lr 5e-5 --rank 16 --grad-accum 16 --seed 42
```

초기 설정은 rank 16, alpha 32, dropout 0.05, 학습률 5e-5, micro batch 1,
gradient accumulation 16, 2 epochs, 최대 길이 1,536입니다.
개발셋 결과에 따라 설정과 체크포인트를 선택합니다.

학습 코드는 언어모델 attention의 q/k/v/o projection에 LoRA를 적용하고 assistant 답변에만 손실을 계산합니다.
학습 입력은 `01_학습용/데이터/train.jsonl`, 개발 입력은 `dev_loss.jsonl`입니다.
데이터 버전과 검증 결과는 `03_기타/검증기록/`에서 자동으로 확인합니다.
완료 후 어댑터와 실행 설정은 지정한 `runs/theory_v2/`에 저장됩니다.

## 4. 개발셋으로 전후 비교

```bash
CUDA_VISIBLE_DEVICES=0 python "02_평가용/predict.py" \
  --model /path/to/team_baseline \
  --questions "02_평가용/데이터/dev_questions.jsonl" --output runs/dev_base.jsonl

CUDA_VISIBLE_DEVICES=0 python "02_평가용/predict.py" \
  --model /path/to/team_baseline --adapter runs/theory_v2/adapter \
  --questions "02_평가용/데이터/dev_questions.jsonl" --output runs/dev_lora.jsonl

python "02_평가용/score.py" \
  --predictions runs/dev_lora.jsonl --baseline runs/dev_base.jsonl \
  --key "02_평가용/데이터/dev_answer_key.jsonl" --output runs/dev_comparison.json
```

텍스트 평가의 초기 생성 조건은 greedy, 최대 출력 256 tokens, seed 42입니다.
기본모델과 LoRA 모델의 조건을 동일하게 유지합니다.

## 5. 최종 평가

체크포인트와 생성 조건을 확정한 뒤 평가 담당자에게 어댑터와 `run_manifest.json`을 전달합니다.
위 명령의 개발셋 파일을 아래 최종 평가 파일로 바꿔 실행합니다.

- 질문: `02_평가용/데이터/test_questions.jsonl`
- 정답: `02_평가용/데이터/test_answer_key.jsonl`

`score.py`는 정확도, 범주별 점수, 파싱 실패율, 길이 제한 도달률, 문제군 단위 bootstrap 구간,
전후 개선/악화 문항 수와 McNemar 통계를 기록합니다.
이후 팀의 기존 평가 조건으로 MMMU Music과 전체 900문항 성능도 확인합니다.

## 6. 새 데이터 생성

새 버전은 `03_기타/생성버전/` 아래의 새 폴더로 생성합니다.
생성된 버전은 내부에 training/evaluation/audit 구조를 가지며 `--release`로 지정해 검증할 수 있습니다.

```bash
python "03_기타/도구/generate.py" --output "03_기타/생성버전/v2_next"
python "03_기타/도구/validate.py" --release "03_기타/생성버전/v2_next"
python "03_기타/도구/fetch_exclusions.py" --pilot-zip /path/to/TalkFile_pilot20.zip
python "03_기타/도구/screen.py" --release "03_기타/생성버전/v2_next"
python "03_기타/도구/check_tokenizer.py" \
  --release "03_기타/생성버전/v2_next" --model Qwen/Qwen3-VL-4B-Instruct
```

공개 평가 텍스트 캐시는 `03_기타/캐시/exclusion_cache/`에 저장됩니다.
검증과 검수가 끝난 새 버전은 다음 명령으로 역할별 ZIP을 생성할 수 있습니다.

```bash
python "03_기타/도구/package_release.py" \
  --release "03_기타/생성버전/v2_next" --output-dir "03_기타/배포결과/v2_next"
```

## 7. 현재 버전 공유

```bash
python "03_기타/도구/package_release.py"
```

`03_기타/배포결과/`에 다음 파일이 생성됩니다.

- `music_theory_text_v2_training.zip`: 학습 담당자용
- `music_theory_text_v2_evaluation.zip`: 평가 담당자용
- `music_theory_text_v2_pipeline.zip`: 관리자용 전체 자료

각 ZIP 안에서도 같은 폴더 구조를 유지합니다. 같은 파일명이 이미 있으면
`--output-dir`로 새 배포 폴더를 지정합니다.

## 8. 팀 보고 항목

데이터 버전과 체크섬, 기본모델과 revision, 학습 설정, 선택한 체크포인트,
텍스트 평가 전후 점수와 신뢰구간, 개선/악화 문항 수, 파싱 실패율과 길이 제한 도달률,
MMMU Music과 전체 점수, 전문가 검수 이력을 함께 기록합니다.
