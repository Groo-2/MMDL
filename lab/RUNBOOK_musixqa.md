# E2: MusiXQA 객관식으로 base 모델 학습 (실습실 RTX 4090)

팀원이 만든 MusiXQA 객관식 데이터(SSMR 변환 결과와 같은 형식)가 도착한 뒤의 절차다. **base에서 새로 학습**한다(SSMR 모델 위가 아님. 이유는 reports/ssmr_grpo_run1/HANDOFF_musixqa.md의 C절).
설정은 E1''(잘림 응답 손실 제외 + KL + 최대 2,048토큰 + 60스텝 안팎)를 따른다. 파이프라인은 `DATA_DIR`, `TAG_NEW`, `TAG_BASE`, `PER_CAT`, `POOL`, `STEPS`로 데이터와 결과 폴더를 바꾼다.
이 문서의 절차는 MusiXQA 실데이터로 돌려 본 적이 없다(가짜 데이터로 선별·변환 로직만 Mac에서 확인).

## 0. 데이터 위치와 이름
```
data/musixqa/train.jsonl, holdout.jsonl, images/        (git에 올리지 않는다. data/는 .gitignore)
```
결과 폴더가 SSMR 것과 섞이지 않게 태그를 바꾼다: 기준 `TAG_BASE=baseline_mx`(hold-out 전용), 학습 모델 `TAG_NEW=musixqa_run1`.
MMMU 평가(`baseline_music4`, `baseline_full2`)의 기준은 기존 `baseline`을 그대로 쓴다.

## 1. 검증과 중복 제거 (GPU 불필요)
```bash
git fetch origin musixqa && git show origin/musixqa:lab/validate_mcq_dataset.py > /tmp/validate_mcq_dataset.py
python /tmp/validate_mcq_dataset.py --data-dir data/musixqa           # ERROR 0개여야 한다
python music_ft/dedupe.py --data-dir data/musixqa --cache-dir data/hf_cache    # MMMU Music(val/test 포함)과 겹치는 문항 제거
```
- 질문은 반드시 `<image 1> `로 시작해야 한다. 빠지면 이미지가 프롬프트에서 사라지는데, 학습·선별·평가 코드가 이제 이 경우 오류를 낸다.
- 선택지는 4개(A~D). 보상(`music_ft/reward.py`)이 4지선다를 가정한다.
- `dedupe`가 `holdout.jsonl`에서 문항을 뺐다면 `cp data/musixqa/holdout.dedup.jsonl data/musixqa/holdout.jsonl`로 바꾼다(hold-out 평가는 `holdout.jsonl`을 읽는다).

## 2. base의 hold-out 점수 (기준선)
```bash
DATA_DIR=data/musixqa TAG_BASE=baseline_mx bash lab/run_pipeline.sh holdout_base 2>&1 | tee logs/mx_holdout_base.log
```

## 3. 문항 선별 (base로 문항당 8번 풀어 통과율 1/8~7/8만)
카테고리 수 n에 대해 `PER_CAT = (STEPS x 8) / n`이다(예: 60스텝, 6개 카테고리면 80). `POOL`은 통과율 필터를 거치고도 남도록 목표의 4배 안팎으로 잡는다.
```bash
DATA_DIR=data/musixqa STEPS=60 PER_CAT=80 POOL=2000 bash lab/run_pipeline.sh screen 2>&1 | tee logs/mx_screen.log
cat data/musixqa/select_report.json        # 카테고리별 적격 수, 부족 메모를 확인
```
SSMR에서는 적격률이 카테고리마다 27%~59%로 달랐다. 부족한 카테고리가 있으면 다른 카테고리로 채워지고 `notes`에 남는다.

## 4. 메모리와 스텝 시간 (풀페이지 악보는 이미지 토큰이 약 1,960개로 SSMR의 약 2배, 추정)
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export GRPO_EXTRA="--beta 0.04 --mask-truncated --save-total-limit 12 --empty-cache-every 1"
DATA_DIR=data/musixqa PROMPTS_PER_STEP=1 MAX_COMPLETION=2048 bash lab/run_pipeline.sh stress 2>&1 | tee logs/mx_stress.log
```
SSMR 1장 문항은 12.8GB, 161~168초였다. 이번 값이 크게 다르면(예약이 24GB에 가까워지거나 프롬프트 1개에 10분 이상) 학습을 시작하지 말고 알린다.
최대 길이를 4,096으로 늘리지 않는다(WSL2에서 예약 33GB, 프롬프트 1개 30분으로 실패한 이력).

## 5. 학습
```bash
DATA_DIR=data/musixqa TAG_NEW=musixqa_run1 STEPS=60 SAVE_STEPS=10 MAX_COMPLETION=2048 \
  bash lab/run_pipeline.sh grpo 2>&1 | tee logs/musixqa_run1.log
```
볼 것(E1''의 경험): `completions/mean_terminated_length`가 200토큰대로 무너지지 않는지, `kl`이 천천히 오르는지, 스텝 시간. 10~20스텝에서 무너지면 멈추고 설계를 다시 본다.

## 6. 평가 (모두 `TAG_NEW=musixqa_run1`)
```bash
TAG_NEW=musixqa_run1 bash lab/run_pipeline.sh merge 2>&1 | tee logs/mx_merge.log
DATA_DIR=data/musixqa TAG_NEW=musixqa_run1 bash lab/run_pipeline.sh holdout_new 2>&1 | tee logs/mx_holdout_new.log
TAG_NEW=musixqa_run1 bash lab/run_pipeline.sh music4 full2 2>&1 | tee logs/mx_mmmu.log
```
비교 기준(실습실 base, 같은 조건): MMMU 전체 Avg@2 62.83%, Music Avg@4 25.8%. SSMR 단계 결과(E1 55.39%, E1'' 60.00%)와도 나란히 비교한다.
Music 4번을 8번으로 늘리는 것은 사용자 확인 후에만 한다.
