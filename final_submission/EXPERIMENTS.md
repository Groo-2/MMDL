# 실험별 코드·문서·데이터 위치

이 저장소에서 한 모든 실험(SSMR, DeepVision, MusiXQA)이 어디에 있는지 정리한 색인이다. 제출 모델과 결과 요약은 [`README.md`](README.md), 실험을 한 이유와 원인 분석은 [`report/실험_정리.md`](report/실험_정리.md)에 있다.
**모든 명령은 `final_submission/code/`에서 실행한다.**

| 실험 | 데이터 | 방법 | 결과(MMMU 전체 Avg@2) |
|---|---|---|---|
| E1~E3 | SSMR-Bench 음악 | GRPO | 55.4 / 60.0 / 61.4% (CUDA 12.9, 같은 환경 base 62.83%) |
| E4 | DeepVision 수학 | RFT | 56.89% (seed 42 한 번) |
| E5 | DeepVision 수학 | DPO | 65.94% (원본), 65.11% (재학습), CUDA 12.9 / 65.17% CUDA 13 |
| E6 | DeepVision + MusiXQA | DPO | **65.89%** (CUDA 13, base 61.56%). **제출 모델** |
| E7 | MusiXQA | GRPO(팀원) | 64.33% (CUDA 13) |

## 1. SSMR (E1~E3, GRPO)
| 무엇 | 위치 |
|---|---|
| 데이터 변환(SSMR-Bench Train_visual) | `code/music_ft/build_ssmr.py` |
| 후보 선별(base로 문항당 8번 풀이, 통과율 1/8~7/8만) | `code/music_ft/select_ssmr.py` |
| 보상 함수와 테스트 | `code/music_ft/reward.py`, `code/music_ft/test_reward.py` |
| GRPO 학습(LoRA, rollout 8, KL과 잘림 제외 옵션) | `code/music_ft/train_grpo.py`, 단계 실행은 `code/lab/run_pipeline.sh` |
| SSMR hold-out 평가 | `code/lab/eval_holdout.py` |
| 결과 보고서 | `code/reports/ssmr_grpo_run1/README.md`, `report/실험_정리.md`의 E1~E3 |
| 학습 곡선 | `report/data/training_curve_grpo_run1.txt`, `report/figures/` |

설정: run1은 스텝당 8문항 × 140스텝(KL 없음), run2는 8문항 × 60스텝(잘림 제외, KL β 0.04), run3는 32문항 × 15스텝. SSMR hold-out은 18.25% → 47.25%(E1)였지만 MMMU는 하락했다.

## 2. DeepVision (E4 RFT, E5 DPO)
| 무엇 | 위치 |
|---|---|
| 데이터 변환(DeepVision-103K math-77k, 영어·이미지 1장·객관식 또는 숫자 정답만) | `code/music_ft/build_general.py` |
| MMMU와의 중복 제거(텍스트 유사도, 이미지 pHash) | `code/music_ft/dedupe.py` |
| base로 문항당 8번 풀이(응답 저장) | `code/music_ft/select_ssmr.py --save-samples` |
| RFT | `code/music_ft/build_rft.py`, `code/music_ft/train_rft.py` |
| DPO 쌍 구성(좋은 응답: 맞고 끝난 것, 나쁜 응답: 잘린 응답 60% 우선) | `code/music_ft/build_dpo.py` |
| DPO 학습(LoRA, 기준 모델은 어댑터를 끈 base) | `code/music_ft/train_dpo.py`, 중단 복구는 `code/music_ft/ckpt_utils.py` |
| 어댑터 병합 | `code/music_ft/merge_lora.py` |
| 학습 쌍 파일과 해시 | `data/dpo/dpo_dv.jsonl`(1,897쌍), `data/dpo/README.md` |
| 결과 보고서 | `code/reports/dpo_dv1/README.md`, `report/실험_정리.md`의 E4, E5 |

## 3. MusiXQA (E6 DPO, E7 GRPO)
| 무엇 | 위치 |
|---|---|
| **객관식 데이터 구성**(원본 주관식 → 4지선다: 음 높이, 조표, 마디 수, 음자리표) | `code/music_ft/build_musixqa.py`, `code/music_ft/build_musixqa_recipe.py`, `code/music_ft/musixqa_pilot20_recipe.json` |
| 표본 검토와 정정 | `code/music_ft/prepare_musixqa_review.py`, `package_musixqa_sample.py`, `apply_musixqa_sample_review.py`, `apply_musixqa_pilot20_correction.py`, `finalize_musixqa_review.py`, `record_musixqa_review_status.py`, `inspect_pilot20_report.py` |
| 검증 | `code/lab/validate_mcq_dataset.py`, `code/music_ft/check_musixqa_internal_overlap.py`(학습과 hold-out의 내부 중복) |
| 요청서와 실행 문서 | `code/lab/MusiXQA_객관식_요청서.md`, `code/lab/RUNBOOK_musixqa_data.md`, `code/lab/RUNBOOK_musixqa.md` |
| 구성 근거 보고서 | `code/reports/musixqa_pilot20_recipe.md`, `musixqa_pilot20_review.md`, `musixqa_sample_review.md`, `pilot20_recipe_evidence.json` |
| 필요한 패키지(데이터 준비용) | `code/music_ft/requirements_musixqa.txt` |
| 테스트 | `code/music_ft/test_build_musixqa.py`(8개), `test_musixqa_recipe.py`(6개) |
| 데이터(학습 3,200 + hold-out 400) | GitHub 릴리스 `musixqa-mcq-pilot20-v1`(MIT, 원본 puar-playground/MusiXQA) |
| 제출 모델에 쓴 쌍 | `data/dpo/dpo_mx.jsonl`(1,052쌍), 섞은 학습 파일은 `mix_dpo.py`로 만들며 해시는 `data/dpo/README.md` |
| E6(최종 모델) 학습 | `code/music_ft/mix_dpo.py` → `train_dpo.py` → `merge_lora.py` (순서는 `README.md`) |

- 객관식 데이터 구성 코드(`build_musixqa.py` 등)의 작성자는 **choe, kwan sik**이다(원 커밋 `2ef60cb`, 브랜치 `musixqa-mcq-a`). 이 폴더에는 그 코드를 복사했고 내용은 바꾸지 않았다.
- 테스트(`test_build_musixqa.py`, `test_musixqa_recipe.py`)는 이 폴더의 최신 코드와 함께 통과를 확인했다(`PYTHONPATH=music_ft python -m unittest test_build_musixqa test_musixqa_recipe`, `pip install -r music_ft/requirements_musixqa.txt` 필요).
- 원 브랜치에 있던 `test_musixqa_preparation.py`는 **넣지 않았다.** 예전 `dedupe.py`(MMMU Music만 검사하는 `load_mmmu_music`)를 시험하는 파일인데, 이 폴더의 `dedupe.py`는 MMMU 전 과목을 대조하도록 바뀌어서 그 함수가 없다. 데이터 구성 단계의 중복 검사 범위가 Music뿐이었다는 점은 `lab/RUNBOOK_musixqa_data.md`에도 적혀 있다.
- **E7(팀원의 MusiXQA GRPO)의 학습 코드와 설정은 이 저장소에 없다.** 평가 결과만 `report/`에 있고, rollout 수와 최대 응답 길이는 기록이 없다.

## 저장소에 없는 것
| 항목 | 이유 |
|---|---|
| 모델 가중치(어댑터, 합친 모델) | 과제 규칙. 제출 모델의 해시는 `README.md` |
| 응답 단위 평가 결과(`final_predictions.jsonl` 등) | 용량이 크다(모델당 수십 MB) |
| 이미지(DeepVision, MusiXQA) | 용량과 원본 라이선스. 쌍 파일에는 이미지 경로만 있다 |

## 다른 브랜치
- `music_reading`, `music_resolution`, `music_theory`, `qwen-mmmu-62-44`, `baseline_final`은 초기 파이프라인과 baseline 작업이다. 이 폴더에는 옮기지 않았다.
- `ssmr`, `musixqa`, `musixqa-mcq-a` 브랜치는 개발 이력이고, 최종 코드는 이 폴더에 모아 두었다.
