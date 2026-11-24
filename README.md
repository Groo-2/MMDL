# MMDL: Qwen3-VL-4B-Instruct의 MMMU / MMMU-Pro 성능 개선

멀티모달 딥러닝 팀 프로젝트. 기본 모델 `Qwen/Qwen3-VL-4B-Instruct`(revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16)를 파인튜닝해서 MMMU와 MMMU-Pro 성능을 baseline보다 높인다.
**평가 원칙**: MMMU validation으로 조정하고, MMMU test와 MMMU-Pro test로 최종 평가한다. MMMU dev는 학습과 평가에 쓰지 않았다(학습 데이터와의 중복 제거 대조에만 썼다). 모든 학습 데이터는 MMMU, MMMU-Pro와 중복을 제거했다.

## 저장소 구성
| 경로 | 내용 |
|---|---|
| [`assignment/assignment1.md`](assignment/assignment1.md) | **과제 1: baseline 평가 보고서** (환경, 프롬프트, 생성 설정, 채점 방식, 과목별 결과, 공식 수치와의 비교, 격차 분석) |
| [`baseline/`](baseline/) | **baseline 평가 코드**(평가 파이프라인). 실행 방법은 [`baseline/README_mmmu_eval.md`](baseline/README_mmmu_eval.md), baseline 결정 과정은 [`baseline/reports/baseline_decision.md`](baseline/reports/baseline_decision.md) |
| [`reports/mmmu_baseline.md`](reports/mmmu_baseline.md) | 과제 1 보고서 원본(`assignment/assignment1.md`와 같은 내용) |
| [`final_submission/`](final_submission/README.md) | **최종 제출(파인튜닝)**: 파인튜닝 코드와 평가 코드(`code/`), 실험 설정과 결과 문서(`report/`), 학습 쌍(`data/dpo/`), 재현 절차([`REPRODUCE.md`](final_submission/REPRODUCE.md)), 실험별 위치 색인([`EXPERIMENTS.md`](final_submission/EXPERIMENTS.md)) |

## 결과 요약 (MMMU validation 900문항, 같은 환경에서 seed 42·43 평균)
| 모델 | 전체 | Music(30문항) |
|---|---|---|
| baseline (Qwen3-VL-4B-Instruct) | 61.56% | 18.3% |
| **제출 모델** (DeepVision 수학 쌍 + MusiXQA 악보 쌍으로 DPO) | **65.89%** (+4.33%p, 95% 구간 [+2.44, +6.22]) | **43.3%** (+25.0%p, [+10.0, +40.0]) |

- 이득은 base가 32,768토큰 안에 끝내지 못하던 문항(반복 루프)이 답을 내게 된 데서 온다. 끝낸 문항의 정확도는 그대로여서 새 지식이 늘었다는 증거는 없다.
- 여러 후보를 validation 점수로 비교해 골랐으므로 선택 편향이 있다. 자세한 해석과 한계는 [`final_submission/README.md`](final_submission/README.md).
- **MMMU test와 MMMU-Pro 평가는 코드를 준비했고 실행 중이다.** 결과가 나오면 [`final_submission/README.md`](final_submission/README.md)에 추가한다.

## 재현
모델 가중치는 저장소에 없다. 모두 코드와 데이터 변환 절차, 학습 쌍 파일로 다시 만든다. `HF_TOKEN`(judge Llama-3.1-8B-Instruct가 gated 모델)은 환경변수에만 두고 커밋하지 않는다.

**1. baseline 평가(과제 1)** — 모든 명령은 `baseline/` 폴더에서 실행:
```bash
cd baseline
pip install -r requirements.txt
export HF_TOKEN=...

bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root <MMMU HF 캐시 디렉터리> \
  --output_dir ./results/baseline_mmmu_pro_cot
```
**2. 제출 모델 만들기와 평가(환경 → 데이터 → 학습 → 병합 → 평가)**: [`final_submission/REPRODUCE.md`](final_submission/REPRODUCE.md). 모든 명령은 `final_submission/code/`에서 실행한다. 단계마다 확인용 해시와 기대 결과가 적혀 있다.

**3. MMMU test와 MMMU-Pro 평가**: [`final_submission/REPRODUCE.md`](final_submission/REPRODUCE.md) 9절(`bash lab/eval_extra.sh check|smoke|plan|run`).

## 실험 하나하나가 어디에 있나
SSMR(GRPO), DeepVision(RFT, DPO), MusiXQA(DPO, GRPO) 실험의 코드, 문서, 데이터, 결과 위치는 [`final_submission/EXPERIMENTS.md`](final_submission/EXPERIMENTS.md)에 정리했다. 실험을 한 이유와 원인 분석은 [`final_submission/report/실험_정리.md`](final_submission/report/실험_정리.md).

## 브랜치
최종 결과는 모두 기본 브랜치(`main`)에 있다. 아래는 개발 이력이다.
- `ssmr`, `musixqa`, `musixqa-mcq-a`: 파인튜닝 개발 브랜치. 최종 코드는 `final_submission/code/`에 모아 두었다.
- `qwen-mmmu-62-44`(처음 평가한 GitHub 원본 스크립트와 응답), `baseline_final`(`baseline/`로 옮기기 전 구조의 파이프라인 스냅샷), `music_reading`, `music_resolution`, `music_theory`: 초기 실험과 팀원 작업.
