# 최종 제출: Qwen3-VL-4B-Instruct 파인튜닝 (과제 2)

baseline(과제 1)은 최상위 [`baseline/`](../baseline/)과 [`reports/mmmu_baseline.md`](../reports/mmmu_baseline.md)에 그대로 있다. 이 폴더는 **그 위에 한 파인튜닝의 최종 제출물**(코드, 결과 문서, 제출 모델 정보)이다.
**목표**: MMMU validation 전체 정확도가 base보다 높고 Music(30문항)도 base보다 높은 모델.

## 폴더 구성
| 경로 | 내용 |
|---|---|
| [`report/`](report/) | 결과 문서. [`실험_정리.md`](report/실험_정리.md)(실험별 이유, 벤치마크 선택 이유, 파라미터, 원인 분석, 학습 방식의 기본 개념, 한계), `figures/`(그림 11개), `tables/`(표 9개), `data/`, `scripts/make_all.py` |
| [`code/`](code/) | 학습·선별·평가 코드와 실행 문서. 아래 "코드" 절 참고 |

## 제출 모델: DPO (DeepVision 수학 쌍 + MusiXQA 악보 쌍): DPO (DeepVision 수학 쌍 + MusiXQA 악보 쌍)

| 항목 | 값 |
|---|---|
| 기본 모델 | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| 방법 | DPO(+좋은 응답 NLL 보조항), LoRA r=64 / α=128(LLM의 q,k,v,o,gate,up,down) |
| 학습 데이터 | DeepVision 쌍 1,897 + MusiXQA 쌍 1,052 = 2,949쌍 |
| 학습 | lr 5e-6 cosine, β 0.1, 누적 16쌍, 1에폭, 185스텝, 8,321초 |
| 어댑터 파일 | `adapter_model.safetensors` 528,557,816바이트(fp32, 텐서 504개) |
| **어댑터 SHA-256** | `4492c406fe29110a3c7b219890ee80d50bf1981b82956cedea27a00887acd131` |
| 합친 모델(bf16) | `model.safetensors` 8,875,719,408바이트, SHA-256 `358d4a085d91b671aa063947183ea99f9b5968ed523ccc79bf8f2615e0df087b` |

가중치는 과제 규칙에 따라 이 저장소에 올리지 않는다. 위 해시로 별도 전달된 파일의 무결성을 확인한다.

## 결과 (MMMU validation 900문항, cu130 환경에서 모든 모델을 같은 조건으로 측정, seed 42·43)

| 모델 | 전체 Avg@2 | base 대비 (95% 신뢰구간) | Music Avg@2 | base 대비 Music |
|---|---|---|---|---|
| base | 61.56% | - | 18.3% | - |
| MusiXQA GRPO (팀원) | 64.33% | +2.78%p [+0.67, +4.89] | 33.3% | +15.0%p [+5.0, +26.7] |
| DPO (DeepVision) | 65.17% | +3.61%p [+1.72, +5.50] | 31.7% | +13.3%p [-1.7, +30.0] |
| **DPO (DeepVision+MusiXQA, 제출)** | **65.89%** | **+4.33%p [+2.44, +6.22]** | **43.3%** | **+25.0%p [+10.0, +40.0]** |

신뢰구간은 문항 단위 paired bootstrap(10,000회). 모든 모델의 요청 오류는 0이다.

**해석상 주의**
- 세 모델끼리의 전체 점수 차이(최대 1.6%p)는 통계적으로 구분되지 않는다. 제출 모델이 가장 높지만 "더 낫다"고 단정할 수 없다.
- 이득은 거의 전부 base가 32,768토큰 안에 끝내지 못하던 문항(루프)이 답을 내게 된 데서 온다. 끝내는 문항의 정확도는 그대로여서 **새 지식이 늘었다는 증거는 없다.**
- 여러 후보를 MMMU validation으로 비교하고 가장 높은 것을 골랐으므로 선택 편향이 있다. 선택에 쓰지 않은 데이터로는 확인하지 못했다.
- Music은 30문항이라 오차가 크다.

## 코드 (`code/`)
**모든 명령은 `final_submission/code/`에서 실행한다**(코드가 이 폴더를 저장소 루트로 가정한다). 이 코드는 `ssmr` 브랜치의 최신본과 같다.

| 경로 | 역할 |
|---|---|
| `mmmu_eval/` | 동결된 MMMU 평가(프롬프트, 파서, judge). 모든 결과가 이 코드로 측정됐다 |
| `lab/` | 평가 실행(`eval_repeat.py`, `eval_holdout.py`, `eval_queue.py`), 환경 설치와 점검(`setup_env.sh`, `env_triage.py`), 요약과 신뢰구간(`summarize_runs.py`), 중단 복구(`jsonl_repair.py`, `status.py`), 테스트(`test_lab.py`) |
| `music_ft/` | 데이터 변환·중복 제거(`build_general.py`, `build_musixqa.py`, `dedupe.py`), 응답 선별(`select_ssmr.py`), 쌍 구성(`build_dpo.py`), 학습(`train_dpo.py`, `train_grpo.py`, `train_rft.py`), 어댑터 병합(`merge_lora.py`) |
| `reports/` | 이전 실험(GRPO, DPO 첫 모델)의 상세 보고서 |
| `requirements.txt`, `scripts/` | baseline 평가 의존성과 진입 스크립트 |

**제출 모델을 다시 만드는 순서**
```bash
cd final_submission/code
FORCE_PLAN=cu129 CONDA_ENV=<env> bash lab/setup_env.sh                 # 환경(학습·선별은 cu129 환경에서 했다)
# 1) DeepVision 쌍: 변환 -> 중복 제거 -> base로 문항당 8번 풀이 -> 쌍 (README: report/실험_정리.md E5)
python music_ft/build_general.py --parquet <math-77k.parquet> --out-dir data/general/dv_pool --pool-size 12000
python music_ft/dedupe.py --data-dir data/general/dv_pool --subjects all --names train
python music_ft/select_ssmr.py --data-dir data/general/dv_pool --train-file train.dedup.jsonl --pool-size 6000 --max-images 1 --save-samples --start-server
python music_ft/build_dpo.py --source dv,data/general/dv_pool,train.dedup.jsonl --out data/dpo/dpo_dv.jsonl
# 2) MusiXQA 쌍: 릴리스 musixqa-mcq-pilot20-v1을 data/musixqa에 풀고 유형별로 선별 후 쌍
python music_ft/select_ssmr.py --data-dir data/musixqa --train-file train_recipe.reviewed.jsonl --pool-size 3200 --max-images 1 --save-samples --samples-only --start-server
python music_ft/build_dpo.py --source mx,data/musixqa,train_recipe.reviewed.jsonl --out data/dpo/dpo_mx.jsonl
# 3) 두 쌍 파일을 섞어서(이어 붙인 뒤 seed 42로 셔플) 학습, 병합, 평가
python music_ft/mix_dpo.py --inputs data/dpo/dpo_dv.jsonl data/dpo/dpo_mx.jsonl --out data/dpo/dpo_dv_mx.jsonl
python music_ft/train_dpo.py --data data/dpo/dpo_dv_mx.jsonl --output-dir results/dpo_dvmx1 --epochs 1 --save-steps 20 --save-total-limit 3 --resume
python music_ft/merge_lora.py --lora results/dpo_dvmx1/final_lora --out results/dpo_dvmx1/merged --low-memory
python lab/eval_queue.py dpo_dvmx1=results/dpo_dvmx1/merged:2          # MMMU 900문항 x seed 42, 43 (CUDA 13 환경 mmdl130)
```
- 샘플링 결과라 쌍을 새로 만들면 학습에 쓴 쌍과 같지 않다. **정확한 재현에는 학습에 쓴 쌍 파일이 필요하다.** 학습에 쓴 파일의 해시:

  | 파일 | 쌍 수 | SHA-256 |
  |---|---|---|
  | `dpo_dv.jsonl` (DeepVision 쌍) | 1,897 | `b2419a08c1228bfc626e5d5daa6ad22d5a98ea85c812f679a42d68327ac6ac25` |
  | `dpo_mx.jsonl` (MusiXQA 쌍) | 1,052 | `d0937312c0c3898cf35fa8789706ec2989e8612601e26ce6f5d7c2779563e5da` |
  | `dpo_dv_mx.jsonl` (섞은 학습 파일) | 2,949 | `a1b386dd6ff753383b26b5c83a1edba03b9b3e54594da26ae298e2d2c233b147` |

  `mix_dpo.py`는 위 두 입력 파일에서 학습에 쓴 섞은 파일과 **바이트 단위로 같은 파일**을 만든다(2026-11-18 실습실에서 `cmp`로 확인). 나쁜 응답 구성은 잘린 응답 2,032 / 틀린 응답 917.
- 학습 로직은 첫 DPO 모델의 코드(태그 `dpo_dv1-code`)와 같고, 이후 추가된 것은 중단 복구(불완전 체크포인트 건너뛰기, 이어쓰기 전 깨진 줄 복구)뿐이다.
- 코드 테스트: `python -m unittest lab.test_lab` (GPU 불필요, 78개).
- 환경 설치와 점검 절차는 `lab/RUNBOOK_resume.md`, `lab/README_lab.md`, `lab/RUNBOOK_실습실_명령어.md`.

## 재현할 수 있는 범위와 한계
- 코드, 쌍·선별 결과, 평가 결과, 해시는 제출물에 포함되거나 위 문서에 적혀 있다. 모델 가중치는 과제 규칙에 따라 저장소에 올리지 않았다.
- 학습과 응답 선별은 CUDA 12.9(torch 2.13.0+cu129, transformers 5.19.0), 최종 MMMU 평가는 CUDA 13(torch 2.13.0, vLLM 0.27.1, transformers 5.22.2)에서 했다. 같은 모델을 두 환경에서 평가했을 때 점수는 같았다(`report/figures/fig8_env_robustness`).
- MusiXQA 객관식 데이터는 `musixqa-mcq-pilot20-v1` 릴리스(MIT, 원본 puar-playground/MusiXQA)를 썼고, 학습과 hold-out은 겹치지 않는다.
- 데이터 공개일 주의: SSMR-Bench(2025-09-06)와 MusiXQA(2025-07-07)는 Qwen3-VL(2025-10-11)보다 앞서 공개되어 모델이 이미 봤을 수 있다. DeepVision(2026-02)은 이후다. MMMU validation과 test는 학습에 쓰지 않았고 모든 외부 데이터는 MMMU와 중복을 제거했다.
