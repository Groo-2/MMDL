# 파인튜닝 결과: Qwen3-VL-4B-Instruct의 MMMU 성능 향상 (과제 2)

baseline(과제 1, [`../mmmu_baseline.md`](../mmmu_baseline.md))을 기준으로, 모델을 고정한 채 LoRA 파인튜닝으로 MMMU validation 점수를 올린 실험의 정리다.
**목표**: MMMU validation 전체 정확도가 base보다 높고 Music(30문항)도 base보다 높은 모델.

## 제출 모델: DPO (DeepVision 수학 쌍 + MusiXQA 악보 쌍)

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

## 문서
| 파일 | 내용 |
|---|---|
| [`실험_정리.md`](실험_정리.md) | 실험별(GRPO 3종, RFT 2종, DPO, 제출 모델, 팀원 GRPO) 이유, 벤치마크 선택 이유, 학습 방법과 파라미터, 성능이 오른/오르지 않은 원인 분석, 학습 방식의 기본 개념, 한계 |
| `figures/` | 그림 11개(png, 벡터 pdf) |
| `tables/` | 표 9개(md, csv, tex) |
| `data/` | 과목별 결과, 악보 hold-out, GRPO run1 학습 지표 |
| `scripts/make_all.py` | 원본 결과에서 표와 그림을 다시 만드는 스크립트 |

## 코드와 재현
- 학습·선별·평가 코드와 실행 문서는 이 저장소의 **`ssmr` 브랜치**에 있다(`music_ft/`, `lab/`, `reports/dpo_dv1/README.md`, `reports/ssmr_grpo_run1/`). 첫 DPO 모델의 코드 버전은 태그 `dpo_dv1-code`.
- 평가 환경: 학습과 응답 선별은 CUDA 12.9(torch 2.13.0+cu129, transformers 5.19.0), 최종 MMMU 평가는 CUDA 13(torch 2.13.0, vLLM 0.27.1, transformers 5.22.2)에서 했다. 같은 모델을 두 환경에서 평가했을 때 점수는 같았다(`figures/fig8_env_robustness`).
- MusiXQA 객관식 데이터는 `musixqa-mcq-pilot20-v1` 릴리스(MIT, 원본 puar-playground/MusiXQA)를 썼고, 학습과 hold-out은 겹치지 않는다.
- 데이터 공개일 주의: SSMR-Bench(2025-09-06)와 MusiXQA(2025-07-07)는 Qwen3-VL(2025-10-11)보다 앞서 공개되어 모델이 이미 봤을 수 있다. DeepVision(2026-02)은 이후다. MMMU validation과 test는 학습에 쓰지 않았고 모든 외부 데이터는 MMMU와 중복을 제거했다.

## 표와 그림 다시 만들기
```bash
python scripts/make_all.py --work <결과를 푼 폴더> --out . --base130 <base 최종 결과 폴더>
```
`--work` 아래에 필요한 결과 폴더 목록은 스크립트 첫머리의 설명을 따른다.
