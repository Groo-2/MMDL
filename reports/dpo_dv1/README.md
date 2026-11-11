# dpo_dv1: 제출 후보 모델 (DeepVision 수학 데이터로 DPO 학습한 Qwen3-VL-4B-Instruct LoRA)

작성 2026-11-11. 이 문서는 모델을 처음부터 다시 만들고 평가하는 방법과, 지금까지 확인된 것과 아직 확인되지 않은 것을 함께 적는다.
가중치와 큰 데이터는 저장소에 올리지 않는다(과제 안내: 팀 저장소에 가중치 커밋 금지). 실습실의 보관 폴더에 있고 `lab/archive_model.sh`로 만든다.

## 결과 (MMMU validation 900문항, 동결 평가 파이프라인)

| 항목 | base (실습실) | dpo_dv1 | 상태 |
|---|---|---|---|
| 전체 정확도, seed 42 | 62.33% (561) | **65.00% (585)** | 확인됨 (한 seed) |
| 잘림 (32,768토큰) | 89 | 33 | 확인됨 |
| 요청 오류 | 0 | 0 | 확인됨 |
| Music, seed 42 | 7/30 (23.3%) | 14/30 (46.7%) | 한 seed, 30문항 |
| 전체, seed 43 | 63.33% (570) | **66.89% (602)**, 잘림 26, 오류 0 | 확인됨 (+3.56%p) |
| **전체 Avg@2 (seed 42, 43)** | 62.83% | **65.94%** (±1.34) | **확인됨 (+3.11%p, 두 seed 모두 base 초과)** |
| Music, seed 43 | 7/30 | 10/30 | 확인됨. Music Avg@2 40.0% (base 23.3%) |
| Music Avg@4 | 25.8% | **(측정 전)** | 미확인 |
| 4과목 진단(Electronics, Materials, Mechanical, Music) 정답/루프 | 47 / 31 | 53 / 14 | 진단용. 최종 선택에 쓰지 않음 |

- 단일 run의 변동은 약 1.2%p이고 Music 30문항은 변동이 더 크다. **두 seed 모두 base를 넘었고 Avg@2 +3.11%p**(차이의 표준오차 약 1.1%p)이다. Music Avg@4와 문항 단위 신뢰구간은 아직 없다.
- seed 43의 첫 시도에서 요청 842건이 시간 초과(서버가 응답하지 못함, 원인 미확인)로 실패했고 같은 명령으로 재시도해 정상 완료했다. 첫 시도의 결과는 쓰지 않았다.
- **선택 편향 주의**: 이 모델에 이르기까지 GRPO 4종, RFT 2종을 MMMU로 평가했다(8번째 모델). 가장 좋은 것을 고르면 점수가 부풀 수 있다. MMMU test/MMMU-Pro 같은 **선택에 쓰지 않은 데이터**로 확인하는 것이 가장 강한 점검이며 아직 하지 않았다.

## 무엇을 했나

1. **데이터**: DeepVision-103K(`skylenage/DeepVision-103K`, MIT, 2026-02-14)의 `math-77k.parquet`만 사용. `test/` 폴더(MMMU 평가 사본 포함)는 쓰지 않았다.
2. **변환** (`music_ft/build_general.py`): 영어, 이미지 1장, 보기 글자가 있는 객관식과 숫자 정답 주관식만. 77,135 → 변환 가능 54,577 → 풀 12,000.
3. **MMMU 중복 제거** (`music_ft/dedupe.py --subjects all`): MMMU 30과목 dev/validation/test 11,550문항, 이미지 13,281개와 대조. 12,000 → 11,995(텍스트 유사 4, 완전 일치 1, 이미지 중복 0).
4. **base로 문항당 8번 풀이** (`music_ft/select_ssmr.py --save-samples`): 풀 중 6,000문항, 최대 2,048토큰, 온도 0.7/top_p 0.8/top_k 20, seed 42. 통과율 1/8~7/8인 문항 1,908개(0/8 41.2%, 8/8 27.0%). 응답 45.3%가 2,048토큰에서 잘림.
5. **DPO 쌍** (`music_ft/build_dpo.py`): 문항마다 좋은 응답(맞고 끝난 것, 반복 줄 제외)과 나쁜 응답(잘린 응답 60% 우선, 없으면 틀린 응답). 쌍 약 1,900개. (정확한 개수는 `data/dpo/dpo_dv.report.json`)
6. **DPO 학습** (`music_ft/train_dpo.py`): LoRA r64/alpha128(LLM의 q,k,v,o,gate,up,down), lr 5e-6 cosine, beta 0.1, sft-alpha 0.1(좋은 응답 토큰당 NLL), 누적 16쌍, 1에폭, seed 42, 기준 모델은 어댑터를 끈 base. 약 1.4시간(4,998초).
7. **합치기** (`music_ft/merge_lora.py --low-memory`): 어댑터를 bf16 base에 합쳐 bf16으로 저장.
8. **평가** (`lab/eval_repeat.py`): 동결 프롬프트(`mmmu_pro_cot`), Qwen 공식 sampling recipe, 최대 32,768토큰, 규칙 파서 → Llama-3.1-8B-Instruct judge(`0e9e39f249a16976918f6564b8830bc894c89659`) → 오답.

## 왜 이렇게 했나 (경과)
- GRPO(SSMR 음악 데이터)는 네 가지 설계 모두 MMMU 전체를 올리지 못했다(최선이 base와 통계적으로 구분 불가).
- RFT(맞은 응답만 지도학습, DeepVision)는 **반복·재고 루프를 늘려**(잘림+오류 89 → 209) 56.89%로 하락했다. 학습률을 3배 낮춰도 루프는 거의 그대로(4과목 진단 72 → 61)여서 학습률이 원인이 아니었다. 길이 편향과 재고 표현 증폭 가설은 데이터로 기각됐다.
- 원인으로 보는 것(가설): 맞은 응답만 학습해서 **끝나지 않는 풀이를 벌하는 신호가 없었다.** DPO는 잘린 루프 응답을 나쁜 쪽으로 명시해 이를 겨냥했고, 루프가 크게 줄었다.

## 재현 방법 (실습실, 저장소 루트, mmdl 환경)
환경은 `lab/setup_env.sh`와 `lab/lab_pins.txt`(torch 2.13.0+cu129, vLLM 0.27.1+cu129, transformers 5.19.0, TRL 1.16.0, PEFT 0.21.2, accelerate 1.15.0, datasets 5.1.0, numpy 2.3.5). 코드 버전은 git 태그 `dpo_dv1-code`(커밋 `57a2b8b`).

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
python - <<'EOF'                                    # 1) 데이터 받기
from huggingface_hub import hf_hub_download
hf_hub_download("skylenage/DeepVision-103K", "math-77k.parquet", repo_type="dataset", local_dir="data/general/deepvision")
EOF
python music_ft/build_general.py --parquet data/general/deepvision/math-77k.parquet --out-dir data/general/dv_pool --pool-size 12000
python music_ft/dedupe.py --data-dir data/general/dv_pool --subjects all --names train --cache-dir data/hf_cache --ref-cache data/mmmu_ref_all.pkl
python music_ft/select_ssmr.py --data-dir data/general/dv_pool --train-file train.dedup.jsonl \
  --pool-size 6000 --per-category 100 --max-images 1 --save-samples --start-server
python music_ft/build_dpo.py --source dv,data/general/dv_pool,train.dedup.jsonl --out data/dpo/dpo_dv.jsonl
python music_ft/train_dpo.py --data data/dpo/dpo_dv.jsonl --output-dir results/dpo_dv1 --epochs 1 --resume
python music_ft/merge_lora.py --lora results/dpo_dv1/final_lora --out results/dpo_dv1/merged --low-memory
python lab/eval_repeat.py --tag dpo_dv1 --model results/dpo_dv1/merged --runs 1                      # seed 42
python lab/eval_repeat.py --tag dpo_dv1 --model results/dpo_dv1/merged --runs 1 --run-start 1       # seed 43
```
- 응답 생성은 vLLM의 샘플링이라 선별과 쌍 구성은 같은 seed여도 **비트 단위로 같지 않을 수 있다.** 보관본의 `data/dpo/dpo_dv.jsonl`(쌍 자체)과 어댑터가 정확한 재현의 기준이다.
- 하드웨어: RTX 4090 24GB, WSL2(드라이버 560.94, CUDA 12.6), WSL 메모리 15GiB.

## 보관 (실습실)
```bash
bash lab/archive_model.sh dpo_dv1        # ~/mmdl/archive/dpo_dv1_<날짜>/ (어댑터, 합친 모델, 결과, 파생 데이터, 로그, 환경, 체크섬)
```
정밀도: 기본 모델 bf16, 학습은 bf16 혼합정밀도, LoRA 어댑터는 fp32로 저장, 손실의 softmax만 fp32, 평가용 모델은 어댑터를 bf16 base에 합쳐 bf16으로 저장(합친 모델의 dtype은 `{torch.bfloat16}`로 확인).

## 한계와 주의
- 한 데이터(DeepVision 수학, K12)만 사용했고 SSMR, MusiXQA, ViRL39K는 쓰지 않았다. Music 개선은 루프 감소의 부수 효과일 가능성이 있으며 음악 데이터의 효과가 아니다.
- DeepVision 공개일(2026-02-14)은 Qwen3-VL(2025-10-11) 이후라 사전학습 노출 가능성은 낮다고 본다.
- 응답 생성과 판정이 sampling이라 같은 모델도 seed마다 약 1.2%p 달라진다.
