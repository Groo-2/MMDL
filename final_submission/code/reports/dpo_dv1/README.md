# dpo_dv1: 제출 후보 모델 (DeepVision 수학 데이터로 DPO 학습한 Qwen3-VL-4B-Instruct LoRA)

작성 2026-11-11, 갱신 2026-11-15(재학습 결과 추가). 이 문서는 모델을 처음부터 다시 만들고 평가하는 방법과, 지금까지 확인된 것과 아직 확인되지 않은 것을 함께 적는다.
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

## 재학습 결과 (dpo_dv1_re, 2026-11-15, 새 머신)

원본 어댑터를 로컬로 옮기는 중에 파일 끝 150MiB가 0으로 채워져 손상됐다(크기와 구조는 정상이라 길이만 봐서는 알 수 없었고, 0으로만 찬 블록 검사로 발견). 그래서 **같은 쌍 파일(`dpo_dv.jsonl`, SHA-256 `b2419a08…6ac25`)과 같은 설정**으로 새 머신에서 다시 학습하고 평가했다. 쌍을 다시 만들지 않았다(샘플링 결과라 같지 않다).

| 항목 | base | dpo_dv1 (원본) | **dpo_dv1_re (재학습)** |
|---|---|---|---|
| seed 42 | 62.33% | 65.00% | **65.11%** (586) |
| seed 43 | 63.33% | 66.89% | **65.11%** (586) |
| **전체 Avg@2** | 62.83% | 65.94% | **65.11%** |
| 잘림 (seed 42 / 43) | 89 / 98 | 33 / 26 | **29 / 35** |
| 요청 오류 | 0 / 7 | 0 / 0 | **0 / 0** |
| Music Avg@2 | 23.3% | 40.0% | **31.7%** (11/30, 8/30) |

문항 단위 paired bootstrap 95% 신뢰구간(`lab/summarize_runs.py compare`):

| 비교 | 전체 | Music |
|---|---|---|
| base → 재학습 | **+2.28%p [+0.56, +4.00]** (0 제외) | +8.3%p [-3.3, +20.0] (0 포함) |
| base → 원본 | +3.11%p [+1.33, +4.94] (0 제외) | +16.7%p [+8.3, +26.7] (0 제외) |
| 원본 → 재학습 | -0.83%p [-2.56, +0.89] (구분 안 됨) | -8.3%p [-20.0, +3.3] (구분 안 됨) |

- 재학습 모델도 base를 넘었고, 원본과는 통계적으로 구분되지 않는다. **같은 설정으로 학습을 두 번 반복해도 base를 넘었다**는 점이 첫 학습 결과의 재현 근거다.
- 루프 억제(잘림 89~98 → 29~35)는 분명히 재현됐다. 이것이 이 방법의 핵심 효과다.
- **Music은 재학습만으로는 확정할 수 없다.** 방향(base보다 높음)은 두 모델에서 같지만 30문항이라 크기는 불확실하다.
- 과목별(과목당 30문항이라 노이즈가 큼): 재학습에서 Mechanical_Engineering +21.7%p, Accounting +10.0%p, Computer_Science +10.0%p가 유의하게 올랐고 Biology -10.0%p, Geography -10.0%p가 유의하게 내렸다.
- 재학습 어댑터: `adapter_model.safetensors` 528,557,816바이트, 텐서 504개(fp32), SHA-256 `7c0eeba44958d44fd58fdc4bf6f982eb730ac3f84eb081dcc14097e0845f5533`. 학습 119스텝, 4,684초(원본 4,998초), 마지막 `pref_acc` 약 0.8~0.9.
- 평가 결과 묶음(`dpo_dv1_re.tar`, 577,187,840바이트, SHA-256 `784bdd11…d58df`)은 Drive를 거쳐 로컬로 옮겼고 두 해시가 모두 일치했다.

### 과목별 정확도 (%, 2 seed 평균, 과목당 30문항이라 과목별 차이는 노이즈가 크다)

| 과목 | base | dpo_dv1 원본 | dpo_dv1_re | 재학습-base |
|---|---|---|---|---|
| Mechanical_Engineering | 43.3 | 48.3 | 65.0 | +21.7 |
| Accounting | 71.7 | 75.0 | 81.7 | +10.0 |
| Computer_Science | 60.0 | 70.0 | 70.0 | +10.0 |
| Music | 23.3 | 40.0 | 31.7 | +8.3 |
| Architecture_and_Engineering | 48.3 | 55.0 | 55.0 | +6.7 |
| Electronics | 38.3 | 46.7 | 45.0 | +6.7 |
| Finance | 65.0 | 75.0 | 71.7 | +6.7 |
| History | 71.7 | 76.7 | 76.7 | +5.0 |
| Chemistry | 40.0 | 46.7 | 45.0 | +5.0 |
| Clinical_Medicine | 63.3 | 66.7 | 66.7 | +3.3 |
| Diagnostics_and_Laboratory_Medicine | 31.7 | 40.0 | 35.0 | +3.3 |
| Energy_and_Power | 50.0 | 61.7 | 53.3 | +3.3 |
| Manage | 63.3 | 66.7 | 66.7 | +3.3 |
| Economics | 81.7 | 86.7 | 83.3 | +1.7 |
| Marketing | 88.3 | 90.0 | 90.0 | +1.7 |
| Sociology | 65.0 | 58.3 | 66.7 | +1.7 |
| Materials | 43.3 | 45.0 | 45.0 | +1.7 |
| Literature | 86.7 | 86.7 | 88.3 | +1.7 |
| Pharmacy | 78.3 | 83.3 | 80.0 | +1.7 |
| Basic_Medical_Science | 68.3 | 76.7 | 68.3 | +0.0 |
| Psychology | 71.7 | 75.0 | 71.7 | +0.0 |
| Public_Health | 90.0 | 90.0 | 90.0 | +0.0 |
| Agriculture | 58.3 | 58.3 | 56.7 | -1.7 |
| Art | 55.0 | 55.0 | 53.3 | -1.7 |
| Design | 73.3 | 75.0 | 71.7 | -1.7 |
| Art_Theory | 73.3 | 70.0 | 70.0 | -3.3 |
| Math | 70.0 | 63.3 | 66.7 | -3.3 |
| Physics | 83.3 | 75.0 | 80.0 | -3.3 |
| Biology | 65.0 | 63.3 | 55.0 | -10.0 |
| Geography | 63.3 | 58.3 | 53.3 | -10.0 |
| **평균(30과목)** | **62.83** | **65.94** | **65.11** | **+2.28** |

재학습 기준 base보다 오른 과목 19, 내린 과목 8, 같은 과목 3. 과목 수준의 유의한 변화는 위 신뢰구간 항목을 따른다.

## 무엇을 했나

1. **데이터**: DeepVision-103K(`skylenage/DeepVision-103K`, MIT, 2026-02)의 `math-77k.parquet`만 사용. `test/` 폴더(MMMU 평가 사본 포함)는 쓰지 않았다.
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

## 새 머신에서 다시 만든 방법 (Ubuntu 24.04, sudo 없음)
실습실 PC가 회수되어 같은 사양의 리눅스 머신(RTX 4090 24GB, 드라이버 595.99.02, CUDA 13.2, RAM 31GiB)으로 옮겼다. `apt`를 쓸 수 없어서 Miniconda를 홈 폴더에 설치했다.
```bash
curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -y -n mmdl -c conda-forge --override-channels python=3.12 pip tmux gcc_linux-64 gxx_linux-64   # python3-venv와 tmux를 apt 없이
conda activate mmdl
git clone https://github.com/Groo-2/MMDL.git MMDL_ssmr && cd MMDL_ssmr && git checkout dpo_dv1-code
FORCE_PLAN=cu129 CONDA_ENV=mmdl bash lab/setup_env.sh      # 드라이버가 CUDA 13이어도 평가 환경(cu129)과 같게 맞춘다
```
- **torchcodec**: PyPI의 `torchcodec 0.17.0`(CUDA 13용)이 설치되면 vLLM 서버가 `libtorchcodec_image.so`(libcudart.so.13)를 못 찾아 죽는다. 이전 평가 환경과 같은 `torchcodec==0.17.0+cpu`로 교체하면 해결된다(`setup_env.sh`에 반영).
- 데이터: `dpo_dv.jsonl`은 받아 둔 파일(위 해시)을 쓰고, 이미지는 `build_general.py --pool-size 12000`(seed 42)로 다시 만들었다. 새로 만든 `train.jsonl`의 SHA-256이 원본(`677e8f3d…29c1c2`)과 일치함을 확인한 뒤 학습했다.
- 학습: `python music_ft/train_dpo.py --data data/dpo/dpo_dv.jsonl --output-dir results/dpo_dv1_re --epochs 1 --resume`, 평가: `python lab/eval_repeat.py --tag dpo_dv1_re --model results/dpo_dv1_re/merged --preset full2`.
- 이 머신의 다운로드가 느려서(약 2MB/s) Llama judge(약 16GB)를 따로 `snapshot_download`로 받아야 했다. 서버 기동 중에 받으면 대기 시간이 끝나 `vLLM 서버가 제시간에 준비되지 않았습니다`로 실패한다.

## 보관 (실습실)
```bash
bash lab/archive_model.sh dpo_dv1        # ~/mmdl/archive/dpo_dv1_<날짜>/ (어댑터, 합친 모델, 결과, 파생 데이터, 로그, 환경, 체크섬)
```
정밀도: 기본 모델 bf16, 학습은 bf16 혼합정밀도, LoRA 어댑터는 fp32로 저장, 손실의 softmax만 fp32, 평가용 모델은 어댑터를 bf16 base에 합쳐 bf16으로 저장(합친 모델의 dtype은 `{torch.bfloat16}`로 확인).

## 한계와 주의
- 한 데이터(DeepVision 수학, K12)만 사용했고 SSMR, MusiXQA, ViRL39K는 쓰지 않았다. Music 개선은 루프 감소의 부수 효과일 가능성이 있으며 음악 데이터의 효과가 아니다.
- DeepVision 공개일(2026-02)은 Qwen3-VL-4B(2025-10) 이후라 사전학습 노출 가능성은 낮다고 본다.
- 응답 생성과 판정이 sampling이라 같은 모델도 seed마다 약 1.2%p 달라진다.
- **재학습은 머신이 달라졌다**(드라이버, 커널, GPU 인스턴스). base는 새 머신에서 다시 재지 않았다. 새 머신에서 base의 Music 30문항 1회(seed 42, 스모크)를 돌렸을 때 잘림이 9로, 이전 머신의 같은 조건 4회(14, 15, 14, 17)보다 적었다. 하드웨어 때문에 같은 seed여도 sampling 결과가 다를 수 있어서, 재학습 점수에 머신 효과가 일부 섞였을 가능성을 배제하지 못한다.
- 원본 어댑터는 로컬로 옮기는 중 손상됐다(해시와 0 블록 검사로 발견). 원본의 평가 결과와 쌍 데이터는 온전하다. 어댑터의 온전한 사본은 아직 확인하지 못했다.
- 어댑터 같은 큰 파일은 **크기와 구조만으로 정상이라고 판단하지 않고** 해시와 0으로만 찬 블록 검사로 확인한다.
