# 처음부터 끝까지 재현하기 (데이터 → 학습 → 평가)

제출 모델(`dpo_dvmx1`: DeepVision 쌍 1,897 + MusiXQA 쌍 1,052를 섞어 DPO)을 이 저장소만으로 다시 만들고 평가하는 절차다. **모델 가중치는 저장소에 없으므로 학습부터 다시 한다.** 결과 요약은 [`README.md`](README.md), 실험별 코드 위치는 [`EXPERIMENTS.md`](EXPERIMENTS.md)에 있다.

**모든 명령은 `final_submission/code/`에서 실행한다**(코드가 이 폴더를 저장소 루트로 가정한다).

## 0. 두 가지 재현 방법
| | 방법 A (권장): 제공한 쌍 파일로 학습 | 방법 B: 쌍부터 새로 만들기 |
|---|---|---|
| 하는 일 | 3단계(데이터 준비) → 5단계(학습) → 평가 | 4단계의 응답 선별부터 전부 |
| 입력 | `data/dpo/`의 쌍 파일(학습에 쓴 것) | 원본 데이터와 base 모델 |
| 학습 데이터 | **같음**(해시로 확인) | 다름. base가 샘플링한 응답이라 새로 만들면 쌍이 달라진다 |
| 기대 결과 | 점수가 노이즈 범위에서 같음 | 같은 방향이지만 쌍이 달라서 점수 차이가 더 날 수 있음 |

## 1. 준비물
| 항목 | 값 |
|---|---|
| GPU | RTX 4090 24GB 한 장(같은 사양 권장). 이 코드는 24GB에 맞춰 LoRA와 마이크로 배치 1로 짠 것이다 |
| 디스크 | 약 60GB(Qwen 9GB + Llama 16GB + 데이터 + 체크포인트) |
| 계정 | Hugging Face 계정, `meta-llama/Llama-3.1-8B-Instruct` 사용 승인(채점 judge), `HF_TOKEN` 환경변수 |
| 시간(측정값) | DPO 학습 8,321초(약 2.3시간). MMMU 900문항 평가 한 번은 최종 모델이 약 1.6시간(base는 잘림이 많아 훨씬 오래 걸린다) |

`HF_TOKEN`은 환경변수나 저장소 루트의 `.env`에만 넣고 **커밋하지 않는다.**

## 2. 환경
학습과 선별은 CUDA 12.9 환경, 최종 평가는 CUDA 13 환경에서 했다(같은 모델의 점수는 두 환경에서 거의 같았다. [`report`](report/)의 부록 C).

```bash
cd final_submission/code
FORCE_PLAN=cu129 CONDA_ENV=mmdl    bash lab/setup_env.sh      # 학습·선별 환경
FORCE_PLAN=cu130 CONDA_ENV=mmdl130 bash lab/setup_env.sh      # 평가 환경
python lab/env_triage.py                                       # 문제가 있으면 코드를 고치기 전에 이것부터
```
- 실제로 쓴 패키지 버전은 `env/train_cu129_pip_freeze.txt`(torch 2.13.0+cu129, transformers 5.19.0, trl 1.16.0, peft 0.21.2, vLLM 0.27.1+cu129)와 `env/eval_cu130_pip_freeze.txt`(torch 2.13.0+cu130, transformers 5.22.2, vLLM 0.27.1, Python 3.12.16)에 있다. 기록용이고 설치는 `lab/setup_env.sh`가 한다.
- 단일 환경(cu130만)으로 학습한 적은 없어서 그 경우의 결과는 확인하지 않았다.

## 3. 데이터 준비
```bash
# (1) DeepVision 수학: 원본 내려받기. SHA-256 e926644ba752ca4a22cd384c6d6518d572672984c002b88d827bb5e501e371c8
python -c "from huggingface_hub import hf_hub_download as d; d('skylenage/DeepVision-103K','math-77k.parquet',repo_type='dataset',revision='d90619a39b1d2db7815ec958f2d728c78daa80cb',local_dir='data/general/deepvision')"
shasum -a 256 data/general/deepvision/math-77k.parquet           # 위 값과 같아야 한다

# (2) 변환(영어, 이미지 1장, 객관식 또는 숫자 정답) -> 균형 표본 12,000문항 풀과 이미지
python music_ft/build_general.py --parquet data/general/deepvision/math-77k.parquet --out-dir data/general/dv_pool --pool-size 12000
shasum -a 256 data/general/dv_pool/train.jsonl                   # 677e8f3dfb68e83f4e97db035fb4e811e2c032147ed1525ef33b95f2c629c1c2

# (3) MusiXQA 객관식 릴리스 (원본 puar-playground/MusiXQA, MIT). SHA-256 f34e1e334398dbbcaedc0b7161421e020f1db143b8c1443858eaea11fd0f1ea0
mkdir -p data/musixqa
curl -fL -o /tmp/musixqa.zip https://github.com/Groo-2/MMDL/releases/download/musixqa-mcq-pilot20-v1/musixqa_mcq_pilot20_v1.zip
shasum -a 256 /tmp/musixqa.zip                                    # 위 값과 같아야 한다
unzip -q /tmp/musixqa.zip -d data/musixqa                         # train_recipe.reviewed.jsonl(3,200), holdout.dedup.jsonl(400), images/
ls data/musixqa/images | head -1                                  # 압축 안의 구조가 그대로 풀려 images/ 가 data/musixqa 바로 아래에 생긴다
```
- 쌍 파일의 `image_paths`는 `data/general/dv_pool/images/…`와 `data/musixqa/images/…`를 가리킨다. 위 경로로 만들면 그대로 읽힌다.
- (2)의 결과는 저장소 밖 환경에서 다시 만들어 확인했다: `train.jsonl`의 SHA-256이 같았고, 쌍 1,897개의 이미지가 모두 생성됐다. MusiXQA 쌍 1,052개의 이미지도 모두 릴리스에 있다.

## 4. 방법 B 전용: 쌍을 처음부터 만들기 (방법 A는 건너뛴다)
```bash
python music_ft/dedupe.py --data-dir data/general/dv_pool --subjects all --names train          # MMMU 전 과목과 대조해 12,000 -> 11,995
python music_ft/select_ssmr.py --data-dir data/general/dv_pool --train-file train.dedup.jsonl --pool-size 6000 --max-images 1 --save-samples --start-server
python music_ft/build_dpo.py --source dv,data/general/dv_pool,train.dedup.jsonl --out data/dpo/dpo_dv.jsonl
python music_ft/select_ssmr.py --data-dir data/musixqa --train-file train_recipe.reviewed.jsonl --pool-size 3200 --max-images 1 --save-samples --samples-only --start-server
python music_ft/build_dpo.py --source mx,data/musixqa,train_recipe.reviewed.jsonl --out data/dpo/dpo_mx.jsonl
```
base로 문항당 8번 풀이(최대 2,048토큰, T 0.7, top_p 0.8, top_k 20)하는 단계라 **GPU와 시간이 든다.** 결과는 샘플링에 따라 달라지므로 쌍 파일의 해시가 달라지는 것이 정상이다.

## 5. 쌍 섞기와 학습 (방법 A는 여기서 시작)
```bash
# 방법 A: 학습에 쓴 쌍 파일을 쓴다
mkdir -p data/dpo && cp ../data/dpo/dpo_dv.jsonl ../data/dpo/dpo_mx.jsonl data/dpo/
shasum -a 256 data/dpo/dpo_dv.jsonl data/dpo/dpo_mx.jsonl
#   b2419a08c1228bfc626e5d5daa6ad22d5a98ea85c812f679a42d68327ac6ac25  dpo_dv.jsonl
#   d0937312c0c3898cf35fa8789706ec2989e8612601e26ce6f5d7c2779563e5da  dpo_mx.jsonl

# 이어 붙이고 seed 42로 셔플 (학습에 쓴 파일과 바이트 단위로 같다)
python music_ft/mix_dpo.py --inputs data/dpo/dpo_dv.jsonl data/dpo/dpo_mx.jsonl --out data/dpo/dpo_dv_mx.jsonl
shasum -a 256 data/dpo/dpo_dv_mx.jsonl        # a1b386dd6ff753383b26b5c83a1edba03b9b3e54594da26ae298e2d2c233b147

# 학습(CUDA 12.9 환경 mmdl): 기본값이 학습에 쓴 값이다(LoRA r64/α128, β 0.1, NLL 보조항 0.1, lr 5e-6 cosine, 누적 16쌍, seed 42)
python music_ft/train_dpo.py --data data/dpo/dpo_dv_mx.jsonl --output-dir results/dpo_dvmx1 --epochs 1 --save-steps 20 --save-total-limit 3 --resume
```
- 2,949쌍 ÷ 16 = 185스텝, 약 8,300초(RTX 4090)가 나와야 한다. 끝나면 `results/dpo_dvmx1/final_lora/adapter_model.safetensors`가 생긴다.
- 학습 중 끊겨도 같은 명령에 `--resume`이 있어 마지막 완전한 체크포인트에서 이어간다.
- **어댑터의 해시는 같지 않을 수 있다.** GPU 연산 순서 때문에 같은 쌍과 같은 seed로도 비트 단위로 같은 어댑터가 나오지 않는다. 제출 모델의 어댑터는 `4492c406fe29110a3c7b219890ee80d50bf1981b82956cedea27a00887acd131`이고, 같은 설정으로 다시 학습한 첫 DPO 모델(원본 vs 재학습)의 MMMU 점수 차이는 −0.83%p [−2.56, +0.89]로 구분되지 않았다.

## 6. 병합과 평가
```bash
python music_ft/merge_lora.py --lora results/dpo_dvmx1/final_lora --out results/dpo_dvmx1/merged --low-memory

# 평가는 CUDA 13 환경(mmdl130)에서. MMMU validation 900문항 x seed 42, 43. base도 같은 조건으로 잰다
export HF_TOKEN=<본인 토큰>
python lab/eval_queue.py base_130=Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17:2 \
                         dpo_dvmx1=results/dpo_dvmx1/merged:2
python lab/summarize_runs.py summary results/dpo_dvmx1
python lab/summarize_runs.py compare results/base_130 results/dpo_dvmx1      # 전체와 Music의 차이, 문항 단위 paired bootstrap 95% 신뢰구간
```
평가 프롬프트, sampling(T 0.7, top_p 0.8, top_k 20, presence 1.5), 최대 32,768토큰, 파서와 Llama-3.1-8B judge는 `mmmu_eval/`에 고정돼 있고 모델 경로만 바뀐다.

## 7. 기대 결과
| 항목 | 제출 모델의 값 (CUDA 13, 2 seed 평균) | 허용 범위 |
|---|---|---|
| base 전체 정확도 | 61.56% | seed와 환경 차이로 ±1%p 안팎 |
| 제출 모델 전체 정확도 | 65.89% (base 대비 +4.33%p [+2.44, +6.22]) | 재학습하면 ±1~2%p. base보다 높은지는 구간으로 확인 |
| Music(30문항) | 43.3% (base 18.3%) | 문항 30개라 한 문항이 3.3%p. 오차가 크다 |
| 잘림(시간 초과) | 10 / 17 (seed 42 / 43), base는 105 / 104 | |

점수는 두 seed 평균이고, 같은 모델도 seed만 바꾸면 문항별 정오가 약 14% 달라진다. **base와의 차이와 신뢰구간으로 비교**하고, 개별 문항이나 한 번의 점수로 비교하지 않는다.

## 8. 이 문서가 확인한 것과 확인하지 못한 것
확인함
- DeepVision 풀을 `build_general.py`로 새로 만들어 SHA-256이 같음(`677e8f3d…c1c2`), 쌍 1,897개의 이미지가 모두 생성됨.
- 제공한 두 쌍 파일에서 `mix_dpo.py`가 학습에 쓴 섞은 파일과 바이트 단위로 같은 파일을 만듦(`a1b386dd…b147`).
- MusiXQA 릴리스 zip의 SHA-256이 GitHub가 기록한 값과 같고, 쌍 1,052개의 이미지가 모두 들어 있음.
- `train_dpo.py` 기본값이 학습에 쓴 설정과 같음. 단위 테스트 통과(`lab` 79개, MusiXQA 구성 14개).

확인하지 못함
- **새 머신에서 학습과 평가를 끝까지 한 번 더 돌려 보지는 않았다.** 위 단계들은 학습에 쓴 실제 명령과 코드이고 GPU가 필요한 부분은 실습실에서 실행한 기록이 있지만, 이 문서의 순서 그대로를 처음부터 끝까지 실행한 것은 아니다.
- 방법 B(쌍을 새로 만드는 경로)로 만든 모델의 점수.
- MMMU test와 MMMU-Pro 평가의 **실행**(코드는 준비했고 9절 방법으로 GPU에서 돌리면 된다). 실행하면 `README.md`의 해당 절에 결과를 추가한다.

## 9. MMMU test와 MMMU-Pro 평가 (6단계의 병합까지 끝난 뒤, 평가 환경 mmdl130에서)
| 벤치마크 | 문항 수 | 데이터 | 비고 |
|---|---|---|---|
| `mmmu_test` | 10,500 | MMMU/MMMU test (정답은 2026-02 공개) | 선택에 쓰지 않은 데이터 |
| `mmmu_pro_standard10` | 1,730 | MMMU/MMMU_Pro "standard (10 options)" | 공식 MMMU-Pro 주 설정 |
| `mmmu_pro_vision` | 1,730 | MMMU/MMMU_Pro "vision" (질문과 선택지가 스크린샷 안에) | 공식 점수는 standard(10)와 vision의 평균 |
| `mmmu_pro_standard4` | 1,730 | MMMU/MMMU_Pro "standard (4 options)" | 선택사항 |

```bash
cd final_submission/code
bash lab/eval_extra.sh check      # 1) 데이터 점검(내려받기 포함, GPU 불필요). 과목별 문항 수와 정답, 요청 형식을 확인한다
bash lab/eval_extra.sh smoke      # 2) 벤치마크마다 과목당 1문항을 끝까지 돌려 서버 기동과 전 과정을 확인한다
bash lab/eval_extra.sh plan       # 3) 무엇이 남았는지만 출력
bash lab/eval_extra.sh run        # 4) 본 평가. 끊겨도 같은 명령을 다시 실행하면 끝난 문항은 건너뛰고 이어서 한다
```
- 기본값: 모델 `results/dpo_dvmx1/merged`(`MODEL_PATH`, `MODEL_TAG`로 바꾼다), `RUNS_PRO=2`(seed 42, 43), `RUNS_TEST=1`(seed 42). base도 같은 조건으로 재려면 `WITH_BASE=1`. 벤치마크는 `BENCHMARKS="mmmu_pro_standard10 mmmu_pro_vision mmmu_test"`로 고른다.
- 결과는 `results/<태그>/<벤치마크>/run_NN/`에 쌓이고 validation 결과(`results/<태그>/run_NN`)와 섞이지 않는다. 요약은 `python lab/summarize_runs.py summary results/<태그>/<벤치마크>`, base와의 비교는 `compare`.
- 직접 부를 수도 있다: `python lab/eval_queue.py --benchmark mmmu_test dpo_dvmx1=results/dpo_dvmx1/merged:1`, `python lab/eval_repeat.py --tag ... --model ... --benchmark mmmu_pro_vision --runs 2`.
- **시간(추정)**: 최종 모델이 validation 900문항을 seed 한 번에 약 1.6시간에 끝낸 기록을 문항 수로 환산하면 MMMU-Pro 한 설정은 seed 한 번에 약 3시간 이상, MMMU test는 약 19시간 이상이다. 측정값이 아니다. vision은 큰 스크린샷이라 더 걸릴 수 있다. base는 루프가 많아 훨씬 오래 걸리므로 시간이 모자라면 최종 모델만 먼저 돌린다.
- **점수 정의**: validation은 과목당 30문항이라 과목 평균과 전체 평균이 같지만, test와 MMMU-Pro는 과목별 문항 수가 달라서 공식 정의대로 **전체 문항 평균(micro)** 을 `overall_accuracy`로 쓴다(`summary.json`에 과목 평균도 함께 있다). 채점은 validation과 같은 방식(Answer: 줄 → Llama judge → 오답, 무작위 추측 없음)이라 공식 평가 코드의 점수와 조금 다를 수 있다. base와 같은 방식으로 비교한다.
- **선택 편향 주의**: MMMU-Pro 1,730문항 중 **577문항이 MMMU validation에서 온 문항**이다. 모델을 validation 점수로 골랐으므로 그 577문항은 선택에 쓰지 않은 데이터가 아니다. `python lab/pro_origin_split.py results/<태그>/mmmu_pro_standard10`가 출처(validation / test)별 정확도를 나눠 준다(`run`이 끝나면 자동으로 출력한다). MMMU test와 MMMU-Pro의 `test_` 1,153문항이 선택과 무관한 결과다. 두 데이터는 일부 겹친다(Pro의 `test_` 문항은 MMMU test의 문항을 다시 만든 것).
- 확인한 것: 벤치마크 정의와 요청 형식, 추론부터 채점까지 전 과정(가짜 서버), validation 결과가 예전 코드와 같음, 실제 데이터(MMMU-Pro 세 설정 전체, MMMU test의 Music과 Math)로 문항 수와 정답, 이미지를 확인(`python -m unittest lab.test_extra_eval`). 확인하지 못한 것: 실제 모델 서버로 돌린 결과(GPU 필요), MMMU test 30과목 전체의 다운로드(`check`가 확인한다).

문제가 생기면 [`code/lab/RUNBOOK_resume.md`](code/lab/RUNBOOK_resume.md)(중단과 복구), [`code/lab/README_lab.md`](code/lab/README_lab.md)(환경과 평가 설명)를 본다.
