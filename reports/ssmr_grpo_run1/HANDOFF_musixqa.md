# 인수인계: SSMR 학습 모델 위에 MusiXQA 추가 학습 (E2)

대상: SSMR로 학습한 `grpo_run1` 위에 MusiXQA를 추가 학습하려는 팀원. 작성 2026-11-02.
SSMR 단계의 결과와 설정은 [README.md](README.md)에 있다. 이 문서는 **이어서 할 때 알아야 할 것**만 적는다.
확인된 사실과 추정을 구분해서 적었다. "미검증"이라고 쓴 것은 실제로 돌려 보지 않았다.

## 0. 지금 해도 되는 일 / 기다릴 일 / 하지 말 일 (2026-11-02 기준)

MMMU 전체 900문항 평가(`full2`)는 2026-11-04에 끝났다. **결과: SSMR 모델은 전체 점수를 62.83% -> 55.39%(Avg@2, -7.4%p)로 떨어뜨렸다.** 아래 C절을 먼저 읽을 것.
실습실 GPU는 다른 작업이 쓰고 있을 수 있으니 시작 전에 `nvidia-smi`로 확인한다.

### A. 지금 해도 된다 (GPU 필요 없음)

| 할 일 | 방법 | 끝났다는 기준 |
|---|---|---|
| 1. 어댑터 받고 무결성 확인 | 1절의 `gh release download`, `shasum -a 256 -c SHA256SUMS` | 두 파일 모두 `OK` |
| 2. MusiXQA 객관식 데이터 만들기 | `musixqa` 브랜치의 `lab/MusiXQA_객관식_요청서.md`대로(4지선다, 카테고리, 오답 규칙, 분량, 문서당 최대 2문항) | train/holdout 분리, holdout 문서가 train에 없음 |
| 3. 형식 검증 | `python lab/validate_mcq_dataset.py --data-dir <폴더>` (`musixqa` 브랜치) | ERROR 0개 |
| 4. 눈으로 내용 확인 | 카테고리별 무작위 20개 이상: 정답이 맞는지, 보기가 모호하지 않은지, 이미지에서 답이 보이는지 | 문제 있는 문항을 고쳤다 |
| 5. MMMU 중복 제거 | `python music_ft/dedupe.py ...`(SSMR에서는 겹침 0건이었다). MMMU validation/test와 겹치면 학습에서 뺀다 | 겹침 보고서 확인 |
| 6. 이미지 토큰 수 측정 | 아래 코드. SSMR 이미지는 약 990토큰이었다 | MusiXQA 풀페이지의 토큰 수를 알았다 |
| 7. 보고서 재료 정리 | 데이터 출처(MusiXQA, MIT), 만든 방법, 중복 제거 방법, 카테고리별 개수 | 위 항목이 문서에 있다 |

이미지 토큰 수 측정(CPU, 학습 코드와 같은 방식):
```python
from transformers import AutoProcessor
from PIL import Image
p = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17")
ip = p.image_processor
ip.min_pixels, ip.max_pixels = 1003520, 4014080
ip.size = {"shortest_edge": 1003520, "longest_edge": 4014080}
out = ip(images=[Image.open("악보_예시.png").convert("RGB")], return_tensors="np")
t, h, w = out["image_grid_thw"][0]
print("이미지 토큰:", int(t * h * w) // (ip.merge_size ** 2))
```
이 값이 SSMR의 약 2배(약 2,000)라면 메모리를 다시 시험해야 한다(2절 5번).

### B. 실습실 GPU가 비어 있을 때 한다

- 어댑터를 merge하고 SSMR hold-out에서 47.25% 근처가 나오는지 확인(로딩 검증)
- `--stress`로 MusiXQA 최악 문항 1스텝 → 파일럿 2스텝(스텝 시간, `grad_norm`, 보상)
- (선택) SSMR 모델로 통과율 선별

### C. 결정 (`full2` 결과 반영, 2026-11-04)

- **SSMR 모델은 MMMU 전체 점수를 7.4%p 떨어뜨렸다**(실습실 base 62.83% -> 55.39%). 과제의 주 지표가 전체 성능이므로, SSMR 모델 위에 그대로 MusiXQA를 얹는 것은 권하지 않는다.
- 원인은 아직 분리하지 못했다(짧아진 응답, 4지선다만 학습, 정책 쏠림 등은 가설일 뿐). **기본 방침은 base에서 새로 MusiXQA를 학습하는 것**이다. SSMR 모델 위에서 이어서 하려면 하락의 원인을 먼저 확인해야 한다.
- 아래는 이전 기록이다.

- **SSMR 모델 위에서 이어서 할지, base에서 새로 학습할지, 둘 다 할지.** 응답이 짧아진 SSMR 모델이 다른 과목 점수를 떨어뜨렸다면 그 위에서 시작하는 것이 불리할 수 있다.
- 이어서 학습하는 코드(3절의 방법 A 또는 B) 구현. 구현은 팀 합의 후에 한다.

### D. 하지 않는다

- **실습실 PC에서 GPU를 쓰는 작업**(`full2`와 충돌해 둘 다 실패하거나 같은 결과 폴더를 덮어쓸 수 있다). `nvidia-smi`가 500MiB 안팎일 때만 시작한다.
- **실습실의 `~/mmdl/MMDL_ssmr` 폴더에서 `git checkout`/`git pull`로 `lab/` 파일을 바꾸는 것.** 실행 중인 `run_pipeline.sh`는 bash가 실행 도중에 읽기 때문에 파일이 바뀌면 오동작할 수 있다. 작업은 다른 폴더에 새로 clone해서 한다.
- 본 학습 시작(위 C를 정하기 전), 평가 조건(프롬프트, sampling, 해상도, judge) 변경, Music 4번을 8번으로 늘리기(팀 합의 필요).
- `.env`나 HF 토큰, 100MB 넘는 파일을 git에 올리는 것(가중치는 GitHub Release를 쓴다).

## 1. 받는 것

| 항목 | 내용 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` (bf16, 양자화 금지) |
| 학습된 가중치 | LoRA 어댑터 `grpo_run1/final_lora`(r64/alpha128, LLM의 q,k,v,o,gate,up,down). **위치: GitHub Release [`ssmr-grpo-run1-lora`](https://github.com/Groo-2/MMDL/releases/tag/ssmr-grpo-run1-lora)** (`adapter_model.safetensors` 528,557,816바이트, `adapter_config.json`, `SHA256SUMS`). `gh release download ssmr-grpo-run1-lora --repo Groo-2/MMDL`로 받고 `shasum -a 256 -c SHA256SUMS`(Linux는 `sha256sum -c`)로 확인. `adapter_config.json`의 `revision`이 null이므로 base는 위 revision을 직접 지정하고, `inference_mode`가 true라서 이어서 학습하려면 `is_trainable=True`가 필요 |
| 합친 모델 | 어댑터를 base에 merge한 전체 모델은 약 9GB라 저장소에 올리지 않는다. `python music_ft/merge_lora.py --lora <어댑터> --out <출력>`로 만든다(몇 분). 값이 바뀐 가중치가 252개 나와야 정상 |
| 코드 | `ssmr` 브랜치의 `music_ft/`(학습·선별·merge), `lab/`(실행 파이프라인·평가·환경 점검) |
| MusiXQA 요청서·검증기 | `musixqa` 브랜치의 `lab/MusiXQA_객관식_요청서.md`, `lab/validate_mcq_dataset.py` (**`ssmr` 브랜치에는 없다**) |

어댑터를 불러오는 방법(확인된 방식은 merge뿐이고 학습 재개는 아래 3절 참고):
```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration
import torch
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "path/to/final_lora")    # 학습을 이어 가려면 is_trainable=True
```

## 2. SSMR 단계에서 알게 된 것 (MusiXQA에 영향이 큼)

1. **SSMR 학습은 MMMU Music 점수를 올렸다고 말할 수 없다.** hold-out은 +29%p였지만 대부분이 "잘림 제거"였고, MMMU Music Avg@4는
   25.8% → 27.5%로 신뢰구간 [-14, +18]%p 안이다. 잘림(약 50% → 2.5%)은 전이됐지만 정답은 늘지 않았다. 이것이 E2를 하는 이유다. 더 나아가 **MMMU 전체 900문항은 62.83% -> 55.39%(Avg@2)로 7.4%p 떨어졌다.** 잘림은 줄었는데(실습실 base 89/98 -> 66/55) 끝낸 문항의 정확도가 약 69% -> 약 59%로 내려갔다.
2. **MusiXQA 위에 학습한 결과를 SSMR 단계 모델과 비교하려면 같은 평가를 써야 한다.** 4번(seed 42~45)의 Music과 전체 900문항 2번(seed 42, 43),
   그리고 MusiXQA hold-out. 평가 조건(프롬프트, sampling, 해상도, judge)은 동결이다.
3. **학습 신호는 금방 소진된다.** 보상은 20스텝 만에 올라 정체됐고 41스텝부터 문항의 80~89%는 8개 응답이 모두 같은 보상이었다(학습 신호 없음).
   SSMR 모델은 이미 짧게 끝내는 습관(약 110토큰)이 있어서 이 습관이 MusiXQA에도 이어질 가능성이 높다(미검증).
4. **문항 선별이 중요하다.** 통과율 0/8, 8/8 문항은 신호가 없다. SSMR에서는 base로 문항당 8번 풀어 1/8~7/8만 남겼다(`music_ft/select_ssmr.py`).
   **MusiXQA는 학습 시작점이 base가 아니라 SSMR 모델이므로 선별도 SSMR 모델로 해야 의미가 있다.** 현재 `select_ssmr.py`는 base 경로·revision을
   전제로 vLLM을 띄우므로 로컬 경로 모델에는 수정이 필요하다(미검증).
5. **이미지 크기가 메모리와 시간을 결정한다(가장 큰 함정).**
   - SSMR 이미지 5장짜리 문항(선택지가 이미지, 약 2,731개)은 24GB에서 한 스텝에 78분, GPU 최대 24.2GB로 한계였다. 그래서 학습에서 제외했다.
   - 이미지 1장 문항(약 990토큰)은 최악 조건에서도 12.8GB, 파일럿 14.6GB였다.
   - MusiXQA 풀페이지 악보(1191×1684)는 평가 해상도 설정에서 리사이즈 없이 약 1,960토큰으로 추정된다(SSMR의 약 2배, 미검증).
     **본 학습 전에 반드시 `--stress`로 메모리와 스텝 시간을 재라.** `--max-images N`으로 이미지 수가 많은 문항을 뺄 수 있다.
6. **학습 시작 시 응답이 길면 시간이 폭증한다.** base는 2,048토큰에서 잘리는 응답이 50~60%여서 SSMR 초반 스텝이 22분, 후반(응답 약 110토큰)은 2분이었다.
   SSMR 모델에서 시작하면 초반부터 빠를 가능성이 높지만 미검증이다.
7. **체크포인트 보존 수가 3개다**(`save_total_limit=3`). 중간 시점을 비교하려면 이 값을 늘려라. SSMR에서는 120/130/140만 남았다.
8. **(가설) 이 학습은 풀이 방식을 짧은 정형 틀로 바꿔 전체 점수를 떨어뜨렸을 가능성이 크다.** 같은 seed(42), 같은 실습실 조건에서 base 561 -> grpo 495(-66)이고
   21개 과목이 떨어졌으며, base 잘림이 0~1개였던 14개 과목에서만 -30문항이다(잘림과 무관한 하락). Biology에서 base 정답 -> grpo 오답 9문항 중 4개를 읽으면,
   base는 선택지를 비교하고 스스로 재검토하는 긴 풀이(363~1,958토큰)인데 grpo는 모두 "Step 1, 2, 3 -> Answer"의 짧은 틀(111~238토큰)로 바로 결론을 낸다.
   원인 후보: 학습의 `max_completion_length=2048`에서 **잘린 응답은 보상 0**이라 "짧게 끝내기"가 가장 빨리 보상받는 행동이 됐다(잘림 0.62 -> 0과 응답 길이 감소가 같은 시점).
   평가는 최대 32,768토큰이라 이 압력이 맞지 않는다. 시험하지 않은 가설이며, MusiXQA 학습에서는 **잘린 응답을 손실에서 제외하거나(예: TRL `mask_truncated_completions`)
   완료 길이를 늘리고, KL 페널티(beta>0)와 짧은 스텝 수를 쓰는 것**을 고려하라(이 환경의 TRL 1.16.0에서 옵션 지원 여부는 확인하지 못했다).

## 3. 코드 현황: 이어서 학습하는 기능은 없다

`music_ft/train_grpo.py`는 항상 **base 모델 + 새 LoRA**에서 시작한다. SSMR 모델에서 이어 학습하려면 아래 둘 중 하나가 필요하고 **둘 다 구현·시험되지 않았다**.

| 방법 | 내용 | 필요한 수정 |
|---|---|---|
| A. 같은 어댑터를 계속 학습 | `PeftModel.from_pretrained(model, 어댑터, is_trainable=True)`로 올린 모델을 `GRPOTrainer`에 넘기고 `peft_config`는 넘기지 않는다 | `train_grpo.py`의 모델 로딩부(`from_pretrained` 뒤)와 `peft_config` 인자. TRL이 이미 PEFT 모델인 경우를 처리하는지(gradient checkpointing 포함)는 확인 필요 |
| B. merge한 모델 위에 새 LoRA | merge 결과를 base로 삼고 새 어댑터를 학습 | `MODEL_REPO`/`MODEL_REVISION` 상수를 로컬 경로로 바꾸는 옵션. 평가 시에는 어댑터가 두 개이므로 최종 모델을 다시 merge해야 함 |

- A는 한 모델의 연속 학습이라 해석이 단순하고, B는 코드 수정이 작다. 어느 쪽을 쓸지는 팀 결정.
- 어느 쪽이든 **먼저 SSMR 모델 자체의 점수를 그대로 재현하는지** 확인하고(SSMR hold-out 47.25% 근처), 그 뒤에 MusiXQA를 얹어라.
  합친 모델에서 hold-out 47.25%가 안 나오면 로딩이 잘못된 것이다.
- 비교군은 둘: base → MusiXQA(이어서가 아닌 새로 학습)와 SSMR → MusiXQA. 효과가 MusiXQA 때문인지 SSMR 선행 학습 덕분인지는 이 둘을 같이 봐야 가려진다.

## 4. 데이터 요구사항 (교수님 규칙 + 파이프라인 요구)

- 학습 데이터 조건: ① 대학 수준 난이도 ② MMMU 형식(이미지+문제+선택지+정답) ③ **MMMU validation/test 데이터가 아닐 것**.
  MMMU Music과 겹치는지 `music_ft/dedupe.py`로 확인하라(SSMR는 겹침 0건).
- 보상(`music_ft/reward.py`)은 4지선다(`NUM_OPTIONS = 4`, A~D)를 가정하고 `Answer: X` 줄을 읽는다. 선택지가 4개가 아니면 수정이 필요하다.
- 형식 검증: `musixqa` 브랜치의 `lab/validate_mcq_dataset.py --data-dir <폴더>`. ERROR가 있으면 학습 전에 고친다. 요청서에 오답 구성 규칙,
  분량(train 카테고리당 800+, holdout 100+, 문서당 최대 2문항), SSMR 결함 교훈이 있다.
- 검증기가 형식은 잡지만 문제 내용(정답이 맞는지, 보기가 모호하지 않은지)은 못 잡는다. 샘플을 눈으로 확인하라.

## 5. 실습실 운영 주의 (WSL2 + RTX 4090, 모두 실제로 겪은 것)

- 환경: conda `mmdl`. 새 셸에서는 `conda activate mmdl`, `export CONDA_ENV=mmdl`, 학습 전 `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8`
  (파이프라인은 이 변수가 없을 때 `expandable_segments:True`를 기본으로 쓰는데, 5장 문항 시험에서 `CUDA driver error: out of memory`로 실패했다.
  원인이 이 옵션인지는 확인하지 못했다. SSMR 학습은 위 설정으로 했고 OOM이 없었다).
- 에러가 반복되면 코드를 고치기 전에 `python lab/env_triage.py`(CUDA·패키지 버전)부터 본다.
- WSL2에서 vLLM을 직접 띄우면 `UVA is not available`로 실패한다. `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`이 필요하며 `eval_repeat.py`가 자동으로 설정한다.
- Llama judge는 gated 모델이라 `.env`에 `HF_TOKEN`(커밋 금지)이 필요하다. **첫 다운로드가 실습실 네트워크에서 약 1.5시간(2~3MB/s)** 걸려서 서버 대기
  60분 제한에 걸린다. 이미 받아 뒀으면 캐시를 쓴다. 새 PC에서는 `huggingface_hub.snapshot_download(..., allow_patterns=["*.json","*.safetensors","tokenizer*"], ignore_patterns=["original/*"])`로 미리 받는다.
- GPU는 학습·평가·vLLM 서버가 동시에 쓸 수 없다. 서버가 남아 있으면 `pkill -f "vllm serve"`.
- 학습은 tmux 안에서, 중단되면 같은 명령으로 재개된다(`--resume`, 체크포인트 10스텝마다, 자동 재시도 3번).
- Windows 앱(브라우저 등)이 VRAM을 쓰면 여유가 줄고, 절전·자동 업데이트로 WSL이 꺼지면 학습이 끊긴다.

## 6. 권장 절차

1. 어댑터를 받아 merge하고 SSMR hold-out에서 47.25% 근처가 나오는지 확인(로딩 검증).
2. 팀원이 만든 MusiXQA를 `validate_mcq_dataset.py`와 `dedupe.py`로 검증, 샘플을 눈으로 확인.
3. 메모리/시간: `--stress`로 최악 문항 1스텝 → 파일럿 2스텝에서 스텝 시간·`grad_norm`·보상 확인.
4. (선택) SSMR 모델로 통과율 선별 → 1/8~7/8 문항만 학습.
5. 본 학습. 스텝 수는 SSMR 경험상 보상이 정체되는 지점(약 20~40스텝)을 지나면 효과가 작으므로 많이 잡을 필요가 없다(추정).
6. 평가: MusiXQA hold-out → MMMU Music 4번 → 전체 900문항 2번. 학습 모델과 base, SSMR 모델을 같은 조건으로.
7. Music 4번을 8번으로 늘리는 것은 팀 합의가 필요하다. 현재 차이 크기에서는 8번으로도 구분이 어렵다는 계산이 있다.

## 7. 동결된 규칙 (바꾸면 비교가 깨진다)

- 모델 revision, bf16, 양자화 금지, 프롬프트 `mmmu_pro_cot`, 파서(규칙 → Llama judge → 오답), 해상도 min 1,003,520 / max 4,014,080,
  최대 생성 32,768토큰, 평가 sampling(temperature 0.7, top_p 0.8, top_k 20, repetition 1.0, presence_penalty 1.5), seed 42~
- 보고서에는 HW, 툴 패키지 버전, seed, 학습 데이터 출처와 중복 제거 방법을 쓴다(교수님 요구).
- 학습 rollout에는 presence_penalty를 쓰지 않았다(HF generate 제약). 평가는 그대로다. 이 차이를 보고서에 적을 것.

## 8. 모르는 것

- 어떤 변화가 MMMU Music을 올릴지(데이터 종류, 학습량, 분포 차이)는 아직 모른다. SSMR는 합성 문제라서 실제 문제와의 분포 차이가 병목일 수 있다.
- SSMR 모델의 MMMU 전체 점수가 떨어진 **원인**(응답 길이, 4지선다 편향, 학습 설정, 어느 과목 유형)은 분리하지 못했다.
- 위에서 "미검증"이라고 쓴 모든 항목.
