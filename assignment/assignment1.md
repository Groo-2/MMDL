# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 에이쁠받고싶삼
- **팀원**: 이상준(팀장), 고은혁, 최다연
- **작성일**: 2026-09-29
- **재현 커맨드**: `cd baseline && bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --data_root <MMMU_데이터경로> --output_dir ./results/baseline_mmmu_pro_cot` (judge 모델이 gated라 `HF_TOKEN` 필요, §1 참고)

> **용어**: 이 보고서의 **Baseline**은 **Run B**(MMMU-Pro 공식 CoT 프롬프트)다. **Run A**는 baseline을 정하기 위해 먼저 돌린 **첫 실험**(팀 GitHub 프롬프트 사용, 나머지 조건은 Baseline과 동일)이다. 두 실행의 비교는 [baseline/reports/baseline_decision.md](../baseline/reports/baseline_decision.md)에 있다.

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.27.1 (`vllm serve` OpenAI 호환 서버, 동시 요청 32), torch 2.13.0+cu130 |
| 사용 GPU | NVIDIA A100-SXM4-40GB, 40GB (Google Colab Pro) |
| 실측 peak VRAM | 36.1GB (추론 단계, `nvidia-smi` 5초 간격 최댓값). `--gpu-memory-utilization 0.90`(0.90 × 40GB = 36GB)과 거의 같아서, 실제 필요량이 아니라 vLLM이 미리 점유한 양에 가깝다 |
| 총 소요 시간 | 추론 232.7분 + judge 3.8분 = 236.5분 (900문제 기준) |
| 의존성 | [baseline/requirements.txt](../baseline/requirements.txt): `vllm==0.27.1`, datasets, huggingface_hub, requests, Pillow, pandas (torch, transformers는 vLLM과 함께 설치). judge 모델이 gated라 `HF_TOKEN` 환경변수가 필요하다 |
| 실행 커맨드 | 아래 코드 블록 (모델 위치는 `--model_path`, MMMU 데이터 위치는 `--data_root` 인자로 받는다) |

```bash
cd baseline
pip install -r requirements.txt
export HF_TOKEN=<HF 토큰>   # judge(meta-llama/Llama-3.1-8B-Instruct)가 gated 모델 — HF에서 라이선스 동의 필요

bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root <MMMU HF 캐시 디렉터리> \
  --output_dir ./results/baseline_mmmu_pro_cot
```

- 스크립트는 평가 모델 서버 기동 → infer → judge 서버 기동 → judge → score를 차례로 실행한다. 중간에 끊기면 같은 커맨드를 다시 실행하면 끝난 문항은 건너뛴다(resume).
- fine-tuning 후 재평가: `--model_path`만 체크포인트 경로로 바꾸고 `--model_revision`은 생략한다. 프롬프트·sampling·파서·judge는 코드에 고정돼 있어 그대로 재사용된다.
- infer 서버를 띄울 때 스크립트가 환경변수 `VLLM_USE_V2_MODEL_RUNNER=0`을 설정한다(judge 서버에는 설정하지 않음).

**추론 백엔드 선택 근거:**
- 응답이 최대 32,768 토큰까지 길어지는 CoT 생성이라, 순정 `transformers.generate()`로 900문항을 돌리면 시간이 매우 오래 걸린다(가이드 §5의 우려와 같다). 그래서 vLLM 서버에 동시 요청 32개를 보내는 방식으로 처리했고, 실측 추론 시간은 232.7분이다. (`transformers`와의 속도 비교 실험은 하지 않았다.)
- OpenAI 호환 서버라서 평가 모델(`vllm serve <모델 또는 체크포인트>`)과 judge를 같은 스크립트로 다루고, 요청 단위로 resume할 수 있다.
- 한계: temperature 0.7 샘플링이므로 seed를 고정해도 동시 요청 구성에 따라 결과가 조금씩 달라질 수 있어, 비트 단위 재현은 보장하지 않는다(§3.1, §7).

---

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시). 객관식(847문항):

```
{question}
A. {option_A}
B. {option_B}
...
Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.
```

주관식(short-answer) 문항(MMMU val 53문항):

```
{question}
Answer the preceding question. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes). Think step by step before answering.
```

**입력 구성** (코드: [baseline/mmmu_eval/prompting.py](../baseline/mmmu_eval/prompting.py)):
- 요청은 user 메시지 1개이며, content는 위 텍스트가 먼저, 그 뒤에 이미지가 온다.
- 질문 속 `<image N>`은 `[image]`로 바꾸고, 이미지는 `<image N>`이 등장한 순서대로 텍스트 뒤에 붙인다. 이미지는 JPEG 품질 95의 base64 `image_url`로 보낸다(MMMU-Pro 공식 `replace_images_tokens()`와 같은 방식).
- 텍스트에 `<image N>` 표시가 없는 이미지는 넣지 않는다(MMMU-Pro 공식 `origin_mmmu_doc_to_visual`과 같은 처리). validation에서 이런 이미지가 있는 4문항(Agriculture_26, Materials_15, Pharmacy_4, Pharmacy_22)을 직접 열어 보니 문제 자료가 아니라 풀이 해설 이미지여서, 넣으면 정답이 유출된다. 이 4문항도 **문항은 제외하지 않고** 900문항 모두 평가했다.

- **출처**:
  - 객관식: [MMMU-Pro 공식 `prompts.yaml`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml)의 `cot.standard` 문구를 그대로 사용. 조립 방식은 [`mmmu-pro/infer/infer_transformers.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_transformers.py)의 `construct_prompt()`를 따랐다.
  - 주관식: MMMU-Pro에는 객관식만 있어 공식 문구가 없다. 위 문장 구조를 유지한 채 `multiple choice`와 `$LETTER ... one of options` 부분만 바꾼 **직접 변형**이다.
- **선택 이유**:
  - 벤치마크 제작진이 공개한 CoT 프롬프트라 출처가 분명하고, 이후 MMMU-Pro 평가에도 같은 계열을 그대로 쓸 수 있다.
  - 최종 답 형식(`Answer: $LETTER`)이 고정돼 있어 파싱 규칙(§4)을 일관되게 적용할 수 있다.
  - Baseline을 정하기 위한 첫 실험(Run A, 팀 GitHub 프롬프트 `Keep the reasoning concise`)과 나머지 조건을 같게 비교했더니 점수 차이가 유의하지 않았다(McNemar p = 0.756). 그래서 출처가 분명한 이 프롬프트를 Baseline(Run B)으로 채택했다. 비교 과정은 [baseline/reports/baseline_decision.md](../baseline/reports/baseline_decision.md) 참고.
  - 단점: `Think step by step`이 일부 문항에서 반복 루프를 유발해 응답이 잘리는 경우가 있다(§7).

---

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | True (vLLM은 `do_sample` 개념이 없고 `temperature > 0`이면 sampling) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42 |

- **출처**: Qwen 공식 평가 코드
  - `temperature`/`top_p`/`top_k`/`repetition_penalty`/`presence_penalty`/`max_new_tokens`: [`evaluation/mmmu/infer_instruct.sh`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/infer_instruct.sh)와 [`evaluation/mmmu/README.md`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)의 "Model-Specific Configurations → Instruct Models". (README의 Instruct 항목은 2B/30B를 예시로 들고 있어 4B-Instruct 전용 값이 따로 공개돼 있지는 않다. 같은 Instruct 계열 값을 그대로 사용했다.)
  - `seed = 42`: 같은 저장소 [`evaluation/mmmu/run_mmmu.py`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/run_mmmu.py)에서 `SamplingParams`가 아니라 `LLM(..., seed=42)` 생성자에 지정돼 있다. 우리 구현은 서버 `--seed 42`와 요청마다 `seed=42`를 함께 전달한다.
- greedy 등 다른 값은 쓰지 않았다. 단, temperature 0.7 샘플링이므로 같은 설정으로 다시 돌려도 점수가 달라질 수 있다(900문항 1회 측정의 95% 신뢰구간 약 ±3.1%p).

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 32,768 |
| `max_model_len` (vLLM) | 40,960 |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels`) | 1,003,520 (1280 × 28²) / 4,014,080 (5120 × 28²), vLLM 요청의 `mm_processor_kwargs`로 전달 |
| 이미지 인코딩 | JPEG 품질 95 |

**선택 근거**:
- **max_new_tokens = 32,768**: Qwen 공식값이다. 처음 쓰던 설정(4,096 토큰, seed 3407, 팀 GitHub 프롬프트)에서는 900문항 중 128문항(14%)이 4,096 토큰에서 잘렸고, 파싱 실패 129건 중 128건이 잘린 응답이었다. 정답 잘림이 점수에 직접 영향을 주므로 공식값으로 올렸다. 그래도 Baseline(Run B)에서 108문항(12.0%)이 32,768 토큰 상한까지 생성했다(`finish_reason=length`, 정답률 약 7%). 대가로 추론 시간이 232.7분 걸렸다.
- **max_model_len = 40,960**: 생성 32,768 + 최대 프롬프트 약 5.7K 토큰을 담는 값이다. 공식은 128,000이지만, 4B 모델의 KV cache가 토큰당 약 0.14 MiB(36층 × KV head 8 × 128 × bf16 × K,V)라 128,000 토큰이면 약 17.6GiB가 필요해 RTX 4090(24GB)에서 서버를 띄우기 어렵다고 산정했다. 40,960이면 최대 길이 응답 1개가 약 5.3GiB다. (4090 실행은 이 baseline에서 실측하지 않았다. A100에서만 측정했다.)
- **이미지 해상도**: Qwen 공식 `build_mmmu_prompt()`와 같은 값이다. 도표·수식 비중이 큰 MMMU에서 글자·숫자 정보가 줄어드는 것을 막으려고 낮추지 않았다.
- **속도/VRAM trade-off**: A100 40GB에서 `--gpu-memory-utilization 0.90`, 동시 요청 32로 실행했다. 위 KV cache 산정으로는 최대 길이 응답을 4개 이상 동시에 담을 수 있다.

---

## 4. 채점(파싱) 방식

- **사용한 파서/로직**: 자체 구현한 혼합 파서(결정적, 랜덤 없음). 코드: [baseline/mmmu_eval/parsing.py](../baseline/mmmu_eval/parsing.py)(rule 단계), [baseline/mmmu_eval/judge_utils.py](../baseline/mmmu_eval/judge_utils.py)(judge 단계).
  - rule 단계는 MMMU-Pro 공식 [`evaluate.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py)의 `parse_multi_choice_response()` 1단계(`Answer:` 줄 기준)를 차용했다.
  - judge 프롬프트와 응답 해석은 Qwen 공식 [`evaluation/mmmu/eval_utils.py`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/eval_utils.py)의 `build_prompt()`, `can_infer()` 원문을 사용했다.

**동작 방식** (위에서부터 순서대로, 앞 단계가 성공하면 멈춘다):

1. **rule 파서**: 응답에서 **마지막 `Answer:`**(대소문자 구분, `**Answer:**`나 `Final Answer:`도 포함) 뒤의 첫 번째 비어 있지 않은 줄을 가져와 `**`, 백틱, LaTeX 래퍼 등을 정리한다.
   - 객관식: 그 줄이 선택지 표기로 시작하면(`B`, `(B)`, `B.`, `B:`) 그 글자를 답으로 한다. 아니면 줄 안의 독립된 대문자 중 유효한 선택지가 **정확히 하나**일 때 그 글자를 답으로 한다. 대문자만 인정한다(관사 `a`를 A로 읽는 것을 막기 위해). 선택지 개수만큼(A부터) 유효하다.
   - 주관식: 정리한 답이 정답 목록 중 하나와 같으면 정답이다. 숫자끼리는 소수 둘째 자리로 반올림해 비교하고, 문자열은 정규화(소문자화, 공백 정리, 끝의 구두점 제거) 후 완전 일치로 비교한다. 쉼표/세미콜론/`and`로 나뉜 조각이 **모두** 정답 목록 중 하나와 각각 일치해도 정답이다.
2. **judge**: rule이 답을 못 뽑은 문항만 `meta-llama/Llama-3.1-8B-Instruct`(revision `0e9e39f249a16976918f6564b8830bc894c89659`, temperature 0, seed 42, 최대 2,048 토큰)에 보낸다. 판정 대상 **모델 응답이 16,384 토큰을 넘으면 마지막 16,384 토큰만** 넣는다(judge 서버를 4090에서도 띄울 수 있게 max_model_len 20,480으로 고정).
   - 객관식은 아래 프롬프트를 쓰고, judge 응답을 `can_infer()`로 해석한다. `Z`이거나 해석할 수 없으면 실패다.
   - 주관식은 직접 설계한 "최종 답만 뽑기" 프롬프트를 쓰고(답이 없으면 `NONE`), 뽑힌 답을 1번과 같은 방식으로 정답과 비교한다.
3. **judge도 실패하면 오답**으로 처리한다. 공식 코드의 `random.choice` fallback은 재현이 안 되므로 쓰지 않았다.

객관식 judge 프롬프트(Qwen/VLMEvalKit 원문, `{option_*}`은 `A. ...` 형태 목록):

```
You are an AI assistant who will help me to match an answer with several options of a single-choice question. You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. If the meaning of all options are significantly different from the answer, output Z. Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. 
Example 1: 
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: a cute teddy bear
Your output: A
Example 2: 
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: Spider
Your output: Z
Example 3: 
Question: {question}?
Options: There are several options: 
A. {option_A}
B. {option_B}
...

Answer: {model_response}
Your output: 
```

주관식 judge 프롬프트(직접 설계):

```
You are an AI assistant who will help me extract the final answer from a response to a question. You are provided with a question and a response. Output only the final answer that the response gives, as a short phrase or number, without any explanation. If the response does not give a final answer, output NONE.
Question: {question}
Response: {model_response}
Final answer: 
```

**공식 파서를 그대로 쓰지 않은 이유** (팀이 처음 낸 GitHub 응답 900개를 [baseline/scripts/rescore_github.py](../baseline/scripts/rescore_github.py)로 재채점해 확인):
- 공식 MMMU `eval_open()`은 응답 어디에든 정답 문자열이 나오면 정답으로 친다. 최종 답이 틀렸는데도 정답 처리된 경우가 8건이었다.
- 공식 객관식 파서는 답을 못 찾으면 응답 전체에서 글자를 찾고, 그래도 없으면 `random.choice`로 고른다. 잘린 응답에서 찍어 맞힌 경우와 랜덤 선택이 14건이었다.

**Baseline(Run B) 실측 통계**: rule 실패 112/900(12.4%) → judge가 살린 것 16건 → 최종 미해결 96건(10.7%)은 오답 처리. rule만으로 61.78%, judge까지 합치면 62.56%이다.

---

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 0.6667 |
| 2 | Agriculture | 30 | 0.5667 |
| 3 | Architecture_and_Engineering | 30 | 0.4333 |
| 4 | Art | 30 | 0.5333 |
| 5 | Art_Theory | 30 | 0.7667 |
| 6 | Basic_Medical_Science | 30 | 0.7000 |
| 7 | Biology | 30 | 0.6000 |
| 8 | Chemistry | 30 | 0.5333 |
| 9 | Clinical_Medicine | 30 | 0.5667 |
| 10 | Computer_Science | 30 | 0.6000 |
| 11 | Design | 30 | 0.7667 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 0.4000 |
| 13 | Economics | 30 | 0.8000 |
| 14 | Electronics | 30 | 0.3667 |
| 15 | Energy_and_Power | 30 | 0.4667 |
| 16 | Finance | 30 | 0.7000 |
| 17 | Geography | 30 | 0.5333 |
| 18 | History | 30 | 0.8000 |
| 19 | Literature | 30 | 0.8667 |
| 20 | Manage | 30 | 0.6333 |
| 21 | Marketing | 30 | 0.8667 |
| 22 | Materials | 30 | 0.5000 |
| 23 | Math | 30 | 0.6000 |
| 24 | Mechanical_Engineering | 30 | 0.3667 |
| 25 | Music | 30 | 0.1667 |
| 26 | Pharmacy | 30 | 0.8333 |
| 27 | Physics | 30 | 0.8333 |
| 28 | Psychology | 30 | 0.7667 |
| 29 | Public_Health | 30 | 0.8333 |
| 30 | Sociology | 30 | 0.7000 |
| | **Overall (macro avg)** | **900** | **0.6256** |

**계산식**: `Overall = mean(30개 과목 accuracy)`. 각 과목 accuracy = 정답 수 / 30이므로 `Overall = (과목별 정답 수의 합 / 30) / 30 = 563 / 900 = 0.6256` (62.56%).
- 과목별 정답 수의 합은 563이다 (원본 기록: [baseline/reports/runB_subject_accuracy.csv](../baseline/reports/runB_subject_accuracy.csv)).
- 과목당 문제 수가 모두 30개로 같아서 macro 평균과 전체 900문제 중 맞힌 비율(micro)이 일치한다.

---

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 62.56 |
| 차이 (Δ) | **−4.84%p** |

---

## 7. 격차 분석

**격차(−4.84%p, 약 44문항)의 가장 큰 원인은 응답 잘림(반복 루프)이라고 판단한다.**

1. **잘림**: 108문항(12.0%)이 max_new_tokens 32,768에 도달했고(`finish_reason=length`) 정답률은 약 7%다. 최종 `Answer:` 줄이 없어 rule 파서가 실패하고(rule 실패 112건), judge가 살린 것은 16건뿐이다. 처음 4,096토큰으로 돌렸을 때도 파싱 실패 129건 중 128건이 잘린 응답이었다. 즉 "잘림"과 "파싱 실패"는 별개 원인이 아니라 같은 현상이다.
2. **프롬프트(추가 실험)**: 나머지 조건을 같게 두고 첫 실험(Run A, 팀 GitHub 프롬프트 "Keep the reasoning concise")과 비교했다. 첫 실험 63.11%(568/900), Baseline 62.56%(563/900)로 5문항 차이, McNemar p=0.756이라 유의하지 않다. 잘림은 첫 실험 70건, Baseline 108건이고 Baseline은 Music 15건, Architecture 16건에 몰렸다. 둘 다 끝까지 답한 765문항에서는 Baseline 71.1%가 첫 실험 69.5%보다 높아, 차이는 추론 품질이 아니라 루프 빈도에서 나온다.
3. **샘플링 노이즈**: temperature 0.7이라 900문항 1회 측정의 95% 신뢰구간이 약 ±3.1%p다. 격차 4.84%p가 이 폭과 비슷한 수준이어서 전부를 구조적 원인으로 단정하지 않는다.
4. **채점(가설)**: 처음 GitHub 파서는 잘린 응답도 추정해서 채점했고(562/900), 같은 응답을 우리 파서로(judge 없이) 다시 채점하면 534/900으로 28문항 줄었다. 공식 평가가 비슷하게 관대한 파서를 썼다면 우리 수치가 낮게 나올 수 있다. 확인하지는 못했다.

**한계**: 67.4의 정확한 평가 설정(프롬프트·파서)은 확인하지 못했다. 잘림을 줄이면 얼마나 회복되는지는 아직 실험하지 못했다.

---

## 8. 기타 특이사항 / 한계

1. **단일 실행**: 한 번만 측정했고 temperature 0.7 샘플링이라 다시 돌리면 점수가 달라질 수 있다. fine-tuning 전/후 비교에는 문항 단위 검정([baseline/scripts/compare_runs.py](../baseline/scripts/compare_runs.py)의 McNemar)을 함께 쓴다.
2. **RTX 4090 미실측**: baseline은 A100 40GB에서만 측정했다. 4090 가능성은 KV cache 산정에 근거한 추정이다(§3.2).
3. **judge 의존**: judge(Llama-3.1-8B-Instruct)는 gated 모델이라 HF 라이선스 동의와 토큰이 필요하다. judge 없이 돌리려면 `--no_judge`를 쓰면 되고, rule 파서가 실패한 문항은 오답 처리된다(rule만 61.78%).
4. **입력 이미지 제외**: `<image N>` 표시가 없는 해설 이미지(4문항)는 입력에서 뺐다(§2).
5. **못 해본 것**: 잘림(반복 루프)을 줄이는 실험(응답 길이 지시, 반복 억제 등). fine-tuning의 목표로 남겨 둔다.
6. 설계 선택의 상세한 근거: [baseline/README_mmmu_eval.md](../baseline/README_mmmu_eval.md), [baseline/reports/baseline_decision.md](../baseline/reports/baseline_decision.md).
