# MMMU Validation Evaluation — Qwen3-VL-4B-Instruct

MMMU validation(30과목 × 30 = 900문항)으로 Qwen3-VL-4B-Instruct를 평가한다.
fine-tuning 후에도 **`--model_path`만 바꿔서 같은 커맨드로 다시 평가**하는 것이 목적이다
(assignment_guidance.md §0).

## 실행 (한 커맨드)

모든 명령은 저장소의 `baseline/` 폴더에서 실행한다 (`cd baseline`).

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # judge(Llama-3.1-8B-Instruct)가 gated 모델이라 필요 (HF에서 라이선스 동의 먼저)

bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root <MMMU HF 캐시 디렉터리> \
  --output_dir results/baseline_mmmu_pro_cot
```

- fine-tuning 체크포인트: `--model_path /path/to/ckpt`로 바꾸고 `--model_revision`은 빼면 된다
- 스모크 테스트: `--limit 2`(과목당 2문항) 또는 `--subject Accounting --limit 2`. **baseline 수치로 쓸 수 없음**
- 중간에 끊기면 **같은 커맨드를 다시 실행**하면 된다. 끝난 문항은 건너뛴다(resume)
- judge 없이 돌리기: `--no_judge` (rule 파서가 실패한 문항은 오답 처리)
- Colab: `colab_run_mmmu.ipynb`

## 구조

```
mmmu_eval/
├── run_mmmu_eval.py   # infer / judge / score 서브커맨드, 고정값(모델, 데이터 revision, sampling recipe)
├── prompting.py       # 프롬프트: mmmu_pro_cot(기본), github_cot(비교 실험용)
├── parsing.py         # 혼합 파서 rule 단계 ('Answer:' 줄 추출 + 주관식 비교)
└── judge_utils.py     # 혼합 파서 judge 단계 (Llama-3.1-8B-Instruct)
scripts/
├── run_mmmu_eval.sh   # 단일 진입점: 모델 서버 → infer → judge 서버 → judge → score
└── rescore_github.py  # 팀 GitHub baseline(562/900) 응답을 혼합 파서로 다시 채점 (GPU 불필요)
```

## 산출물 (`--output_dir`)

| 파일 | 내용 |
|---|---|
| `summary.json` | overall(macro), micro 검산, rule-only 점수, 카테고리별과 과목별 점수, 파서 경로 분포, 잘림 비율, 소요 시간, peak VRAM, 모든 설정값 |
| `subject_scores.csv` | 30과목 점수와 소요 시간 (보고서 §5 표) |
| `final_predictions.jsonl` | 문항별 응답, rule 예측, judge 결과, 최종 예측, 정오 (fine-tuning 후 McNemar 검정에 사용) |
| `failed_parses.jsonl` | rule 파서가 실패한 문항 (보고서 §7 격차 분석 재료) |
| `predictions.jsonl`, `judge_results.jsonl` | 원시 기록 (resume용) |
| `sessions.json` | 단계별 벽시계 시간 |
| `vllm_*.log`, `gpu_mem_*.log` | 서버 로그, nvidia-smi 5초 간격 VRAM 기록 |

## 파이프라인 선택과 출처 (보고서 §1–§4 재료)

### 고정값 (assignment_guidance.md §1.1, §1.2)
- 모델: `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 (`vllm serve --revision --dtype bfloat16`)
- 데이터: `load_dataset("MMMU/MMMU", <과목>, split="validation", revision="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")` × 30, 과목당 30문항 assert

### 프롬프트 (보고서 §2)
**MMMU-Pro 공식 CoT**, 출처: [`mmmu-pro/prompts.yaml`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml) `cot.standard`,
조립 방식: [`mmmu-pro/infer/infer_transformers.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_transformers.py) `construct_prompt()` / `replace_images_tokens()`

```
{question}
A. {option A}
B. {option B}
...
Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.
```
- `<image N>`는 `[image]`로 바꾸고, 이미지는 텍스트 **뒤에** 등장 순서대로 붙인다(공식과 동일)
- 텍스트에 `<image N>` 표시가 없는 이미지는 **넣지 않는다**(MMMU-Pro 공식, lmms-eval, 팀 GitHub 스크립트와 같은 처리)
  - 이유: MMMU validation에서 이런 이미지가 있는 4문항(Agriculture_26, Materials_15, Pharmacy_4, Pharmacy_22)을 확인해 보니 **해설 이미지**였다. 예: Materials_15의 풀이 그래프가 정답 −105°C를 점선으로 가리키고, Pharmacy_22의 풀이가 정답 "2,1"의 반응 차수를 계산해 보여준다. 넣으면 정답이 유출된다
  - MMMU test에는 이런 문항이 없다(10,500문항 전수 확인)
- [직접 변형] 주관식 53문항: MMMU-Pro는 객관식만 있어서 같은 문장 구조로 `Answer the preceding question. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes). Think step by step before answering.`
- 선택 이유: 벤치마크 제작진이 공개한 CoT 프롬프트라 출처가 분명하고, 나중에 평가할 MMMU-Pro에도 같은 계열을 그대로 쓸 수 있다

### 생성 설정 (보고서 §3)
| 파라미터 | 값 | 출처 / 근거 |
|---|---|---|
| temperature / top_p / top_k | 0.7 / 0.8 / 20 | Qwen 공식 [`evaluation/mmmu/infer_instruct.sh`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/infer_instruct.sh), README "Instruct Models" |
| repetition_penalty / presence_penalty | 1.0 / 1.5 | 위와 같음 |
| seed | 42 | 같은 저장소 `evaluation/mmmu/run_mmmu.py`의 `LLM(..., seed=42)`. 서버 `--seed 42`와 요청마다 `seed=42` |
| max_new_tokens | 32768 | 공식값. 4096에서는 GitHub baseline 900문항 중 128문항(14%)이 잘렸고, 파싱 실패 129건 중 128건이 잘린 응답이었다 |
| max_model_len | 40960 | 공식은 128000. 생성 32768 + 최대 프롬프트 약 5.7K면 약 38.5K 이상에서는 결과가 같다. 128000은 시작할 때 KV cache 약 17.6GiB가 필요해 RTX 4090에서 서버가 뜨지 않으므로 40960으로 정함 |
| min_pixels / max_pixels | 1280·28² / 5120·28² | Qwen 공식 `build_mmmu_prompt()`와 같음. 이미지는 JPEG q95로 전송 |
| dtype | bfloat16 | 양자화 없음 |

### 채점 (보고서 §4): 혼합 파서, 결정적
1. **rule**: 응답의 마지막 `Answer:` 뒤 첫 줄에서 답을 뽑는다(MMMU-Pro 공식 [`evaluate.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py) `parse_multi_choice_response()`의 1단계)
   - 객관식: 줄이 선택지 표기(`B`, `(B)`, `B.`)로 시작하면 그 글자, 아니면 줄 안의 유효한 선택지 대문자가 정확히 하나일 때 그 글자
   - 주관식: 숫자는 소수 둘째 자리 반올림 후 비교, 문자열은 정규화 후 완전 일치. 정답 목록 중 하나라도 맞으면 정답
2. **judge**: rule이 실패한 문항만 `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`에 보낸다(temperature 0, seed 42)
   - 객관식 프롬프트와 응답 해석: Qwen 공식 [`evaluation/mmmu/eval_utils.py`](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/eval_utils.py) `build_prompt()`, `can_infer()` 원문(VLMEvalKit과 같음)
   - 주관식: 최종 답만 뽑게 하는 직접 설계 프롬프트 → 1번과 같은 비교
   - 응답이 16,384 토큰을 넘으면 **마지막 16,384 토큰만** 넣는다(judge 서버를 4090에서도 띄울 수 있도록 max_model_len 20480)
3. judge도 실패하면 **오답**. 공식 코드의 `random.choice` fallback은 재현이 안 돼서 쓰지 않는다
- judge를 평가 대상(Qwen)과 다른 계열로 고른 이유: 같은 계열끼리 판정하는 편향을 피하기 위해
- `summary.json`에 `rule_only`와 `rule+judge` 점수를 모두 남긴다. 대표값은 `overall_accuracy`(rule+judge, macro)

### 공식 파서를 그대로 쓰지 않은 이유 (GitHub 응답 900개로 확인, `scripts/rescore_github.py`)
- 공식 MMMU `eval_open()`은 응답 **어디에든** 정답 문자열이 나오면 정답으로 친다 → 최종 답이 틀렸는데도 정답 처리된 경우가 8건
- 공식 객관식 파서는 답을 못 찾으면 응답 전체에서 글자를 찾고, 그래도 없으면 `random.choice`로 고른다 → 잘린 응답에서 찍어 맞힌 경우와 랜덤 14건

## Baseline 결정 절차
1. 스모크(`--limit 2`)로 속도와 VRAM 실측
2. `--prompt mmmu_pro_cot`로 900문항 전체 실행
3. 비교: GitHub 원래 62.44% / `rescore_github.py` 결과 / 2번 결과. 선택 사항: `--prompt github_cot`로 다른 조건을 같게 두고 실행(Run A)
4. 최종 baseline을 정한 뒤 프롬프트, 파서, judge, 생성 설정을 **동결**한다(fine-tuning 재평가 때 그대로 재사용)

## GPU 메모리 참고
- Qwen3-VL-4B KV cache ≈ 0.14 MiB/토큰(36층 × KV head 8개 × 128 × bf16 × K,V). 최대 길이 응답 하나(약 38.4K 토큰)는 약 5.3GiB
- A100 40GB: 최대 길이 응답을 4개 이상 동시에 처리 가능. RTX 4090: 동작은 하지만 동시 처리량이 적다
- judge(Llama-3.1-8B): 가중치 16.1GB + KV 약 2GB → 4090에서도 동작
