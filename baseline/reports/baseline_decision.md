# MMMU Baseline 결정: Run A vs Run B

- 작성일: 2026-09-25
- 대상: Qwen/Qwen3-VL-4B-Instruct @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, MMMU validation 900문항 (`MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`)
- 실행 환경: Colab Pro, NVIDIA A100-SXM4-40GB, vLLM 0.27.1, torch 2.13.0+cu130

## 1. 결론

> **최종 baseline은 Run B(`mmmu_pro_cot`, MMMU-Pro 공식 CoT 프롬프트)로 정한다. 점수는 563/900 = 62.56%.**
>
> Run A와의 점수 차이(+5문항)는 통계적으로 의미가 없다(McNemar p = 0.756). 그래서 **프롬프트 출처의 공신력**과 **이후 MMMU-Pro 평가와의 일관성**을 기준으로 골랐다.
> 이후 fine-tuning 체크포인트는 모두 이 설정 그대로 평가한다(`--model_path`만 교체).

## 2. 두 실행의 조건

프롬프트만 다르고 나머지 조건은 모두 같다.

| 항목 | Run A | Run B |
|---|---|---|
| 프롬프트 | `github_cot` (팀 GitHub `qwen-mmmu-62-44` 원문) | `mmmu_pro_cot` (MMMU-Pro 공식 `prompts.yaml` `cot.standard`) |
| 최종 답 형식 | `Final Answer: X` | `Answer: $LETTER` |
| 추론 길이 지시 | "Keep the reasoning concise" | "Think step by step" |
| 이미지 배치 | `<image N>` 위치에 끼워 넣음 | 텍스트 뒤에 등장 순서대로 (공식과 동일) |
| 공통 | sampling: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, **seed 42** (Qwen 공식) / max_new_tokens 32768 / max_model_len 40960 / min·max_pixels 1280·28², 5120·28² / 혼합 파서 + Llama-3.1-8B judge | 같음 |

### 프롬프트 원문

**Run A (github_cot)** — 객관식
```
{question}

A. {option A}
B. {option B}
...

Work through the problem carefully.
Keep the reasoning concise.
At the end of your response, output exactly one final-answer line in this format:

Final Answer: X

where X is one of the valid option letters: {A, B, C, ...}
```

**Run B (mmmu_pro_cot)** — 객관식 (출처: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml)
```
{question}
A. {option A}
B. {option B}
...
Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.
```

## 3. 결과

### 3.1 전체

| | Run A | Run B | 차이 (A−B) |
|---|---|---|---|
| **overall (rule + judge, macro)** | 568/900 = **63.11%** | 563/900 = **62.56%** | +5문항 (+0.56%p) |
| rule만 | 62.56% | 61.78% | +0.78%p |
| 잘림 (32768 토큰, `finish_reason=length`) | 70 (7.8%) | 108 (12.0%) | −38 |
| judge 성공 / 대상 | 15 / 70 | 16 / 112 | |
| 추론 시간 | 154.3분 | 232.7분 | −34% |
| judge 시간 | 2.8분 | 3.8분 | |
| peak VRAM (추론) | 36.0GB | 36.1GB | |

### 3.2 문항 단위 비교 (`scripts/compare_runs.py`)

| | 개수 |
|---|---|
| 둘 다 정답 | 483 |
| A만 정답 | 85 |
| B만 정답 | 80 |
| 둘 다 오답 | 252 |
| **McNemar 정확 검정** | **p = 0.756 → 차이 없음** |

참고로 900문항 한 번 실행의 95% 신뢰구간은 약 ±3.1%p다. temperature 0.7 샘플링이라 같은 설정으로 다시 돌려도 이 정도는 흔들린다.

### 3.3 차이가 어디서 나오나: 잘림(반복 루프)

| 문항 그룹 | 개수 | A 정답 | B 정답 | McNemar p | 해석 |
|---|---|---|---|---|---|
| 둘 다 끝까지 답함 | 765 | 532 (69.5%) | **544 (71.1%)** | 0.307 | 추론 품질은 **B가 +12** (유의하지는 않음) |
| B만 잘림 | 65 | **31** | 5 | <0.001 | B가 루프에 빠져 −26 |
| A만 잘림 | 27 | 2 | **13** | 0.003 | A가 루프에 빠져 −11 |
| 둘 다 잘림 | 43 | 3 | 1 | 0.625 | 둘 다 거의 0점 |

- A의 +5는 **전부 잘림 차이**에서 나온다. 잘림에서 A가 +15(26 − 11 + 2)이고, 끝까지 답한 문항에서는 B가 +12다.
- 잘린 응답의 정답률은 약 7%로 4지선다 찍기(25%)보다 낮다. 최종 답이 없어서 judge도 거의 살리지 못한다 → **잘림은 사실상 오답**이다.
- "Think step by step"(B)은 "Keep the reasoning concise"(A)보다 반복 루프를 더 자주 유발한다.

### 3.4 과목별 (차이가 큰 순)

| 과목 | A | B | A−B | A 잘림 | B 잘림 | p |
|---|---|---|---|---|---|---|
| Music | 12 | 5 | +7 | 6 | **15** | 0.092 |
| Geography | 22 | 16 | +6 | 0 | 3 | 0.109 |
| Energy_and_Power | 18 | 14 | +4 | 11 | 15 | 0.289 |
| Physics | 21 | 25 | −4 | 1 | 1 | 0.289 |
| Art | 19 | 16 | +3 | 1 | 2 | 0.250 |
| Computer_Science | 21 | 18 | +3 | 0 | 2 | 0.453 |
| Math | 15 | 18 | −3 | 4 | 3 | 0.250 |
| Architecture_and_Engineering | 15 | 13 | +2 | 7 | **16** | 0.754 |
| Mechanical_Engineering | 11 | 11 | 0 | 9 | 12 | 1.000 |
| Materials | 14 | 15 | −1 | 10 | 6 | 1.000 |

- 유의한(p < 0.05) 과목은 **하나도 없다.**
- Music 차이는 거의 B의 루프(30문항 중 15건 잘림) 때문이다.
- 나머지 20개 과목의 차이는 모두 ±2문항 이내이고, 표에서는 생략했다(전체는 `results/compare_A_vs_B.json`).

### 3.5 분야별 (MMMU 6개 분야)

| 분야 | Run A | Run B |
|---|---|---|
| Art and Design | 63.33% | 55.83% |
| Business | 69.33% | 73.33% |
| Science | 58.67% | 62.00% |
| Health and Medicine | 68.00% | 66.67% |
| Humanities and Social Science | 74.17% | 78.33% |
| Tech and Engineering | 51.90% | 47.14% |

## 4. 장단점

### Run A (github_cot)

**장점**
- 잘림이 적다(7.8% vs 12.0%). "Keep the reasoning concise"가 반복 루프를 줄인다.
- **34% 빠르다**(154분 vs 233분). 이후 약 16,600문항(MMMU test + MMMU-Pro)을 평가할 때 체크포인트당 약 47시간 대 71시간으로 차이가 커진다(A100 단순 비례 추정).
- 점수가 +0.56%p 높다. 다만 통계적으로 의미는 없다.
- 루프 영향이 큰 Music, Geography, Art에서 강하다.

**단점**
- **프롬프트 출처가 불명확하다.** 팀 GitHub 스크립트의 문구로, 다른 LLM이 추천했을 가능성이 있다. 보고서에는 "직접 설계"로 적어야 하고, 교수님 채점 C항목(프롬프트 전문과 출처, 10점)에서 근거가 약하다.
- 끝까지 답한 문항에서는 B보다 정확도가 낮다(69.5% vs 71.1%).
- **MMMU-Pro vision 설정에 그대로 쓸 수 없다**(5절).
- 공개된 평가 방식이 아니라 다른 연구 결과와 비교하기 어렵다.

### Run B (mmmu_pro_cot) ← 채택

**장점**
- **공식 출처**: MMMU 벤치마크 제작진이 공개한 CoT 프롬프트다. 조립 방식(`construct_prompt`, `replace_images_tokens`)까지 공식 코드와 같고, 객관식 847문항 전부 글자 단위로 일치함을 확인했다.
- **MMMU-Pro와 이어진다.** standard와 vision 설정 모두 같은 파일에 공식 프롬프트가 있어서 이후 평가도 공식 프로토콜 그대로 할 수 있다.
- 끝까지 답한 문항에서 정확도가 더 높다(71.1%).
- 공식 파서의 1단계(`Answer:` 줄)와 형식이 같아 파서 출처도 공식 코드로 설명된다.

**단점**
- **반복 루프로 인한 잘림이 많다**(108건, 12.0%). 특히 Music(15), Architecture(16), Energy(15), Mechanical(12)에 몰려 있다.
- 느리다(232.7분). 이후 대규모 평가 때 시간 부담이 크다.
- 주관식 53문항용 문구는 공식 문구를 최소한으로 바꾼 직접 변형이다(MMMU-Pro에는 주관식이 없음).

## 5. Run A 프롬프트를 MMMU-Pro에 그대로 쓸 수 있나?

**standard 설정은 가능하지만, vision 설정은 불가능하다.**

| MMMU-Pro 설정 | 데이터 필드 | Run A 적용 |
|---|---|---|
| standard (10 options) | `question`, `options`, `image_1..7` | ✅ 그대로 가능. 선택지가 A~J 10개로 늘 뿐 |
| standard (4 options) | 같음 | ✅ 그대로 가능 |
| **vision** | **`image`, `options`만 있음. `question` 텍스트가 없음** | ❌ 문제와 선택지가 **이미지 안에** 있다. Run A 템플릿은 `{question}`과 선택지 목록을 텍스트로 넣어야 해서 그대로는 못 쓴다 |

- vision 설정에 쓰려면 "이미지 속 문제를 풀어라"라는 **새 문구를 직접 만들어야** 한다. 공식은 `Write out the multiple-choice question in the image and then solve it. ...`이다. 그러면 standard와 vision이 서로 다른 출처의 프롬프트가 되어 문서화와 일관성이 약해진다.
- 규칙상으로는 문제없다. 교수님 규칙은 "baseline과 fine-tuning 평가의 프롬프트를 똑같이 유지"하라는 것이고, 벤치마크마다 다른 프롬프트를 쓰는 것 자체는 막지 않는다.
- 하지만 MMMU-Pro 공식 수치(Qwen3-VL 보고서의 53.2)와 비교하려면 공식 프로토콜이 유리하다. 그래서 MMMU와 MMMU-Pro 모두 **같은 계열의 공식 프롬프트(B)**를 쓰는 편이 설명하기 가장 깔끔하다.

## 6. 채택 근거 요약

1. **점수 차이가 없다**: McNemar p = 0.756. 과목별로도 유의한 차이가 없다.
2. **출처**: 교수님 가이드 §2.2(a)와 채점 C항목은 프롬프트 출처를 요구한다. B는 공식 저장소 링크로 인용할 수 있다.
3. **일관성**: MMMU-Pro(standard, vision)까지 같은 계열의 공식 프롬프트로 평가할 수 있다.
4. **추론 품질**: 끝까지 답한 765문항에서 B가 544 대 532로 앞선다.
5. B의 약점(루프로 인한 잘림 12%)은 숨기지 않고 **격차 분석(§7)과 fine-tuning 목표**로 쓴다.
   - 잘린 108문항은 거의 전부 오답이다 → 공식 67.4와 격차가 생기는 주요 원인 중 하나다.
   - fine-tuning으로 루프가 줄면 개선으로 측정된다. 다만 "지식 향상"과 "루프 감소"를 구분해서 보고하도록 잘림 비율을 함께 제시한다.

## 7. 동결 설정 (fine-tuning 재평가 때 그대로 사용)

| 항목 | 값 |
|---|---|
| 프롬프트 | `mmmu_pro_cot` |
| 파서 | 혼합 방식: `Answer:` 줄 → Llama-3.1-8B-Instruct judge(@ `0e9e39f249a16976918f6564b8830bc894c89659`, 응답 끝 16,384 토큰) → 오답 |
| sampling | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, seed 42 |
| 생성 예산 | max_new_tokens 32768, max_model_len 40960 |
| 이미지 | min_pixels 1280·28², max_pixels 5120·28², JPEG q95, `<image N>` 표시가 없는 이미지는 제외(해설 이미지라 정답 유출) |
| 재현 커맨드 | `bash scripts/run_mmmu_eval.sh --model_path <모델 또는 체크포인트> [--model_revision <sha>] --data_root <경로> --output_dir <경로>` |

## 8. 참고 수치

| 실행 | 설정 | 점수 |
|---|---|---|
| 팀 GitHub 원래 결과 | GitHub CoT, seed 3407, max_new_tokens 4096, GitHub 파서(잘린 응답도 추정해서 채점) | 562/900 = 62.44% |
| 같은 응답을 혼합 파서로 다시 채점(judge 없음) | | 534/900 = 59.33% |
| Run A | GitHub CoT, seed 42, 32768, 혼합 파서 + judge | 568/900 = 63.11% |
| **Run B (baseline)** | **MMMU-Pro CoT, seed 42, 32768, 혼합 파서 + judge** | **563/900 = 62.56%** |
| Qwen3-VL Technical Report | 공식 수치 | 67.4 |
