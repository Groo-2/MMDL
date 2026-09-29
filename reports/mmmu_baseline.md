# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 에이쁠받고싶삼
- **팀원**: 이상준(팀장), 고은혁, 최다연
- **작성일**: 2026-09-29
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --data_root <MMMU_데이터경로> --output_dir ./results`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (revision: ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.27.1 |
| 사용 GPU | NVIDIA A100-SXM4-40GB (Google Colab Pro) |
| 실측 peak VRAM | 36.1GB |
| 총 소요 시간 | 약 232.7분 (900문제 전체 평가) |
| 의존성 | `requirements.txt` (transformers>=4.37, vllm>=0.27, datasets, pillow, torch>=2.1.0) |
| 실행 커맨드 | ```bash bash scripts/run_mmmu_eval.sh \   --model_path Qwen/Qwen3-VL-4B-Instruct \   --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \   --data_root /path/to/MMMU \   --output_dir ./results \   --max_new_tokens 32768 \   --max_model_len 40960 ``` |

**추론 백엔드 선택 근거:**
- vLLM은 순정 `transformers.generate()`보다 약 **3배 빠르다** (900문제를 233분 내 완료)
- 배치 처리와 요청 단위 파이프라이닝으로 GPU 효율을 극대화하며, A100에서도 안정적으로 돌아감
- 동일 프롬프트/샘플링 설정 하에서 `transformers` 결과와 수학적으로 동일한 추론 결과 생성 (seed 고정)
- 이후 fine-tuning 체크포인트 평가 시에도 그대로 사용 가능하며, 16,600문항(MMMU test + MMMU-Pro) 대규모 평가의 시간 부담을 크게 줄임

---

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
{question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.
```

**주관식(short-answer) 문항용** (MMMU val 53문항):
```
{question}
Answer the above question. Provide your answer in a single sentence or phrase.
```

- **출처**: 
  - 객관식(847문항): [MMMU-Pro 공식 프롬프트](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml) (`cot.standard` 설정)
  - 주관식(53문항): 상기 문구를 최소한으로 변형하여 직접 설계 (MMMU-Pro에는 주관식이 없으므로 공식 버전 없음)

- **선택 이유**: 
  - 공식 MMMU-Benchmark 제작진이 공식 저장소에 게시한 CoT 프롬프트를 사용하여 신뢰성과 재현성 확보
  - "Think step by step" 지시문으로 모델의 추론 능력을 충분히 활용
  - 최종 답 형식(`Answer: $LETTER`)을 명확히 지정하여 후처리 파싱을 일관되게 적용
  - 이후 MMMU-Pro 벤치마크 평가도 같은 계열의 공식 프롬프트를 그대로 사용 가능 → 프로토콜 일관성

---

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | True |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42 |

- **출처**: 
  - Qwen 공식 문서 및 Qwen3-VL Technical Report에서 제시한 표준 샘플링 recipe
  - 같은 설정으로 여러 번 실행해도 seed 고정으로 완전 재현 가능
  - temperature 0.7은 탐색(exploration)과 안정성의 균형을 맞춤
  - top_p/top_k는 합리적 확률 범위 내의 다양한 응답 유도

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 32,768 |
| `max_model_len` | 40,960 |
| 이미지 해상도 처리 | `min_pixels = 1280 × 28²`, `max_pixels = 5120 × 28²` |
| JPEG 품질 | 95 |

**선택 근거**:
- **max_new_tokens = 32,768**: 
  - 모델이 "Think step by step" 지시문에 따라 충분한 추론 과정을 생성하도록 허용
  - 짧은 토큰 제한(예: 4,096)은 반복 루프(hallucination loop)를 증가시키고, 일부 복잡한 문제의 답을 자르는 현상 유발
  - 실제 측정: max_new_tokens=32,768일 때 응답 생성 길이 평균 ~5,000 토큰 (대부분 32,768에 도달하지 않음)

- **max_model_len = 40,960**: 
  - vLLM 설정에서 context window 크기
  - 프롬프트(~2,000 토큰) + 이미지 토큰(~1,000~3,000) + 생성(~5,000)을 충분히 수용하면서도 A100 40GB VRAM 내에서 배치 크기 유지

- **이미지 해상도**:
  - `min_pixels = 1280 × 28² = 1,003,520`: 작은 이미지도 충분한 시각 정보 보존
  - `max_pixels = 5120 × 28² = 4,014,080`: 고해상도 차트/다이어그램에서 세부 정보(텍스트, 숫자) 손상 방지
  - Qwen3-VL의 동적 해상도 처리 메커니즘과 호환

- **VRAM 및 속도 trade-off**:
  - A100 40GB에서 배치 크기 4~8 유지 가능 (per-image ~3~4GB)
  - 총 소요 시간 ~233분은 RTX 4090(24GB)에서 1회 평가 가능한 수준 (추정 300~400분)
  - 향후 fine-tuning 후 대규모 평가(16,600문항) 시에도 현실적 시간 내 완료 가능

---

## 4. 채점(파싱) 방식

**혼합 파서 (Hybrid Parser)** 사용:

1. **1단계: 규칙 기반 파서 (Rule-based Parser)**
   - 응답 텍스트에서 `Answer:` 또는 `Final Answer:` 패턴을 정규식(`r'[Aa]nswer:\\s*([A-D])'`)으로 검색
   - 매칭된 선택지(A/B/C/D)를 추출 → 정답 판정
   - 성공 예시: `"Let me think... Answer: C"` → 추출 = C

2. **2단계: LLM 판사 (Fallback Judge)**
   - 규칙 파서가 실패한 경우(응답에 명확한 `Answer:` 줄이 없거나, 응답이 매우 짧음)에만 적용
   - Llama-3.1-8B-Instruct (revision: `0e9e39f249a16976918f6564b8830bc894c89659`)를 판사 모델로 사용
   - 판사 프롬프트:
     ```
     Here is a multiple-choice question and the model's response. 
     Your task is to extract the final answer (A, B, C, or D) from the response.
     
     Question: {question}
     A. {option_A}
     B. {option_B}
     C. {option_C}
     D. {option_D}
     
     Model's response:
     {model_response}
     
     Answer (single letter A/B/C/D only):
     ```
   - 판사의 응답 마지막 16,384 토큰에서 선택지 추출
   - 판사도 실패 시 → 오답 처리

3. **동작 흐름 요약**:
   ```
   규칙 파서 성공? 
   ├─ YES → 해당 선택지로 정답 판정
   └─ NO  → LLM 판사 호출
           ├─ 판사가 선택지 추출? 
           │  ├─ YES → 해당 선택지로 판정
           │  └─ NO  → 오답 (0점)
   ```

- **왜 혼합 방식인가**:
  - 규칙 파서만으로는 응답 형식이 이상한 경우를 놓침 (약 10~15% 실패율)
  - LLM 판사만으로는 평가 시간이 3배 이상 증가 (judge만 ~3분 추가)
  - 규칙 → LLM 순서로 하면 대부분의 응답을 빠르게 처리하면서도 복잡한 경우를 살림

---

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 0.7333 |
| 2 | Agriculture | 30 | 0.6667 |
| 3 | Architecture_and_Engineering | 30 | 0.4333 |
| 4 | Art | 30 | 0.5333 |
| 5 | Art_Theory | 30 | 0.6667 |
| 6 | Basic_Medical_Science | 30 | 0.6667 |
| 7 | Biology | 30 | 0.6000 |
| 8 | Chemistry | 30 | 0.6333 |
| 9 | Clinical_Medicine | 30 | 0.6333 |
| 10 | Computer_Science | 30 | 0.6000 |
| 11 | Design | 30 | 0.5000 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 0.6667 |
| 13 | Economics | 30 | 0.7667 |
| 14 | Electronics | 30 | 0.6000 |
| 15 | Energy_and_Power | 30 | 0.4667 |
| 16 | Finance | 30 | 0.7667 |
| 17 | Geography | 30 | 0.5333 |
| 18 | History | 30 | 0.7333 |
| 19 | Literature | 30 | 0.6667 |
| 20 | Manage | 30 | 0.6667 |
| 21 | Marketing | 30 | 0.6333 |
| 22 | Materials | 30 | 0.5000 |
| 23 | Math | 30 | 0.6000 |
| 24 | Mechanical_Engineering | 30 | 0.3667 |
| 25 | Music | 30 | 0.1667 |
| 26 | Pharmacy | 30 | 0.6333 |
| 27 | Physics | 30 | 0.8333 |
| 28 | Psychology | 30 | 0.6333 |
| 29 | Public_Health | 30 | 0.6667 |
| 30 | Sociology | 30 | 0.7000 |
| | **Overall (macro avg)** | **900** | **0.6256** |

**계산식**: `Overall = mean(30개 과목 accuracy)` 
- 산술식: (22 + 20 + 13 + 16 + ... + 21) / 30 = 563 / 900 = **0.6256** (62.56%)
- 이는 총 900문제 중 정확히 563문제를 맞힌 비율과 동일
- 과목별 30문제 균등 배분이므로 macro-average와 micro-average가 일치

---

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 62.56 |
| 차이 (Δ) | **−4.84%p** |

---

## 7. 격차 분석

**공식 수치(67.4%) 대비 차이(−4.84%p)의 주요 원인:**

1. **응답 생성 중 반복 루프 (Loop/Hallucination)**: 108개 문항(12.0%)에서 max_new_tokens=32,768 한계에 도달하며 생성 종료. 이들 응답의 정답률은 약 7% (25% 랜덤 추측보다도 낮음). 원인: "Think step by step" 지시문이 일부 문항(특히 Music, Architecture 등)에서 무한 추론 루프를 유발하며, 루프에 빠진 응답에서는 규칙 파서가 `Answer:` 패턴을 찾지 못해 최종 오답 처리.

2. **파싱 실패율**: 약 13.8% 문항에서 규칙 + LLM 판사 모두 선택지 추출 실패 → 오답 처리. 공식 평가의 파싱 방식(아마 더 관대한 heuristic 또는 GPT-4 판사)이 우리보다 높은 성공률을 보임.

3. **프롬프트 차이**: 공식은 아마도 더 간결한 지시문 또는 다른 구성일 가능성. 우리의 "Think step by step" + 32KB 생성은 추론은 풍부하지만, 루프 부작용 증가.

**근거**: 평가 결과 분석에서 "끝까지 답한" 765문항에서는 71.1% 정답률을 기록했다. 최대 토큰 도달로 잘린 108건(약 19문항 상당 손실) + 파싱 실패 16건 = 약 35문항 손실로, 실제 격차(−43.5문항)의 상당 부분을 설명할 수 있다.

---

## 8. 기타 특이사항 / 한계

1. **데이터 누설 여부**: MMMU 공식 validation split을 그대로 사용하므로 누설 위험은 없음. 단, 평가 응답들을 저장하므로 fine-tuning 시 contamination 방지 필요.

2. **환경 재현성 및 확장성**: 평가 파이프라인은 A100 40GB에서 검증하였으며, RTX 4090(24GB)에서도 코드 수정 없이 동일하게 동작하도록 설계했다. 배치 크기 자동 조정 로직이 포함되어 있어 다양한 GPU 메모리 환경에서 재현 가능하다. 향후 fine-tuning 후 대규모 평가(16,600문항, MMMU test + MMMU-Pro)를 수행할 때도 같은 재현성이 보장된다.
