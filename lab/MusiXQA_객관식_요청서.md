# MusiXQA 객관식 변환 요청서

**요청 내용:** MusiXQA(`puar-playground/MusiXQA`)의 자유 서술형 QA를 **4지선다 객관식**으로 바꾼 학습 데이터를 만들어 주세요.
**가장 중요한 조건:** 우리가 이미 쓰고 있는 SSMR-Bench 변환 결과와 **같은 형식**이어야 합니다. 그래야 문항 선별, 학습, 평가 코드를 고치지 않고 그대로 쓸 수 있습니다. 형식은 자동 검증기(`musixqa` 브랜치의 `lab/validate_mcq_dataset.py`)가 확인하니, 완성 전에 직접 돌려 보세요.

---

## 1. 배경 (왜 필요한가)

- 목표: Qwen3-VL-4B의 MMMU **Music** 점수를 올리는 것. 현재 baseline은 Music val 30문항 중 5문항(17%)입니다.
- 오답 분석에서 가장 큰 원인은 **악보를 읽는 능력**(음자리표, 조표, 오선 위 음 높이를 읽는 것)이었습니다. 이론 지식만 부족한 것이 아니라 악보 판독이 병목입니다.
- 그래서 악보 이미지를 읽는 연습 데이터가 필요합니다. MusiXQA는 이미지와 정확한 라벨이 있지만 **자유 서술형**이라, 같은 방식으로 학습하는 SSMR-Bench처럼 객관식으로 바꿔야 합니다.
- 학습 방법(강화학습 또는 지도학습)은 아직 확정 전이지만 **데이터 형식은 어느 쪽이든 같습니다.**
- 교수님 데이터 조건(대학 수준 난이도, MMMU 형식, MMMU 평가 데이터가 아닐 것)에 맞추기 위해 MMMU 형식(이미지 + 질문 + 선택지 + 정답)을 지킵니다.

## 2. 원본 데이터

| 항목 | 내용 |
|---|---|
| 위치 | https://huggingface.co/datasets/puar-playground/MusiXQA (MIT, 로그인 불필요) |
| 기준 커밋 | `d4cae4a5404b9c5d6dc8b85cce4e4c5d3585aae6` (이 커밋을 기준으로 만들고 manifest에 기록) |
| 파일 | `images.tar`(8.7GB, PNG), `train_qa_simple.json`(122MB), `test_qa_simple.json`(14MB), `train_qa_omr.json`(1.1GB), `test_qa_omr.json`(14MB), `metadata.json`(1.3GB) |
| 규모 | 악보 문서: train 86,400장, test 9,600장. 이미지는 **한 페이지 전체**(1191×1684, 피아노 대보표) |
| QA 한 줄 | `{"doc_id": "0086400", "question": "...", "answer": "...", "encode_format": "beat"}` (`doc_id`가 이미지 파일명 `images/0086400.png`와 대응) |

필요한 이미지만 `images.tar`에서 꺼내면 됩니다 (전체 압축 해제 불필요). 문항 4천 개면 문서 약 2천 장, 약 200MB입니다.

## 3. 만들어야 할 파일

```
musixqa_mcq_v1/
├── train.jsonl            # 학습 후보 풀
├── holdout.jsonl          # 평가용 (원본 test 문서에서만)
├── images/                # 위 두 파일이 참조하는 원본 PNG (리사이즈/재압축 금지)
├── manifest.json          # 버전, 날짜, seed(42), 원본 커밋, 카테고리별 문항 수, 사용한 라이브러리 버전
├── build_musixqa.py       # 위 데이터를 처음부터 다시 만드는 스크립트 (같은 seed면 같은 결과)
└── validation_report.json # 검증기가 만든 결과
```
전달은 zip 하나(`musixqa_mcq_v1.zip`)로 해 주세요.

## 4. 한 줄(JSON)의 형식 — SSMR 결과와 똑같이

**실제 SSMR 변환 결과** (이 형식이 기준입니다):
```json
{"id": "train_00002", "question": "<image 1> Select the most suitable key for the following musical score.", "options": ["A#", "A", "Abm", "Ab"], "question_type": "multiple-choice", "answer": "D", "image_paths": {"image_1": "images/train_00002_q.png"}, "category": "Scale", "class_name": null, "source_split": "train", "source_index": 2}
```

**MusiXQA용** (문서 `0002819`의 실제 라벨로 만든 예시. 이 문서는 마디 14개, 4/4, 조표 샵 5개, 템포 98입니다. 이미지와 대조해 확인했습니다):
```json
{"id": "musixqa_train_00001", "question": "<image 1> How many sharps or flats are in the key signature of this piece?", "options": ["3 sharps", "4 sharps", "5 sharps", "5 flats"], "question_type": "multiple-choice", "answer": "C", "image_paths": {"image_1": "images/0002819.png"}, "category": "KeySignature", "source_doc_id": "0002819", "source_split": "train"}
```

| 필드 | 규칙 |
|---|---|
| `id` | 학습은 `musixqa_train_00001`부터, 평가는 `musixqa_holdout_00001`부터. 전체에서 유일 |
| `question` | **반드시 `<image 1> `(공백 한 칸 포함)으로 시작.** 질문만 읽고도 무엇을 답할지 알 수 있게 쓰기. 줄바꿈 금지 |
| `options` | **정확히 4개, 텍스트만**. 이미지 선택지 금지. 앞뒤 공백/줄바꿈 금지. 서로 달라야 함 |
| `question_type` | 항상 `"multiple-choice"` |
| `answer` | `"A"`~`"D"` 중 정답 위치. `options`는 섞은 뒤의 순서 기준 |
| `image_paths` | `{"image_1": "images/<doc_id>.png"}` 하나만. 폴더 기준 상대경로 |
| `category` | 아래 5장의 카테고리 이름 |
| `source_doc_id` | 원본 `doc_id` 문자열 (문서 단위 분리 검사에 필요) |
| `source_split` | 학습 `"train"` / 평가 `"test"` (원본 MusiXQA의 분할 이름) |

**선택지를 섞는 규칙:** seed 42의 난수로 섞고, 정답 위치 A/B/C/D가 전체와 카테고리별로 각각 약 25%가 되게 하세요 (검증기가 20~30%를 벗어나면 오류로 처리).

## 5. 어떤 문제를 만들까

악보를 **읽어야** 풀리는 문제만 만들어 주세요. 아래 표 순서가 우선순위입니다.

### 5-1. 반드시 (Tier 1)

| 카테고리 | 질문 예 | 정답의 출처 | 오답(선택지) 만드는 법 |
|---|---|---|---|
| `KeySignature` | "How many sharps or flats are in the key signature of this piece?" | `What scale is this music in?` 라벨에서 계산 (music21 `key.Key(...).sharps`) | 정답 ±1, ±2개와 **샵↔플랫이 뒤바뀐 값** 하나. 예: 정답 `5 sharps` → `3 sharps`, `4 sharps`, `5 flats`. 개수는 0~7 안에서만, 0개는 `no sharps or flats`로 표기 (1개는 `1 sharp`) |
| `MeasureCount` | "How many measures are in this music sheet?" | `How many measures…` 라벨 (9~19) | 정답 ±1~±3. 정답이 항상 가운데 값이 되지 않게 비대칭으로 |
| `PitchReading` | "Which sequence of pitches is written in bar 2 of the treble clef?" (낮은음자리표도 섞기) | `Please extract the pitch of all notes in the Nth bar of the … clef.` 라벨. 형식 예: `B4 B4 E5 E5 B4 B4 E5 G#4` | 아래 5-3 참고 |
| `Clef` | "Which clef(s) does this music sheet use?" | `What clef is included…` 라벨 (둘 다 50%, 높은음자리표만 38%, 낮은음자리표만 12%) | `treble clef only`, `bass clef only`, `both treble and bass clefs` + 한 개를 다른 표현으로 (`alto clef only`) |

### 5-2. 가능하면 (Tier 2, 2단계 문제: 읽고 나서 이론 적용)

| 카테고리 | 질문 예 | 만드는 법 |
|---|---|---|
| `Interval` | "What is the interval between the first two notes of bar 3 in the treble clef?" | 위 `PitchReading`의 정답 음 높이 두 개를 music21 `interval.Interval`로 계산. 정답 예 `Major third`. 오답은 같은 숫자의 다른 종류(`Minor third`)와 이웃 숫자(`Perfect fourth`) |
| `HighestNote` | "What is the highest note in bar 4 of the treble clef?" | `PitchReading` 정답에서 최고음. 오답은 다른 위치 음이나 한 칸 낮은 음 |

원본의 정답이 맞는지는 우리가 따로 검증할 수 없으니, **계산한 정답은 원본 라벨을 그대로 쓰고** 스스로 만든 값(조표 수, 음정, 최고음)은 music21로 다시 계산해 일치하는지 확인해 주세요.

### 5-3. `PitchReading` 오답 만드는 규칙 (이 데이터에서 가장 중요)

모델이 실제로 하는 판독 실수를 흉내 낸 오답을 만들어 주세요. 세 오답은 정답과 **1~2개 음만** 다르게 하고, 길이(음 개수)는 같게 합니다.

| 실수 유형 | 만드는 법 |
|---|---|
| 오선 위치 착각 | 한 음을 위나 아래 한 칸 (이름이 이웃 음으로) |
| 옥타브 착각 | 한 음의 옥타브를 ±1 |
| 조표/임시표 무시 | 한 음의 `#`/`b`를 빼거나 더함 |
| 음자리표 착각 | 모든 음을 반대 음자리표로 읽은 값 (**한 문항에 한 개까지만**) |

세 오답에 같은 유형을 쓰지 말고 문항마다 무작위로 조합하세요 (항상 같은 조합이면 유형으로 정답을 찍을 수 있습니다). 마디가 길면(음 9개 이상) 제외하세요.

### 5-4. 만들지 말 것

| 원본 질문 | 이유 |
|---|---|
| 제목, 작곡가 | 답이 무작위 문자열(`Cusl Pxxgce Jxzj…`)이라 음악 능력과 무관 |
| 템포(BPM), 코드 이름(`What is the chord of the N bar?`) | 악보에 인쇄된 숫자와 글자를 그대로 읽으면 풀려서 악보 판독이 아님 |
| 반복 기호 | 대부분(약 51%) "없음", 나머지는 특정 마디 번호로 변별력이 낮음 |
| 음계 구성음(`What are all the notes in the A Major scale?`) | 이미지 없이 풀리는 텍스트 지식 문제 |
| "조(scale) 이름"을 묻기 | **조표만으로는 나란한조(G# 단조 vs B 장조, 둘 다 샵 5개)를 구별할 수 없어** 정답이 모호해짐. 반드시 `KeySignature`(샵/플랫 개수)로 바꿔서 묻기 |
| 박자(time signature) | 원본이 3종뿐(4/4 40%, 3/4 40%, 2/4 20%)이라 변별력이 낮음. **꼭 넣고 싶다면 전체의 10% 이하**로만 |

## 6. 분량과 분리

| 구분 | 권장 | 비고 |
|---|---|---|
| `train.jsonl` | 카테고리당 **800문항 이상** (Tier 1 네 개면 3,200개 이상) | 우리가 모델로 풀어 보고 학습 신호가 있는 문항(약 1,120개)만 골라 쓰므로 후보는 넉넉해야 함 |
| `holdout.jsonl` | 카테고리당 **100문항 이상** | 원본 **test 문서에서만** 만들 것. 학습 문서와 겹치면 안 됨 |
| 문서당 문항 수 | **최대 2문항** | 같은 악보로 만든 문항이 많으면 비슷한 이미지로 쏠림 |
| 카테고리 비율 | 가능하면 균등 | `TimeSignature`를 넣는다면 10% 이하 |

## 7. SSMR 작업에서 실제로 겪은 결함 (똑같이 겪지 않도록)

1. **오답 하나에만 글자 그대로의 `\n`(백슬래시+n)이 붙어 있었습니다.** 모델이 "`\n`이 붙은 선택지는 오답"이라는 지름길을 배웁니다. 선택지에는 줄바꿈이나 `\n` 문자가 없어야 합니다.
2. **선택지의 표기가 섞여 있었습니다.** 예: 선택지가 `c#`, `c`, `Fb`, `F#`처럼 소문자와 대문자가 섞였습니다. 측정해 보니 강한 지름길은 아니었지만(정답이 대문자로 시작하는 비율 43%, 오답 40%) 표기만 보고 고를 수 있으니 네 선택지의 표기 규칙(대소문자, 공백, 기호)을 통일하세요.
3. **정답이 가장 긴 선택지이거나 숫자 선택지에서 항상 가운데 값인 경우.** 검증기가 경고합니다 (정답이 유일하게 가장 긴 비율 40% 초과, 숫자 선택지에서 중간 값인 비율 65% 초과).
4. **질문에 정답의 단서가 들어간 경우**(예: "5 sharps를 가진 이 악보의 조는?"). 질문에는 답을 알려 주는 정보를 넣지 마세요.
5. 같은 질문 + 같은 이미지인데 정답이 서로 다른 경우(라벨 모순). 검증기가 오류로 처리합니다.

## 8. 검증하는 방법

```bash
# musixqa 브랜치를 받아서 (검증기는 이 브랜치에 있습니다)
git clone -b musixqa https://github.com/Groo-2/MMDL.git
python MMDL/lab/validate_mcq_dataset.py --data-dir musixqa_mcq_v1
```
- **ERROR 0개**여야 합니다. 어기면 어느 줄(`id`)이 왜 틀렸는지 알려 줍니다.
- WARN은 통과하지만 이유를 확인해서 manifest나 메모에 적어 주세요 (예: "카테고리 X가 650개뿐").
- 검증기는 이미지가 실제로 열리는지, train/holdout이 같은 문서를 쓰지 않는지, 문서당 문항 수, 정답 위치 균형까지 확인합니다. **라벨이 이미지와 맞는지는 확인하지 못합니다.**
- 그 부분은 **카테고리마다 무작위 50문항을 이미지와 직접 대조**해 주세요. 정답과 오답의 정확성, 오답이 정답과 구별되는지(모호하지 않은지)를 봅니다. 결과(몇 개 중 몇 개 맞음)를 manifest에 적어 주세요.

## 9. 일정과 중간 점검

SSMR 학습이 약 2일 걸리니 그동안 진행하시면 됩니다. 다만 **전체를 만들기 전에 먼저 샘플을 보내 주세요.**

1. 카테고리마다 20문항씩(+ 이미지)만 만들어 검증기를 통과시킨 `musixqa_mcq_sample.zip`
2. 형식과 오답 규칙이 맞는지 저희가 확인
3. 확인되면 전체 생성

## 10. 제출 체크리스트

- [ ] `train.jsonl`, `holdout.jsonl`, `images/`, `manifest.json`, `build_musixqa.py`가 모두 있다
- [ ] 검증기: ERROR 0개, WARN은 이유를 적었다
- [ ] Tier 1 네 카테고리가 모두 있고, 카테고리당 train 800개 이상, holdout 100개 이상
- [ ] 제목, 작곡가, 템포, 코드 이름, 반복, 음계 구성음, "조 이름" 문제는 만들지 않았다
- [ ] 박자 문제는 없거나 10% 이하
- [ ] `PitchReading` 오답이 위의 실수 유형으로 만들어졌다 (임의의 무작위 음이 아님)
- [ ] 직접 대조한 50문항/카테고리의 결과를 manifest에 적었다
- [ ] 같은 seed(42)로 `build_musixqa.py`를 다시 돌리면 같은 파일이 나온다

원본 데이터셋의 출처 표기: Chen et al., *MusiXQA: Advancing Visual Music Understanding in Multimodal Large Language Models*, 2025 (arXiv:2506.23009), MIT License.
