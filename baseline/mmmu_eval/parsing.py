"""혼합 파서 (rule 단계). 결정적이고 랜덤이 없다.

흐름
  객관식: 마지막 'Answer:' 줄에서 선택지 글자 추출 → 실패하면 judge → judge도 실패하면 오답
  주관식: 마지막 'Answer:' 줄 추출 → 정답 목록과 비교 → 추출 실패하면 judge → judge도 실패하면 오답

이 모듈은 rule 단계만 담당한다. judge 단계는 judge_utils.py에 있다.

출처
  - 'Answer:' 줄 기준 추출: MMMU-Pro 공식 mmmu-pro/evaluate.py parse_multi_choice_response()의 1단계
    (response.rfind("Answer:"))
    https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py
  - 공식 2단계 이후(응답 전체에서 글자나 선택지 문구를 찾고, 없으면 random.choice)는 쓰지 않는다.
    GitHub 응답 900개로 확인한 결과 잘린 응답에서 찍어 맞히거나 랜덤으로 고르는 경로였고(재현 불가),
    그 자리를 judge로 대체했다.
  - 주관식 비교(clean_answer_text / parse_numeric_value / open_answers_equal): 팀 GitHub
    run_mmmu_reasoning_4096.py 원문. 공식 MMMU eval_open()은 응답 어디에든 정답 문자열이 나오면
    정답으로 쳐서 오판이 많았으므로(900개 중 8건) 쓰지 않는다.
"""

import ast
import math
import re


ANSWER_MARKER = "Answer:"


# ============================================================
# 1. 'Answer:' 줄 추출
# ============================================================

def extract_answer_line(response):
    """응답에서 마지막 'Answer:' 뒤의 첫 번째 비어 있지 않은 줄을 돌려준다. 없으면 None.

    공식 파서는 'Answer:' 뒤 문자열 전체를 보지만, 뒤에 설명이 이어지면 다른 글자가 섞여
    실패하므로 첫 줄만 본다. ("**Answer:** B", "Final Answer: B"도 'Answer:'를 포함하므로 잡힌다.)
    """

    if not response:
        return None

    pos = response.rfind(ANSWER_MARKER)

    if pos == -1:
        return None

    for line in response[pos + len(ANSWER_MARKER):].splitlines():

        if line.strip():
            return line.strip()

    return None


# ============================================================
# 2. 문자열 정리 (GitHub 원본 clean_answer_text + LaTeX 래퍼 제거 보강)
# ============================================================

def clean_answer_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = text.replace("**", "")
    text = text.replace("`", "")

    text = text.replace("−", "-")
    text = text.replace("–", "-")

    # [보강] 문장 중간의 \boxed{...}, \text{...}도 벗긴다 (원본은 전체가 \boxed{}일 때만 처리)
    text = re.sub(r"\\(?:boxed|text|mathrm)\{([^{}]*)\}", r"\1", text)

    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text, flags=re.DOTALL)

    if boxed:
        text = boxed.group(1).strip()

    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. 객관식
# ============================================================

LEADING_LETTER = re.compile(r"^(?:\(([A-Z])\)|([A-Z])(?=$|[\.\):,]))")
STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def parse_multiple_choice(response, num_options):
    """'Answer:' 줄에서 선택지 글자를 뽑는다. 실패하면 None (→ judge 대상).

    1) 줄이 선택지 표기로 시작하면 그 글자  (예: "B", "(B)", "B. 12 m/s", "B: ...")
       "A or B"처럼 글자 뒤에 바로 단어가 오면 표기가 아니므로 2)로 넘긴다
    2) 아니면 줄 안의 독립된 대문자 중 유효한 선택지가 정확히 하나일 때 그 글자
       (공식 파서의 '유일한 매칭' 규칙. 단, 부분 문자열이 아니라 단어 경계로 매칭한다)
    대문자만 인정한다. 소문자까지 받으면 관사 'a'를 A로 잘못 읽기 때문이다.
    """

    choices = [chr(ord("A") + i) for i in range(num_options)]

    line = extract_answer_line(response)

    if line is None:
        return None

    text = clean_answer_text(line)

    leading = LEADING_LETTER.match(text)

    if leading:

        letter = leading.group(1) or leading.group(2)

        if letter in choices:
            return letter

    found = {
        letter
        for letter in STANDALONE_LETTER.findall(text)
        if letter in choices
    }

    if len(found) == 1:
        return found.pop()

    return None


# ============================================================
# 4. 주관식 (GitHub 원본 비교 로직)
# ============================================================

def extract_open_answer(response):
    """'Answer:' 줄을 정리해서 돌려준다. 없거나 비었으면 None (→ judge 대상)."""

    line = extract_answer_line(response)

    if line is None:
        return None

    text = clean_answer_text(line)

    return text or None


def parse_gold_answers(gold):

    if isinstance(gold, (list, tuple)):
        return list(gold)

    if isinstance(gold, str):

        stripped = gold.strip()

        try:
            parsed = ast.literal_eval(stripped)

            if isinstance(parsed, (list, tuple)):
                return list(parsed)

        except (ValueError, SyntaxError):
            pass

        return [stripped]

    return [gold]


NUMBER_PATTERN = re.compile(
    r"[-+]?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+"
    r"|"
    r"\d+"
    r"|"
    r"\.\d+"
    r")"
    r"(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
)

FRACTION_PATTERN = re.compile(r"^[-+]?\d+\s*/\s*[-+]?\d+$")


def parse_numeric_value(text):

    if text is None:
        return None

    value = clean_answer_text(text).strip()

    if FRACTION_PATTERN.fullmatch(value):

        numerator, denominator = value.split("/")

        try:
            numerator = float(numerator.strip())
            denominator = float(denominator.strip())

            if denominator != 0:
                return round(numerator / denominator, 2)

        except ValueError:
            pass

    numbers = NUMBER_PATTERN.findall(value)

    if len(numbers) != 1:
        return None

    try:
        number = float(numbers[0].replace(",", ""))
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return round(number, 2)


def canonicalize_text(text):

    value = clean_answer_text(text).lower()
    value = value.strip("$")
    value = value.rstrip(" .,:;!")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def open_answers_equal(prediction, gold):
    """둘 다 숫자면 소수 둘째 자리 반올림 후 비교, 아니면 정규화한 문자열 완전 일치."""

    pred_num = parse_numeric_value(prediction)
    gold_num = parse_numeric_value(gold)

    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    pred_text = canonicalize_text(prediction)
    gold_text = canonicalize_text(gold)

    if not pred_text or not gold_text:
        return False

    return pred_text == gold_text


def open_is_correct(prediction, gold):
    """정답 목록 중 하나라도 일치하면 정답.

    [보강] 예측이 쉼표/세미콜론/'and'로 나뉘고 **모든 조각**이 각각 정답 목록 중 하나와 일치해도 정답
    (예: 예측 "Tampa, Florida", 정답 ['Tampa', 'Florida']). 조각 하나라도 틀리면 오답이므로
    "2, 3"처럼 여러 답을 나열해 찍는 경우는 인정하지 않는다.
    """

    if prediction is None:
        return False

    gold_answers = parse_gold_answers(gold)

    if any(open_answers_equal(prediction, g) for g in gold_answers):
        return True

    parts = [
        part.strip()
        for part in re.split(r",|;|\band\b", prediction)
        if part.strip()
    ]

    if len(parts) > 1:
        return all(
            any(open_answers_equal(part, g) for g in gold_answers)
            for part in parts
        )

    return False


# ============================================================
# 5. rule 단계 진입점
# ============================================================

def rule_parse(row):
    """row(question_type, options, raw_output)에서 rule 단계 예측을 뽑는다. 실패하면 None."""

    if row["question_type"] == "multiple-choice":
        return parse_multiple_choice(row["raw_output"], len(row["options"]))

    return extract_open_answer(row["raw_output"])


def is_correct(row, prediction):

    if prediction is None:
        return False

    if row["question_type"] == "multiple-choice":
        return prediction == str(row["answer"]).strip().upper()

    return open_is_correct(prediction, row["answer"])
