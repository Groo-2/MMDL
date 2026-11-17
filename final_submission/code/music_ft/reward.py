"""GRPO 보상: 평가와 같은 rule 파서로 채점한다 (정답 1, 그 외 0).

SSMR-Bench 논문은 정확도만 보상으로 쓰고 형식 보상은 넣지 않았다. 우리도 같다.
'Answer: X' 줄이 없거나(잘림 포함) 글자를 못 읽으면 0이다. judge(Llama)는 쓰지 않는다.
"""

import sys
from pathlib import Path

def _mmmu_eval_dir():
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "baseline" / "mmmu_eval", here.parent / "mmmu_eval"):
        if cand.is_dir():
            return str(cand)
    raise SystemExit("mmmu_eval 폴더를 찾지 못했다 (../baseline/mmmu_eval 또는 ../mmmu_eval)")

sys.path.insert(0, _mmmu_eval_dir())

from parsing import extract_open_answer, open_is_correct, parse_multiple_choice  # noqa: E402

NUM_OPTIONS = 4


def completion_text(completion):
    """TRL은 대화형 데이터에서 [{'role': 'assistant', 'content': ...}] 리스트를 넘긴다."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        content = last.get("content") if isinstance(last, dict) else last
        if isinstance(content, list):  # [{'type': 'text', 'text': ...}]
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""
    return ""


def score_completion(completion, answer, question_type="multiple-choice", num_options=NUM_OPTIONS):
    """객관식은 'Answer: X'의 글자, 주관식(open)은 평가와 같은 판정(숫자는 소수 둘째 자리, 그 외 정규화 문자열 일치)으로 채점한다.
    answer는 객관식이면 글자, 주관식이면 정답 문자열 또는 문자열 목록."""
    text = completion_text(completion)
    if question_type == "open":
        return 1.0 if open_is_correct(extract_open_answer(text), answer) else 0.0
    return 1.0 if parse_multiple_choice(text, int(num_options)) == answer else 0.0


def score_one(completion, answer):
    """SSMR 호환(4지선다)."""
    return score_completion(completion, answer)


def accuracy_reward(completions, answer, question_type=None, num_options=None, **kwargs):
    """TRL GRPOTrainer reward_funcs용. dataset의 'answer', 'question_type', 'num_options' 열이 키워드로 들어온다(없으면 4지선다)."""
    n = len(completions)
    qts = question_type if question_type is not None else ["multiple-choice"] * n
    nos = num_options if num_options is not None else [NUM_OPTIONS] * n
    return [score_completion(c, a, qt, no) for c, a, qt, no in zip(completions, answer, qts, nos)]
