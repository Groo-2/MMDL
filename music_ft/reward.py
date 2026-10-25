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

from parsing import parse_multiple_choice  # noqa: E402

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


def score_one(completion, answer):
    predicted = parse_multiple_choice(completion_text(completion), NUM_OPTIONS)
    return 1.0 if predicted == answer else 0.0


def accuracy_reward(completions, answer, **kwargs):
    """TRL GRPOTrainer reward_funcs용. dataset의 'answer' 열이 키워드로 들어온다."""
    return [score_one(c, a) for c, a in zip(completions, answer)]
