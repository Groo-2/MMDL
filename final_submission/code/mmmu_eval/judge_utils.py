"""LLM judge 단계. rule 파서(parsing.py)가 답을 뽑지 못한 문항만 처리한다.

judge 모델: meta-llama/Llama-3.1-8B-Instruct (revision pin, bf16, temperature 0, seed 42)
  - 평가 대상(Qwen)과 다른 계열을 골라 같은 계열 편향을 피한다.
  - 판정 대상 응답은 대부분 max_new_tokens에서 잘린 긴 응답이다. judge 서버를 A100과 4090 모두에서
    같은 설정(max_model_len 20480)으로 띄울 수 있게 응답의 **마지막 16,384 토큰만** 넣는다.
    프롬프트상 최종 답은 마지막 줄에 있으므로 판정에 필요한 정보는 뒤쪽에 있다.

결정성: temperature 0, seed 42로 한 번만 호출한다. 원본 Qwen/VLMEvalKit 코드는 실패하면 25번 재시도한 뒤
random.choice로 답을 골랐지만, 재현성을 위해 쓰지 않고 실패하면 오답('Z') 처리한다.
"""

import copy
import string
import threading

import requests

from parsing import clean_answer_text


JUDGE_MAX_INPUT_TOKENS = 16384
JUDGE_MAX_TOKENS = 2048
JUDGE_TEMPERATURE = 0.0
JUDGE_SEED = 42


# ============================================================
# 1. 객관식 judge 프롬프트
# 출처: QwenLM/Qwen3-VL evaluation/mmmu/eval_utils.py build_prompt()/build_option_str()
#       (VLMEvalKit의 MCQ 답 추출 프롬프트와 같은 것) — 원문 그대로
# ============================================================

def build_option_str(option_dict):

    s = 'There are several options: \n'

    for c, content in option_dict.items():
        s += f'{c}. {content}\n'

    return s


def build_mc_judge_prompt(question, options, prediction):

    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )

    return tmpl.format(question, build_option_str(options), prediction)


# ============================================================
# 2. judge 응답 해석
# 출처: QwenLM/Qwen3-VL evaluation/mmmu/eval_utils.py can_infer_option()/can_infer_text()/can_infer()
#       — 원문 그대로 (can_infer_text가 인자 dict를 소문자로 바꾸는 부작용만 copy로 막음)
# ============================================================

def can_infer_option(answer, choices):

    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]

    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'

    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch

    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'

    return False


def can_infer_text(answer, choices):

    answer = answer.lower()
    choices = dict(choices)

    assert isinstance(choices, dict)

    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()

    cands = []

    for k in choices:
        if choices[k] in answer:
            cands.append(k)

    if len(cands) == 1:
        return cands[0]

    return False


def can_infer(answer, choices):

    answer = str(answer)
    copt = can_infer_option(answer, choices)

    return copt if copt else can_infer_text(answer, choices)


# ============================================================
# 3. 주관식 judge 프롬프트 [직접 설계]
# MMMU 주관식은 공식 judge 프롬프트가 없다. 객관식 프롬프트와 같은 역할(응답에서 최종 답만 뽑기)을 하도록 만들었다.
# 채점은 judge가 아니라 뽑힌 답을 parsing.open_is_correct()로 비교해서 한다.
# ============================================================

OPEN_JUDGE_TEMPLATE = (
    "You are an AI assistant who will help me extract the final answer from a response to a question. "
    "You are provided with a question and a response. "
    "Output only the final answer that the response gives, as a short phrase or number, without any explanation. "
    "If the response does not give a final answer, output NONE.\n"
    "Question: {question}\n"
    "Response: {prediction}\n"
    "Final answer: "
)


def build_open_judge_prompt(question, prediction):
    return OPEN_JUDGE_TEMPLATE.format(question=question, prediction=prediction)


# ============================================================
# 4. judge 서버 호출 (vLLM OpenAI 호환 서버)
# ============================================================

_local = threading.local()


def _session():

    if not hasattr(_local, "session"):
        _local.session = requests.Session()

    return _local.session


def server_root(api_base):
    """'http://host:8001/v1' -> 'http://host:8001' (vLLM의 /tokenize, /detokenize는 루트 경로에 있다)."""

    return api_base[:-3] if api_base.rstrip("/").endswith("/v1") else api_base.rstrip("/")


def truncate_to_tail(text, api_base, model, max_tokens=JUDGE_MAX_INPUT_TOKENS):
    """judge 토크나이저 기준으로 text의 마지막 max_tokens 토큰만 남긴다. (잘린 text, 잘렸는지 여부)"""

    root = server_root(api_base)

    response = _session().post(
        f"{root}/tokenize",
        json={"model": model, "prompt": text, "add_special_tokens": False},
        timeout=120,
    )
    response.raise_for_status()

    tokens = response.json()["tokens"]

    if len(tokens) <= max_tokens:
        return text, False

    response = _session().post(
        f"{root}/detokenize",
        json={"model": model, "tokens": tokens[-max_tokens:]},
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["prompt"], True


def call_judge(prompt, api_base, model):

    response = _session().post(
        f"{api_base}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": JUDGE_TEMPERATURE,
            "seed": JUDGE_SEED,
            "max_tokens": JUDGE_MAX_TOKENS,
        },
        timeout=600,
    )
    response.raise_for_status()

    choice = response.json()["choices"][0]

    return choice["message"]["content"] or "", choice.get("finish_reason")


def judge_row(row, api_base, model):
    """rule 단계에서 실패한 row 하나를 판정한다. judge_pred가 None이면 오답 처리 대상."""

    prediction, truncated = truncate_to_tail(row["raw_output"] or "", api_base, model)

    if row["question_type"] == "multiple-choice":

        options = {
            chr(ord("A") + i): option
            for i, option in enumerate(row["options"])
        }

        prompt = build_mc_judge_prompt(row["question"], options, prediction)
        judge_raw, finish_reason = call_judge(prompt, api_base, model)

        inferred = can_infer(judge_raw, options)
        judge_pred = inferred if inferred and inferred != 'Z' else None

    else:

        prompt = build_open_judge_prompt(row["question"], prediction)
        judge_raw, finish_reason = call_judge(prompt, api_base, model)

        extracted = clean_answer_text(judge_raw)
        judge_pred = None if not extracted or extracted.upper().rstrip(".") == "NONE" else extracted

    return {
        "id": row["id"],
        "judge_model": model,
        "judge_pred": judge_pred,
        "judge_raw": judge_raw,
        "judge_finish_reason": finish_reason,
        "judge_input_truncated": truncated,
    }
