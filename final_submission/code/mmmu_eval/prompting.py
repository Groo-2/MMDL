"""MMMU 프롬프트 생성.

두 가지 변형을 지원한다. baseline과 fine-tuning 재평가는 반드시 같은 변형을 써야 한다
(assignment_guidance.md §0).

- mmmu_pro_cot (기본, baseline용): MMMU-Pro 공식 CoT 프롬프트
- github_cot: 팀 GitHub(Groo-2/MMDL, qwen-mmmu-62-44 브랜치)의 원래 프롬프트. 비교 실험(Run A) 전용
"""

import ast
import base64
import io
import re

from PIL import Image


# ============================================================
# 이미지 해상도 (GitHub 원본 스크립트와 동일, Qwen 공식 build_mmmu_prompt()의 값과도 동일)
# 출처: QwenLM/Qwen3-VL evaluation/mmmu/run_mmmu.py build_mmmu_prompt()
# ============================================================

MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28


# ============================================================
# 1. mmmu_pro_cot
# ============================================================

# 출처: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml  (cot.standard) — 원문 그대로
MMMU_PRO_COT_MC_INSTRUCTION = (
    "Answer the preceding multiple choice question. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
    "Think step by step before answering."
)

# [직접 변형] MMMU-Pro는 객관식만 있어 주관식용 공식 문구가 없다. MMMU val 주관식(53문항)용으로
# 위 문장 구조를 유지한 채 "multiple choice"와 "$LETTER ... one of options"만 바꿨다.
MMMU_PRO_COT_OPEN_INSTRUCTION = (
    "Answer the preceding question. "
    "The last line of your response should be of the following format: "
    "'Answer: $ANSWER' (without quotes). "
    "Think step by step before answering."
)

# 출처: mmmu-pro/infer/infer_transformers.py replace_images_tokens()
IMAGE_TOKEN_PATTERN = re.compile(r"<image\s+(\d+)>")


def parse_options(options):
    """HF MMMU의 options 필드("['a', 'b']" 문자열)를 리스트로 바꾼다."""

    if options is None:
        return []

    if isinstance(options, (list, tuple)):
        return list(options)

    if isinstance(options, str):

        try:
            value = ast.literal_eval(options)

            if isinstance(value, (list, tuple)):
                return list(value)

        except (ValueError, SyntaxError):
            pass

    return []


def build_mmmu_pro_cot_text(sample):
    """프롬프트 텍스트와 이미지 순서를 만든다.

    출처: mmmu-pro/infer/infer_transformers.py
      - parse_options():      "A. {opt}\\nB. {opt}..."
      - construct_prompt():   f"{question}\\n{parsed_options}\\n{prompt_config['standard']}"
      - replace_images_tokens(): <image N> -> [image], 등장 순서를 image_order로 기록
    """

    question = sample["question"]
    options = parse_options(sample["options"])

    if sample["question_type"] == "multiple-choice":

        choices_str = "\n".join(
            f"{chr(ord('A') + i)}. {option}"
            for i, option in enumerate(options)
        )

        text = f"{question}\n{choices_str}\n{MMMU_PRO_COT_MC_INSTRUCTION}"

    else:

        text = f"{question}\n{MMMU_PRO_COT_OPEN_INSTRUCTION}"

    image_order = [
        int(n)
        for n in IMAGE_TOKEN_PATTERN.findall(text)
    ]

    text = IMAGE_TOKEN_PATTERN.sub("[image]", text)

    return text, image_order, options


def build_mmmu_pro_cot_content(sample):
    """OpenAI chat 형식 content. 공식 방식대로 텍스트 먼저, 이미지는 등장 순서대로 그 뒤에 붙인다.

    출처: infer_transformers.py run_inference_on_dataset()
      content = [{"type": "text", ...}] + [image for image in images]
    """

    text, image_order, options = build_mmmu_pro_cot_text(sample)

    # 텍스트에 <image N> 표시가 없는 이미지는 넣지 않는다 (MMMU-Pro 공식 origin_mmmu_doc_to_visual과 동일).
    # MMMU validation에서 이런 이미지가 있는 4문항(Agriculture_26, Materials_15, Pharmacy_4, Pharmacy_22)을
    # 확인해 보니 문제 자료가 아니라 해설 이미지(풀이 그래프, 계산식)라서 넣으면 정답이 유출된다. test는 0건.

    content = [{"type": "text", "text": text}]

    for number in image_order:

        data_url = image_to_data_url(sample.get(f"image_{number}"))

        if data_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

    return content, options


# ============================================================
# 2. github_cot (팀 GitHub 원문 그대로 — 비교 실험용)
# 출처: Groo-2/MMDL qwen-mmmu-62-44 code/run_mmmu_reasoning_4096.py
# ============================================================

GITHUB_MC_REASONING_INSTRUCTION = """
Work through the problem carefully.
Keep the reasoning concise.
At the end of your response, output exactly one final-answer line in this format:

Final Answer: X

where X is one of the valid option letters: {valid_letters}
""".strip()


GITHUB_OPEN_REASONING_INSTRUCTION = """
Work through the problem carefully.
Keep the reasoning concise.
At the end of your response, output exactly one final-answer line in this format:

Final Answer: <answer>
""".strip()


GITHUB_IMAGE_PATTERN = re.compile(
    r"<image\s*(\d+)>",
    re.IGNORECASE,
)


def build_github_prompt(sample):

    question = str(sample["question"]).strip()
    options = parse_options(sample["options"])

    if sample["question_type"] == "multiple-choice":

        valid_letters = [
            chr(ord("A") + i)
            for i in range(len(options))
        ]

        lines = [question, ""]

        for i, option in enumerate(options):
            lines.append(f"{chr(ord('A') + i)}. {option}")

        lines += [
            "",
            GITHUB_MC_REASONING_INSTRUCTION.format(
                valid_letters=", ".join(valid_letters)
            ),
        ]

        return "\n".join(lines), options

    return question + "\n\n" + GITHUB_OPEN_REASONING_INSTRUCTION, options


def build_github_content(sample):
    """<image N> 위치에 이미지를 끼워 넣는다. placeholder가 없으면 image_1~7을 앞에 붙인다."""

    prompt, options = build_github_prompt(sample)

    content = []
    last_pos = 0
    placeholder_found = False

    for match in GITHUB_IMAGE_PATTERN.finditer(prompt):

        placeholder_found = True

        text_before = prompt[last_pos:match.start()]

        if text_before:
            content.append({"type": "text", "text": text_before})

        data_url = image_to_data_url(sample.get(f"image_{int(match.group(1))}"))

        if data_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        last_pos = match.end()

    remaining_text = prompt[last_pos:]

    if remaining_text:
        content.append({"type": "text", "text": remaining_text})

    if not placeholder_found:

        images = []

        for i in range(1, 8):

            data_url = image_to_data_url(sample.get(f"image_{i}"))

            if data_url:
                images.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })

        content = images + content

    return content, options


# ============================================================
# 3. 공통
# ============================================================

PROMPT_VARIANTS = {
    "mmmu_pro_cot": build_mmmu_pro_cot_content,
    "github_cot": build_github_content,
}


def build_content(sample, variant):
    return PROMPT_VARIANTS[variant](sample)


def image_to_data_url(image):
    """PIL 이미지를 JPEG(quality=95) base64 data URL로 바꾼다 (GitHub 원본과 동일)."""

    if image is None:
        return None

    if isinstance(image, Image.Image):

        img = image.convert("RGB")

    elif isinstance(image, dict):

        if image.get("bytes") is not None:
            img = Image.open(io.BytesIO(image["bytes"])).convert("RGB")

        elif image.get("path"):
            img = Image.open(image["path"]).convert("RGB")

        else:
            return None

    else:
        return None

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
