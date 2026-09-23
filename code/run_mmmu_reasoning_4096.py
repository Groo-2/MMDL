import argparse
import ast
import base64
import csv
import io
import json
import math
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image


# ============================================================
# 1. 기본 설정
# ============================================================

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen3-vl-4b"

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

DATASET_REPO = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

DATA_ROOT = Path.home() / "mmdl" / "data" / "MMMU"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "mmdl"
    / "results"
    / "reasoning_4096"
)


# ============================================================
# 2. MMMU 30개 과목
# ============================================================

SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography", 
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]


# ============================================================
# 3. Generation 설정
# 기존 54.89% baseline과 동일
# ============================================================

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

REPETITION_PENALTY = 1.0
PRESENCE_PENALTY = 1.5

SEED = 3407

MAX_NEW_TOKENS = 4096


# ============================================================
# 4. 이미지 처리
# 기존 baseline과 동일
# ============================================================

MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28


# ============================================================
# 5. Prompt
# ============================================================

MC_REASONING_INSTRUCTION = """
Work through the problem carefully.
Keep the reasoning concise.
At the end of your response, output exactly one final-answer line in this format:

Final Answer: X

where X is one of the valid option letters: {valid_letters}
""".strip()


OPEN_REASONING_INSTRUCTION = """
Work through the problem carefully.
Keep the reasoning concise.
At the end of your response, output exactly one final-answer line in this format:

Final Answer: <answer>
""".strip()


# ============================================================
# 6. 선택지 Parsing
# ============================================================

def parse_options(options):

    if options is None:
        return []

    if isinstance(options, list):
        return options

    if isinstance(options, tuple):
        return list(options)

    if isinstance(options, str):

        try:
            value = ast.literal_eval(options)

            if isinstance(value, (list, tuple)):
                return list(value)

        except (ValueError, SyntaxError):
            pass

    return []


# ============================================================
# 7. 이미지 -> Base64
# 기존 baseline과 동일하게 JPEG quality=95
# ============================================================

def image_to_data_url(image):

    if image is None:
        return None

    if isinstance(image, Image.Image):

        img = image.convert("RGB")

    elif isinstance(image, dict):

        if image.get("bytes") is not None:

            img = Image.open(
                io.BytesIO(
                    image["bytes"]
                )
            ).convert("RGB")

        elif image.get("path"):

            img = Image.open(
                image["path"]
            ).convert("RGB")

        else:
            return None

    else:
        return None

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# 8. Prompt 생성
# ============================================================

def build_prompt(sample):

    question = str(
        sample["question"]
    ).strip()

    question_type = (
        sample["question_type"]
    )

    options = parse_options(
        sample["options"]
    )

    # --------------------------------------------------------
    # Multiple Choice
    # --------------------------------------------------------

    if question_type == "multiple-choice":

        valid_letters = [
            chr(ord("A") + i)
            for i in range(len(options))
        ]

        lines = [
            question,
            "",
        ]

        for i, option in enumerate(options):

            letter = chr(
                ord("A") + i
            )

            lines.append(
                f"{letter}. {option}"
            )

        lines += [
            "",
            MC_REASONING_INSTRUCTION.format(
                valid_letters=", ".join(
                    valid_letters
                )
            ),
        ]

        return (
            "\n".join(lines),
            options,
        )

    # --------------------------------------------------------
    # Open-ended
    # --------------------------------------------------------

    prompt = (
        question
        + "\n\n"
        + OPEN_REASONING_INSTRUCTION
    )

    return (
        prompt,
        options,
    )


# ============================================================
# 9. <image N> placeholder 처리
# ============================================================

IMAGE_PATTERN = re.compile(
    r"<image\s*(\d+)>",
    re.IGNORECASE,
)


def build_content(
    sample,
    prompt,
):

    content = []

    last_pos = 0
    placeholder_found = False

    for match in IMAGE_PATTERN.finditer(
        prompt
    ):

        placeholder_found = True

        text_before = prompt[
            last_pos:match.start()
        ]

        if text_before:

            content.append({
                "type": "text",
                "text": text_before,
            })

        image_number = int(
            match.group(1)
        )

        image = sample.get(
            f"image_{image_number}"
        )

        if image is not None:

            data_url = image_to_data_url(
                image
            )

            if data_url:

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    },
                })

        last_pos = match.end()

    remaining_text = prompt[
        last_pos:
    ]

    if remaining_text:

        content.append({
            "type": "text",
            "text": remaining_text,
        })

    # placeholder가 없지만 image가 있는 경우
    if not placeholder_found:

        images = []

        for i in range(1, 8):

            image = sample.get(
                f"image_{i}"
            )

            if image is None:
                continue

            data_url = image_to_data_url(
                image
            )

            if data_url:

                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    },
                })

        if images:

            content = (
                images
                + content
            )

    return content


# ============================================================
# 10. vLLM API
# ============================================================

SESSION = requests.Session()


def check_server():

    try:

        response = SESSION.get(
            "http://localhost:8000/v1/models",
            timeout=10,
        )

        response.raise_for_status()

    except Exception as e:

        raise RuntimeError(
            "vLLM server is not reachable at "
            "http://localhost:8000. "
            "Start the vLLM server first."
        ) from e


def call_model(content):

    payload = {

        "model":
            MODEL_NAME,

        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        "top_k":
            TOP_K,

        "repetition_penalty":
            REPETITION_PENALTY,

        "presence_penalty":
            PRESENCE_PENALTY,

        "seed":
            SEED,

        "max_tokens":
            MAX_NEW_TOKENS,

        "mm_processor_kwargs": {

            "min_pixels":
                MIN_PIXELS,

            "max_pixels":
                MAX_PIXELS,
        },
    }

    response = SESSION.post(
        API_URL,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()

    result = response.json()

    choice = result["choices"][0]

    raw_output = (
        choice["message"]["content"]
    )

    finish_reason = choice.get(
        "finish_reason"
    )

    usage = result.get(
        "usage",
        {},
    )

    return (
        raw_output,
        finish_reason,
        usage,
    )


# ============================================================
# 11. 공통 문자열 정리
# ============================================================

def clean_answer_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    # Markdown
    text = text.replace("**", "")
    text = text.replace("`", "")

    # 흔한 Unicode 기호
    text = text.replace("−", "-")
    text = text.replace("–", "-")

    # \boxed{...}
    boxed = re.fullmatch(
        r"\\boxed\{(.+)\}",
        text,
        flags=re.DOTALL,
    )

    if boxed:

        text = boxed.group(1).strip()

    # 양 끝의 수식 $
    if (
        len(text) >= 2
        and text.startswith("$")
        and text.endswith("$")
    ):

        text = text[1:-1].strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# 12. Multiple-choice deterministic parser
# ============================================================

def parse_multiple_choice(
    text,
    num_options,
):

    choices = [
        chr(ord("A") + i)
        for i in range(num_options)
    ]

    raw = str(text).strip()

    if not raw:
        return None, "none"

    choice_group = (
        "[" + "".join(choices) + "]"
    )

    # --------------------------------------------------------
    # 1순위: Final Answer: X
    # --------------------------------------------------------

    patterns = [

        rf"(?im)^\s*final\s+answer\s*[:：=\-]\s*"
        rf"\(?({choice_group})\)?"
        rf"(?:\s*[\.\)])?\s*$",

        rf"(?i)\bfinal\s+answer\s*[:：=\-]\s*"
        rf"\(?({choice_group})\)?",

    ]

    matches = []

    for pattern in patterns:

        matches.extend(
            re.findall(
                pattern,
                raw,
            )
        )

    valid_matches = [
        x.upper()
        for x in matches
        if x.upper() in choices
    ]

    if valid_matches:

        return (
            valid_matches[-1],
            "final_answer",
        )

    # --------------------------------------------------------
    # 2순위: Answer: X / answer is X
    # --------------------------------------------------------

    matches = re.findall(

        rf"(?i)\banswer\s*"
        rf"(?:is|[:：=\-])\s*"
        rf"\(?({choice_group})\)?",

        raw,
    )

    valid_matches = [
        x.upper()
        for x in matches
        if x.upper() in choices
    ]

    if valid_matches:

        return (
            valid_matches[-1],
            "answer_pattern",
        )

    # --------------------------------------------------------
    # 3순위: 마지막 비어있지 않은 줄이 문자 하나
    # --------------------------------------------------------

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    for line in reversed(lines):

        match = re.fullmatch(
            rf"\(?({choice_group})\)?[\.\)]?",
            line,
            flags=re.IGNORECASE,
        )

        if match:

            return (
                match.group(1).upper(),
                "last_line",
            )

    # --------------------------------------------------------
    # 4순위: 전체 응답이 문자 하나
    # --------------------------------------------------------

    if raw.upper() in choices:

        return (
            raw.upper(),
            "exact_letter",
        )

    # --------------------------------------------------------
    # 5순위:
    # 응답 마지막 300자 내 마지막 standalone option
    # --------------------------------------------------------

    tail = raw[-300:]

    candidates = []

    pattern = (
        rf"(?<![A-Za-z])"
        rf"({choice_group})"
        rf"(?![A-Za-z])"
    )

    for match in re.finditer(
        pattern,
        tail,
        flags=re.IGNORECASE,
    ):

        letter = (
            match.group(1)
            .upper()
        )

        if letter in choices:

            candidates.append(
                (
                    match.start(),
                    letter,
                )
            )

    if candidates:

        candidates.sort(
            key=lambda x: x[0]
        )

        return (
            candidates[-1][1],
            "tail_fallback",
        )

    return None, "none"


# ============================================================
# 13. Open gold alias 처리
# ============================================================

def parse_gold_answers(gold):

    if isinstance(gold, list):

        return gold

    if isinstance(gold, tuple):

        return list(gold)

    if isinstance(gold, str):

        stripped = gold.strip()

        try:

            parsed = ast.literal_eval(
                stripped
            )

            if isinstance(
                parsed,
                (list, tuple),
            ):

                return list(parsed)

        except (
            ValueError,
            SyntaxError,
        ):
            pass

        return [stripped]

    return [gold]


# ============================================================
# 14. Open answer: Final Answer 추출
# ============================================================

def extract_open_final_answer(
    text,
):

    raw = str(text).strip()

    if not raw:

        return None, "none"

    # --------------------------------------------------------
    # 1순위: Final Answer: ...
    # 여러 개 있으면 마지막 것 사용
    # --------------------------------------------------------

    matches = re.findall(

        r"(?im)^\s*final\s+answer\s*"
        r"[:：=\-]\s*(.+?)\s*$",

        raw,
    )

    if matches:

        answer = clean_answer_text(
            matches[-1]
        )

        return (
            answer,
            "final_answer",
        )

    # --------------------------------------------------------
    # 2순위: Answer: ...
    # --------------------------------------------------------

    matches = re.findall(

        r"(?im)^\s*answer\s*"
        r"[:：=\-]\s*(.+?)\s*$",

        raw,
    )

    if matches:

        answer = clean_answer_text(
            matches[-1]
        )

        return (
            answer,
            "answer_line",
        )

    # --------------------------------------------------------
    # 3순위: 마지막 비어있지 않은 line
    # --------------------------------------------------------

    lines = [
        clean_answer_text(x)
        for x in raw.splitlines()
        if x.strip()
    ]

    if lines:

        return (
            lines[-1],
            "last_line",
        )

    return None, "none"


# ============================================================
# 15. 숫자 정규화
# ============================================================

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


FRACTION_PATTERN = re.compile(
    r"^[-+]?\d+\s*/\s*[-+]?\d+$"
)


def parse_numeric_value(text):

    if text is None:
        return None

    value = clean_answer_text(
        text
    )

    value = value.strip()

    # --------------------------------------------------------
    # fraction: 24/7
    # --------------------------------------------------------

    if FRACTION_PATTERN.fullmatch(
        value
    ):

        numerator, denominator = (
            value.split("/")
        )

        try:

            numerator = float(
                numerator.strip()
            )

            denominator = float(
                denominator.strip()
            )

            if denominator != 0:

                return round(
                    numerator / denominator,
                    2,
                )

        except ValueError:
            pass

    # --------------------------------------------------------
    # 숫자 한 개만 포함된 answer
    # 예: "$1,000", "10.2 V", "-150 V"
    # --------------------------------------------------------

    numbers = NUMBER_PATTERN.findall(
        value
    )

    if len(numbers) != 1:
        return None

    try:

        number = float(
            numbers[0]
            .replace(",", "")
        )

    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return round(
        number,
        2,
    )


# ============================================================
# 16. 문자열 canonicalization
# ============================================================

def canonicalize_text(text):

    value = clean_answer_text(
        text
    )

    value = value.lower()

    # 다시 한번 양 끝 $
    value = value.strip("$")

    # 마지막 punctuation 정리
    value = value.rstrip(
        " .,:;!"
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# 17. Open answer equivalence
# ============================================================

def open_answers_equal(
    prediction,
    gold,
):

    pred_num = parse_numeric_value(
        prediction
    )

    gold_num = parse_numeric_value(
        gold
    )

    # 둘 다 numeric이면 numeric 비교
    if (
        pred_num is not None
        and gold_num is not None
    ):

        return (
            pred_num
            == gold_num
        )

    # 그렇지 않으면 canonical exact match
    pred_text = canonicalize_text(
        prediction
    )

    gold_text = canonicalize_text(
        gold
    )

    if not pred_text or not gold_text:
        return False

    return (
        pred_text
        == gold_text
    )


# ============================================================
# 18. Open 평가
# ============================================================

def evaluate_open(
    gold,
    raw_output,
):

    prediction, parser_method = (
        extract_open_final_answer(
            raw_output
        )
    )

    if prediction is None:

        return (
            None,
            False,
            parser_method,
        )

    gold_answers = (
        parse_gold_answers(
            gold
        )
    )

    for gold_answer in gold_answers:

        if open_answers_equal(
            prediction,
            gold_answer,
        ):

            return (
                prediction,
                True,
                parser_method,
            )

    return (
        prediction,
        False,
        parser_method,
    )


# ============================================================
# 19. Answer 평가
# ============================================================

def evaluate_answer(
    sample,
    raw_output,
    options,
):

    gold = sample["answer"]

    question_type = (
        sample["question_type"]
    )

    # --------------------------------------------------------
    # MC
    # --------------------------------------------------------

    if question_type == "multiple-choice":

        prediction, parser_method = (
            parse_multiple_choice(
                raw_output,
                len(options),
            )
        )

        gold_mc = str(
            gold
        ).strip().upper()

        correct = (
            prediction
            == gold_mc
        )

        return (
            prediction,
            correct,
            parser_method,
        )

    # --------------------------------------------------------
    # Open
    # --------------------------------------------------------

    return evaluate_open(
        gold,
        raw_output,
    )


# ============================================================
# 20. Validation parquet 찾기
# ============================================================

def find_validation_file(
    subject,
):

    subject_dir = (
        DATA_ROOT
        / subject
    )

    files = list(
        subject_dir.glob(
            "validation*.parquet"
        )
    )

    if len(files) != 1:

        raise RuntimeError(

            f"{subject}: "
            f"validation parquet expected 1, "
            f"got {files}"
        )

    return files[0]


# ============================================================
# 21. Subject 로딩
# ============================================================

def load_subject(
    subject,
):

    parquet_file = (
        find_validation_file(
            subject
        )
    )

    dataset = load_dataset(

        "parquet",

        data_files={
            "validation":
                str(parquet_file)
        },

        split="validation",
    )

    if len(dataset) != 30:

        raise RuntimeError(

            f"{subject}: "
            f"expected 30 samples, "
            f"got {len(dataset)}"
        )

    return dataset


# ============================================================
# 22. 한 과목 실행
# ============================================================

def run_subject(
    subject,
    predictions_fp,
    limit=None,
):

    dataset = load_subject(
        subject
    )

    if limit is not None:

        dataset = dataset.select(

            range(
                min(
                    limit,
                    len(dataset),
                )
            )
        )

    total = len(dataset)

    correct_count = 0

    subject_start = (
        time.time()
    )

    for idx, sample in enumerate(
        dataset
    ):

        print()
        print("=" * 76)

        print(
            f"[{subject}] "
            f"[{idx + 1}/{total}] "
            f"{sample['id']}"
        )

        prompt, options = (
            build_prompt(
                sample
            )
        )

        content = (
            build_content(
                sample,
                prompt,
            )
        )

        start = time.time()

        try:

            (
                raw_output,
                finish_reason,
                usage,
            ) = call_model(
                content
            )

            (
                prediction,
                correct,
                parser_method,
            ) = evaluate_answer(
                sample,
                raw_output,
                options,
            )

            error = None

        except Exception as e:

            raw_output = ""

            finish_reason = "error"

            usage = {}

            prediction = None

            correct = False

            parser_method = "error"

            error = repr(e)

        elapsed = (
            time.time()
            - start
        )

        if correct:
            correct_count += 1

        result = {

            "id":
                sample["id"],

            "subject":
                subject,

            "question_type":
                sample["question_type"],

            "gold":
                sample["answer"],

            "prediction":
                prediction,

            "correct":
                correct,

            "parser_method":
                parser_method,

            "finish_reason":
                finish_reason,

            "raw_output":
                raw_output,

            "elapsed_sec":
                round(
                    elapsed,
                    3,
                ),

            "usage":
                usage,

            "error":
                error,
        }

        predictions_fp.write(

            json.dumps(
                result,
                ensure_ascii=False,
            )

            + "\n"
        )

        predictions_fp.flush()

        print(
            "type   :",
            sample["question_type"],
        )

        print(
            "gold   :",
            sample["answer"],
        )

        print(
            "pred   :",
            prediction,
        )

        print(
            "parser :",
            parser_method,
        )

        print(
            "correct:",
            correct,
        )

        print(
            "finish :",
            finish_reason,
        )

        print(
            "time   :",
            f"{elapsed:.2f} sec",
        )

        # raw 전체 출력이 너무 길어질 수 있으므로 앞부분만 표시
        preview = (
            raw_output[:500]
            .replace("\n", " ")
        )

        print(
            "raw    :",
            preview,
        )

        if error:

            print(
                "error  :",
                error,
            )

    subject_elapsed = (
        time.time()
        - subject_start
    )

    accuracy = (

        correct_count
        / total

        if total

        else 0.0
    )

    return {

        "subject":
            subject,

        "samples":
            total,

        "correct":
            correct_count,

        "accuracy":
            accuracy,

        "elapsed_sec":
            round(
                subject_elapsed,
                3,
            ),
    }


# ============================================================
# 23. Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--subject",
        choices=SUBJECTS,
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(

        "--limit",

        type=int,

        default=None,
    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # vLLM server 확인
    # --------------------------------------------------------

    check_server()

    if args.all:

        subjects = SUBJECTS

    else:

        subjects = [
            args.subject
        ]

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        output_dir
        / "predictions.jsonl"
    )

    scores_file = (
        output_dir
        / "subject_scores.csv"
    )

    summary_file = (
        output_dir
        / "summary.json"
    )

    all_start = time.time()

    subject_results = []

    # ========================================================
    # 평가
    # ========================================================

    with predictions_file.open(

        "w",

        encoding="utf-8",

    ) as predictions_fp:

        for subject in subjects:

            print()
            print("#" * 76)

            print(
                "SUBJECT:",
                subject,
            )

            print("#" * 76)

            result = run_subject(

                subject=
                    subject,

                predictions_fp=
                    predictions_fp,

                limit=
                    args.limit,
            )

            subject_results.append(
                result
            )

            print()

            print(

                f"{subject}: "
                f"{result['correct']}/"
                f"{result['samples']} "
                f"= "
                f"{result['accuracy'] * 100:.2f}%"
            )

    total_elapsed = (
        time.time()
        - all_start
    )

    # ========================================================
    # Subject CSV
    # ========================================================

    with scores_file.open(

        "w",

        encoding="utf-8",

        newline="",

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "subject",
                "samples",
                "correct",
                "accuracy",
                "elapsed_sec",
            ],
        )

        writer.writeheader()

        writer.writerows(
            subject_results
        )

    # ========================================================
    # Overall
    # ========================================================

    macro_accuracy = (

        sum(
            row["accuracy"]
            for row
            in subject_results
        )

        / len(subject_results)

        if subject_results

        else 0.0
    )

    total_samples = sum(
        row["samples"]
        for row
        in subject_results
    )

    total_correct = sum(
        row["correct"]
        for row
        in subject_results
    )

    micro_accuracy = (

        total_correct
        / total_samples

        if total_samples

        else 0.0
    )

    # ========================================================
    # 전체 validation 검증
    # ========================================================

    if (
        args.all
        and args.limit is None
    ):

        if len(subject_results) != 30:

            raise RuntimeError(
                "Expected exactly 30 subjects."
            )

        if total_samples != 900:

            raise RuntimeError(

                f"Expected exactly 900 samples, "
                f"got {total_samples}"
            )

        if not math.isclose(

            macro_accuracy,
            micro_accuracy,

            rel_tol=1e-12,
            abs_tol=1e-12,

        ):

            raise RuntimeError(
                "Macro and micro accuracy differ."
            )

    # ========================================================
    # Parser 통계
    # ========================================================

    parser_stats = {}

    finish_stats = {}

    with predictions_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            row = json.loads(
                line
            )

            parser_name = row.get(
                "parser_method",
                "unknown",
            )

            parser_stats[
                parser_name
            ] = (
                parser_stats.get(
                    parser_name,
                    0,
                )
                + 1
            )

            finish = row.get(
                "finish_reason",
                "unknown",
            )

            finish_stats[
                finish
            ] = (
                finish_stats.get(
                    finish,
                    0,
                )
                + 1
            )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {

        "experiment":
            "reasoning_final_answer_v2",

        "model": {

            "repo":
                MODEL_REPO,

            "served_name":
                MODEL_NAME,

            "revision":
                MODEL_REVISION,

            "dtype":
                "bfloat16",
        },

        "dataset": {

            "repo":
                DATASET_REPO,

            "revision":
                DATASET_REVISION,

            "split":
                "validation",
        },

        "subjects":
            len(subject_results),

        "samples":
            total_samples,

        "correct":
            total_correct,

        "macro_accuracy":
            macro_accuracy,

        "micro_accuracy":
            micro_accuracy,

        "elapsed_sec":
            round(
                total_elapsed,
                3,
            ),

        "generation": {

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "top_k":
                TOP_K,

            "repetition_penalty":
                REPETITION_PENALTY,

            "presence_penalty":
                PRESENCE_PENALTY,

            "seed":
                SEED,

            "max_new_tokens":
                MAX_NEW_TOKENS,
        },

        "image_processing": {

            "min_pixels":
                MIN_PIXELS,

            "max_pixels":
                MAX_PIXELS,

            "encoding":
                "JPEG quality=95",
        },

        "prompt": {

            "multiple_choice_instruction":
                MC_REASONING_INSTRUCTION,

            "open_instruction":
                OPEN_REASONING_INSTRUCTION,
        },

        "parser": {

            "name":
                "deterministic_final_answer_v2",

            "multiple_choice_priority": [
                "Final Answer",
                "Answer pattern",
                "last-line option",
                "exact letter",
                "tail fallback",
            ],

            "open_priority": [
                "Final Answer",
                "Answer line",
                "last non-empty line",
            ],

            "numeric_round_digits":
                2,

            "llm_judge":
                False,
        },

        "parser_method_counts":
            parser_stats,

        "finish_reason_counts":
            finish_stats,

        "subject_results":
            subject_results,
    }

    with summary_file.open(

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            summary,

            f,

            ensure_ascii=False,

            indent=2,
        )

    # ========================================================
    # 최종 출력
    # ========================================================

    print()
    print("=" * 76)

    print(
        "FINAL RESULT"
    )

    print("=" * 76)

    print(
        "experiment      :",
        "reasoning_final_answer_v2",
    )

    print(
        "subjects        :",
        len(subject_results),
    )

    print(
        "samples         :",
        total_samples,
    )

    print(
        "correct         :",
        total_correct,
    )

    print(
        "macro accuracy  :",
        f"{macro_accuracy * 100:.2f}%",
    )

    print(
        "micro accuracy  :",
        f"{micro_accuracy * 100:.2f}%",
    )

    print(
        "elapsed         :",
        f"{total_elapsed:.2f} sec",
    )

    print(
        "parser methods  :",
        parser_stats,
    )

    print(
        "finish reasons  :",
        finish_stats,
    )

    print(
        "predictions     :",
        predictions_file,
    )

    print(
        "scores          :",
        scores_file,
    )

    print(
        "summary         :",
        summary_file,
    )


if __name__ == "__main__":
    main()
