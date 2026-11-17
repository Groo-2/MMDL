"""SSMR-Bench(Sylence/SSMR-Bench) Train_visual / Test_visual → MMMU 형식 변환.

출력 (--out-dir)
  images/<id>_q.<ext>        악보 이미지 (= image_1)
  images/<id>_o<N>.<ext>     이미지 선택지 (= image_2.. , 화면에 보이는 A, B, C, D 순서)
  train.jsonl                Train_visual 전체 (8,000)
  holdout.jsonl              Test_visual에서 카테고리 비율대로 뽑은 N개 (모델 선택용, 학습 제외)

각 줄의 필드는 mmmu_eval/prompting.py의 build_mmmu_pro_cot_text()가 요구하는 형식을 따른다.
  question       "<image 1> " + 원문 질문
  options        선택지 문자열 리스트. 이미지 선택지는 "<image 2>" ~ "<image 5>"
  question_type  "multiple-choice"
  answer         정답 글자 (A~D)
  image_paths    {"image_1": "...", "image_2": "..."}  out-dir 기준 상대경로
  category, source_split, source_index

선택지 순서는 seed 42로 섞는다(논문도 선택지를 섞어서 평가). 섞은 뒤 정답 글자가 원래 정답을
가리키는지 전수 검사한다.
"""

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


REPO = "Sylence/SSMR-Bench"
REVISION = "5f6f53d1de73d26d050ca27d10ee595f8018e3bd"  # 2025-09-07 커밋
SEED = 42
LETTERS = "ABCD"

OPTION_TEXT = ["correct_answer", "incorrect_answer1", "incorrect_answer2", "incorrect_answer3"]
OPTION_IMG = ["correct_answer_img", "incorrect_answer1_img", "incorrect_answer2_img", "incorrect_answer3_img"]


def download(split, cache_dir):
    root = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"data/{split}-*"],
        cache_dir=cache_dir,
    )
    files = sorted(Path(root, "data").glob(f"{split}-*.parquet"))
    if not files:
        raise RuntimeError(f"{split} parquet을 찾지 못했다: {root}")
    return files


def read_rows(split, cache_dir):
    rows = []
    for path in download(split, cache_dir):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def image_bytes(cell):
    if cell is None:
        return None
    data = cell.get("bytes")
    return data if data else None


def save_image(data, path_no_ext):
    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    ext = "jpg" if fmt == "jpeg" else fmt
    path = path_no_ext.with_suffix("." + ext)
    path.write_bytes(data)
    return path


def clean_option(text):
    """선택지 끝의 줄바꿈 찌꺼기를 지운다.

    원본에는 오답 3번(incorrect_answer3)에만 글자 그대로의 '\\n'(백슬래시+n)이 붙어 있다
    (예: 'Fb\\n'). 그대로 두면 모델이 '\\n이 붙은 선택지는 오답'이라는 지름길을 배우므로 지운다.
    """
    if text is None:
        return None
    text = text.strip()
    while text.endswith("\\n"):
        text = text[:-2].rstrip()
    return text


def convert(row, split, index, rng, out_dir):
    sample_id = f"{split}_{index:05d}"
    question_img = image_bytes(row["sheet_music_img"])
    if question_img is None:
        raise ValueError(f"{sample_id}: 악보 이미지가 없다")

    texts = [clean_option(row[k]) for k in OPTION_TEXT]
    imgs = [image_bytes(row[k]) for k in OPTION_IMG]

    has_text = all(t for t in texts)
    has_img = all(i is not None for i in imgs)
    # 텍스트 선택지와 이미지 선택지가 섞인 문항은 없다고 확인했다. 있으면 멈춘다.
    if has_text == has_img:
        raise ValueError(f"{sample_id}: 선택지 유형이 모호하다 (text={has_text}, img={has_img})")

    # 원래 순서에서 0번이 정답
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    answer_pos = order.index(0)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    q_path = save_image(question_img, images_dir / f"{sample_id}_q")
    image_paths["image_1"] = str(q_path.relative_to(out_dir))

    options = []
    for pos, original in enumerate(order):
        if has_text:
            options.append(texts[original])
        else:
            number = 2 + pos
            o_path = save_image(imgs[original], images_dir / f"{sample_id}_o{number}")
            image_paths[f"image_{number}"] = str(o_path.relative_to(out_dir))
            options.append(f"<image {number}>")

    question = "<image 1> " + (row["question"] or "").strip()

    record = {
        "id": sample_id,
        "question": question,
        "options": options,
        "question_type": "multiple-choice",
        "answer": LETTERS[answer_pos],
        "image_paths": image_paths,
        "category": row["category"],
        "class_name": row.get("class_name"),
        "source_split": split,
        "source_index": index,
    }

    # 검증용 원본 지문(저장하지 않고 호출자가 확인한다)
    check = (
        texts[0] if has_text else hashlib.sha1(imgs[0]).hexdigest(),
        has_text,
    )
    return record, check


def verify(record, check, out_dir):
    """섞은 뒤 정답 글자가 가리키는 선택지가 원래 정답과 같은지 확인한다."""
    gold, is_text = check
    pos = LETTERS.index(record["answer"])
    if is_text:
        assert record["options"][pos] == gold, record["id"]
    else:
        number = 2 + pos
        data = (out_dir / record["image_paths"][f"image_{number}"]).read_bytes()
        assert hashlib.sha1(data).hexdigest() == gold, record["id"]


def stratified_indices(rows, n, rng):
    by_cat = defaultdict(list)
    for i, row in enumerate(rows):
        # Test_visual에는 악보 이미지가 비어 있는 행이 있다(예: 1229번). 건너뛴다.
        if image_bytes(row["sheet_music_img"]) is None:
            continue
        by_cat[row["category"]].append(i)
    total = sum(len(v) for v in by_cat.values())
    picked = []
    cats = sorted(by_cat)
    quota = {c: round(n * len(by_cat[c]) / total) for c in cats}
    # 반올림 오차 보정
    diff = n - sum(quota.values())
    quota[max(cats, key=lambda c: len(by_cat[c]))] += diff
    for c in cats:
        picked.extend(rng.sample(by_cat[c], quota[c]))
    return sorted(picked)


def write_split(rows, split, indices, out_path, out_dir, rng):
    counts = Counter()
    with out_path.open("w", encoding="utf-8") as f:
        for i in indices:
            record, check = convert(rows[i], split, i, rng, out_dir)
            verify(record, check, out_dir)
            counts[record["category"]] += 1
            counts["ans_" + record["answer"]] += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--holdout-size", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 각 split에서 앞 N개만")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    train_rows = read_rows("Train_visual", args.cache_dir)
    test_rows = read_rows("Test_visual", args.cache_dir)
    print(f"Train_visual {len(train_rows)}, Test_visual {len(test_rows)}")

    train_idx = list(range(len(train_rows)))
    hold_idx = stratified_indices(test_rows, args.holdout_size, random.Random(SEED))
    if args.limit:
        train_idx = train_idx[: args.limit]
        hold_idx = hold_idx[: args.limit]

    c_train = write_split(train_rows, "train", train_idx, out_dir / "train.jsonl", out_dir, rng)
    c_hold = write_split(test_rows, "holdout", hold_idx, out_dir / "holdout.jsonl", out_dir, rng)

    print("train  ", dict(c_train))
    print("holdout", dict(c_hold))
    print("정답 글자 검증 통과")


if __name__ == "__main__":
    main()
