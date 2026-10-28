"""객관식 데이터셋 형식 검증기. 팀원이 만든 MusiXQA 객관식이 우리 SSMR 변환 결과와 **같은 형식**인지 확인한다.

  python lab/validate_mcq_dataset.py --data-dir musixqa_mcq_v1                      # MusiXQA 요청서의 기준
  python lab/validate_mcq_dataset.py --data-dir data/ssmr --id-prefix "" --max-images 5 --allow-image-options --no-require-doc   # SSMR(참고)

기대하는 폴더:  <data-dir>/train.jsonl, holdout.jsonl, images/ (jsonl의 image_paths는 이 폴더 기준 상대경로)
종료 코드: ERROR가 하나라도 있으면 1. WARN은 통과하지만 확인이 필요하다.
결과는 <data-dir>/validation_report.json에도 저장한다.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCD"
REQUIRED = ["id", "question", "options", "question_type", "answer", "image_paths", "category"]


def load_jsonl(path):
    rows, errs = [], []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errs.append(f"{Path(path).name}:{i} JSON 파싱 실패: {e}")
    return rows, errs


def norm_option(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def as_number(text):
    m = re.fullmatch(r"\s*[-+]?\d+(?:[.,]\d+)?\s*(?:[a-zA-Z%]+(?: [a-zA-Z/]+)?)?\s*", str(text))
    if not m:
        return None
    try:
        return float(re.search(r"[-+]?\d+(?:[.,]\d+)?", str(text)).group(0).replace(",", ""))
    except ValueError:
        return None


def check_row(row, where, split_name, args, data_dir, errors):
    """행 하나의 형식 검사. 문제가 있으면 errors에 (where, 메시지)를 넣는다."""
    e = lambda msg: errors.append(f"{where} id={row.get('id')!r}: {msg}")  # noqa: E731
    for key in REQUIRED:
        if key not in row:
            e(f"필수 필드 없음: {key}")
    if any(k not in row for k in REQUIRED):
        return

    expected_prefix = (f"{args.id_prefix}_" if args.id_prefix else "") + split_name + "_"
    if not str(row["id"]).startswith(expected_prefix):
        e(f"id는 '{expected_prefix}'로 시작해야 한다")

    q = row["question"]
    if not isinstance(q, str) or not q.startswith("<image 1> "):
        e("question은 '<image 1> '(뒤에 공백 한 칸)로 시작하는 문자열이어야 한다")
    elif "\\n" in q:
        e("question에 글자 그대로의 '\\n'(백슬래시+n)이 들어 있다")
    if isinstance(q, str):
        if q.count("<image") != len(re.findall(r"<image \d+>", q)):
            e("question의 '<image'는 '<image N>' 형식이어야 한다")
        extra = sorted(set(re.findall(r"<image (\d+)>", q)) - {"1"})
        if args.max_images == 1 and extra:
            e(f"이미지는 1장(image_1)만 쓴다. question에 <image {extra[0]}>가 있다")

    opts = row["options"]
    if not isinstance(opts, list) or len(opts) != 4:
        e("options는 정확히 4개인 리스트여야 한다")
    else:
        for i, o in enumerate(opts):
            if not isinstance(o, str) or not o:
                e(f"options[{i}]가 빈 문자열이거나 문자열이 아니다")
            elif o != o.strip():
                e(f"options[{i}] {o!r}: 앞뒤 공백/줄바꿈이 있다")
            elif "\\n" in o or "\n" in o:
                e(f"options[{i}] {o!r}: 줄바꿈(글자 그대로의 '\\n' 포함)이 있다")
            elif "<image" in o and not (args.allow_image_options and re.fullmatch(r"<image [2-9]>", o)):
                e(f"options[{i}]: 선택지에 이미지를 쓰지 않는다 (텍스트 선택지만)")
        norm = [norm_option(o) for o in opts if isinstance(o, str)]
        if len(set(norm)) != len(norm):
            e("options에 같은(대소문자·공백만 다른 경우 포함) 선택지가 있다")

    if row["answer"] not in list(LETTERS):
        e("answer는 'A','B','C','D' 중 하나여야 한다")
    if row["question_type"] != "multiple-choice":
        e("question_type은 'multiple-choice'여야 한다")
    if not isinstance(row["category"], str) or not row["category"].strip():
        e("category가 비어 있다")

    paths = row["image_paths"]
    want = {f"image_{k}" for k in range(1, args.max_images + 1)}
    if not isinstance(paths, dict) or "image_1" not in paths or not set(paths) <= want:
        e(f"image_paths는 image_1{'' if args.max_images == 1 else '..image_%d' % args.max_images}만 가져야 한다")
    elif args.check_images:
        for key, rel in paths.items():
            p = Path(data_dir) / rel
            if not p.exists():
                e(f"{key} 파일이 없다: {rel}")
            else:
                try:
                    from PIL import Image

                    with Image.open(p) as im:
                        im.verify()
                except Exception as ex:  # noqa: BLE001
                    e(f"{key} 이미지를 열 수 없다: {rel} ({type(ex).__name__})")

    if args.require_doc:
        if not isinstance(row.get("source_doc_id"), str) or not row["source_doc_id"]:
            e("source_doc_id(원본 MusiXQA doc_id 문자열)가 필요하다")
        want_split = "train" if split_name == "train" else "test"
        if row.get("source_split") != want_split:
            e(f"source_split은 '{want_split}'이어야 한다 (MusiXQA 원본의 분할: "
              f"{'학습' if split_name == 'train' else 'hold-out'}은 원본 {want_split} 문서에서)")


def letter_check(rows, label, lo, hi, min_n, warns, errors, as_error):
    n = len(rows)
    if n < min_n:
        return
    c = Counter(r["answer"] for r in rows)
    for letter in LETTERS:
        frac = c.get(letter, 0) / n
        if frac < lo or frac > hi:
            (errors if as_error else warns).append(
                f"{label}: 정답 위치 {letter}가 {frac:.1%} (허용 {lo:.0%}~{hi:.0%}, n={n}). 정답 위치를 고르게 섞어야 한다")


def shortcut_checks(rows, warns):
    """정답만 보고 맞힐 수 있는 지름길이 있는지 (카테고리별)."""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, rs in sorted(by_cat.items()):
        rs = [r for r in rs if isinstance(r.get("options"), list) and len(r["options"]) == 4 and r.get("answer") in LETTERS]
        if len(rs) < 100:
            continue
        longest = 0
        for r in rs:
            lens = [len(str(o)) for o in r["options"]]
            correct = lens[LETTERS.index(r["answer"])]
            if correct == max(lens) and lens.count(correct) == 1:
                longest += 1
        if longest / len(rs) > 0.40:
            warns.append(f"[{cat}] 정답이 유일하게 가장 긴 선택지인 비율 {longest / len(rs):.0%} (우연이면 25%). 길이로 맞힐 수 있다")
        nums = []
        for r in rs:
            vals = [as_number(o) for o in r["options"]]
            if all(v is not None for v in vals):
                nums.append((vals, LETTERS.index(r["answer"])))
        if len(nums) >= 100:
            mid = sum(1 for vals, k in nums if sorted(vals).index(vals[k]) in (1, 2))
            if mid / len(nums) > 0.65:
                warns.append(f"[{cat}] 숫자 선택지에서 정답이 중간 두 값인 비율 {mid / len(nums):.0%} (우연이면 50%). 오답이 정답 양옆에 몰려 있다")


def validate_dataset(data_dir, args):
    data_dir = Path(data_dir)
    errors, warns, report = [], [], {}
    splits = {}
    for name in ("train", "holdout"):
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            errors.append(f"{path.name} 파일이 없다")
            continue
        rows, errs = load_jsonl(path)
        errors += errs
        splits[name] = rows
        for r in rows:
            check_row(r, path.name, name, args, data_dir, errors)

    # id 중복 (파일 안과 파일 사이)
    all_ids = Counter(r.get("id") for rows in splits.values() for r in rows)
    dup = [i for i, c in all_ids.items() if c > 1]
    if dup:
        errors.append(f"id 중복 {len(dup)}개 (예: {dup[:3]})")

    # 문항 중복
    seen = Counter()
    for rows in splits.values():
        for r in rows:
            if isinstance(r.get("options"), list) and isinstance(r.get("image_paths"), dict):
                # 같은 질문·선택지라도 이미지가 다르면 다른 문항이다 (SSMR는 템플릿 문항이 많다)
                seen[(r.get("question"), tuple(map(str, r["options"])), tuple(sorted(r["image_paths"].items())), r.get("category"))] += 1
    if any(c > 1 for c in seen.values()):
        errors.append(f"질문·선택지·이미지가 완전히 같은 문항이 {sum(1 for c in seen.values() if c > 1)}개 있다")

    # 같은 문서(악보)에 같은 질문인데 정답이 서로 다르면 라벨이 모순이다 (텍스트 선택지 기준)
    if args.require_doc:
        answers = defaultdict(set)
        for rows in splits.values():
            for r in rows:
                if isinstance(r.get("options"), list) and len(r["options"]) == 4 and r.get("answer") in LETTERS and r.get("source_doc_id"):
                    answers[(r["source_doc_id"], r.get("question"))].add(norm_option(r["options"][LETTERS.index(r["answer"])]))
        conflicts = [k for k, v in answers.items() if len(v) > 1]
        if conflicts:
            errors.append(f"같은 문서·같은 질문인데 정답이 다른 문항 {len(conflicts)}개 (예: {conflicts[:2]}). 라벨이 모순이다")

    if args.require_doc and "train" in splits and "holdout" in splits:
        train_docs = {r.get("source_doc_id") for r in splits["train"]}
        hold_docs = {r.get("source_doc_id") for r in splits["holdout"]}
        both = train_docs & hold_docs
        if both:
            errors.append(f"train과 holdout이 같은 문서를 쓴다 ({len(both)}개, 예: {sorted(both)[:3]}). hold-out은 원본 test 문서만 써야 한다")
        for name, rows in splits.items():
            per_doc = Counter(r.get("source_doc_id") for r in rows)
            over = {d: c for d, c in per_doc.items() if c > args.max_per_doc}
            if over:
                errors.append(f"{name}: 한 문서에서 {args.max_per_doc}문항을 넘게 만들었다 ({len(over)}개 문서, 예: {dict(list(over.items())[:3])}). "
                              "같은 이미지의 변형 문항이 학습/평가에 몰린다")

    for name, rows in splits.items():
        cats = Counter(r.get("category") for r in rows)
        report[name] = {"n": len(rows), "per_category": dict(cats),
                        "answer_letters": dict(Counter(r.get("answer") for r in rows))}
        if rows and len(cats) < 2:
            errors.append(f"{name}: 카테고리가 {len(cats)}개뿐이다 (2개 이상 필요)")
        floor = args.min_train_per_category if name == "train" else args.min_holdout_per_category
        for c, n in sorted(cats.items()):
            if n < floor:
                warns.append(f"{name}: 카테고리 '{c}'가 {n}개뿐이다 (권장 {floor}개 이상)")
        letter_check([r for r in rows if r.get("answer") in LETTERS], f"{name} 전체", 0.20, 0.30, 200, warns, errors, as_error=True)
        for c in cats:
            letter_check([r for r in rows if r.get("category") == c and r.get("answer") in LETTERS],
                         f"{name}/{c}", 0.15, 0.35, 100, warns, errors, as_error=False)
        shortcut_checks([r for r in rows if "options" in r and "category" in r], warns)

    report.update(errors=errors[:50], error_count=len(errors), warnings=warns[:50], warning_count=len(warns))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--id-prefix", default="musixqa", help="id는 <prefix>_train_..., <prefix>_holdout_... (빈 문자열이면 train_..., holdout_...)")
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--allow-image-options", action="store_true", help="선택지가 '<image N>'인 문항을 허용 (SSMR 비교용. MusiXQA는 쓰지 않는다)")
    parser.add_argument("--no-require-doc", dest="require_doc", action="store_false",
                        help="source_doc_id/source_split 검사를 끈다 (SSMR 같은 문서 개념이 없는 데이터용)")
    parser.add_argument("--no-check-images", dest="check_images", action="store_false")
    parser.add_argument("--max-per-doc", type=int, default=2)
    parser.add_argument("--min-train-per-category", type=int, default=200)
    parser.add_argument("--min-holdout-per-category", type=int, default=40)
    args = parser.parse_args()

    report = validate_dataset(args.data_dir, args)
    Path(args.data_dir, "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    for name in ("train", "holdout"):
        if name in report:
            print(f"[{name}] {report[name]['n']}문항 | 카테고리 {report[name]['per_category']} | 정답 위치 {report[name]['answer_letters']}")
    for w in report["warnings"]:
        print(f"⚠️  WARN  {w}")
    for er in report["errors"]:
        print(f"❌ ERROR {er}")
    print(f"\n결과: ERROR {report['error_count']}개, WARN {report['warning_count']}개"
          + (" -> ERROR를 모두 고쳐야 한다" if report["error_count"] else " -> 형식 통과" + (" (WARN은 확인 필요)" if report["warning_count"] else "")))
    sys.exit(1 if report["error_count"] else 0)


if __name__ == "__main__":
    main()
