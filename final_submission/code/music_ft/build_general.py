"""DeepVision-103K(math-77k.parquet)를 우리 선별/학습 형식으로 변환한다 (일반 데이터 RFT/GRPO용).

DeepVision의 `test/` 폴더(MMMU, MMMU-Pro 평가 사본 포함)는 평가용이라 쓰지 않는다. 이 스크립트는 math-77k.parquet만 읽는다.

변환 규칙 (정답을 정확히 채점할 수 있는 문항만 남긴다. 채점이 틀리면 RFT에 오답 응답이 섞인다)
  - 영어 문항만 (CJK 문자가 있으면 제외), 이미지는 정확히 1장
  - 질문 끝에 'A. ..', 'B. ..' 줄이 이어지고 정답이 그 보기 글자이면 객관식(multiple-choice)으로 분리
  - 정답이 글자(A~J)인데 질문에 보기 글자 줄이 없으면 풀 수 없는 문항이라 제외
  - 객관식이 아니고 정답이 숫자이면 주관식(open). 수식, 문장 정답은 우리 판정기로 정확히 채점하기 어려워 제외
  - MiMo 통과율(mimo-pass_rate, 'k/n')이 [lo, hi] 안인 것만 후보로 둔다 (우리 base의 통과율 선별 비용을 줄이는 1차 거름)
출력 <out-dir>/train.jsonl (id, question('<image 1> '로 시작), options, question_type, answer, image_paths, category, source),
      <out-dir>/images/, <out-dir>/convert_report.json

  python music_ft/build_general.py --parquet data/general/deepvision/math-77k.parquet --out-dir data/general/dv_pool --pool-size 12000
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reward  # noqa: E402,F401  (mmmu_eval 경로를 sys.path에 넣는다)
from parsing import parse_numeric_value  # noqa: E402  (평가 파서의 숫자 판정과 같은 함수)

CJK = re.compile(r"[㐀-鿿]")
OPTION_LINE = re.compile(r"^\s*\(?([A-J])[\.\):]\s*(\S.*?)\s*$")
LETTERS = "ABCDEFGHIJ"
EMPTY_MARK = "__EMPTY__"


def split_mcq(question):
    """질문 끝의 'A. ..' 줄들을 분리한다. 보기 줄이 A부터 연속된 글자로 끝까지 이어져야 한다. (stem, options) 또는 None."""
    lines = question.strip().split("\n")
    start = next((i for i, line in enumerate(lines) if (m := OPTION_LINE.match(line)) and m.group(1) == "A"), None)
    if start is None:
        return None
    options = []
    for line in lines[start:]:
        if not line.strip():
            continue
        m = OPTION_LINE.match(line)
        if not m or m.group(1) != LETTERS[len(options)]:
            return None
        options.append(m.group(2))
    stem = "\n".join(lines[:start]).strip()
    if len(options) < 2 or not stem:
        return None
    return stem, options


def parse_pass_rate(text):
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(text))
    return (int(m.group(1)), int(m.group(2))) if m else None


def convert_row(index, row, pass_lo=1, pass_hi=7):
    """(변환된 dict 또는 None, 제외 사유). 이미지는 따로 쓰므로 image_paths는 호출 쪽에서 채운다."""
    question = str(row["question"]).replace("<image>", " ").strip()
    if CJK.search(question):
        return None, "cjk"
    images = row["images"]
    if images is None or len(images) != 1:
        return None, "image_count"
    rate = parse_pass_rate(row.get("mimo-pass_rate"))
    if rate is None or not (pass_lo <= rate[0] <= min(pass_hi, rate[1] - 1)):
        return None, "pass_rate"
    reward = row["reward_model"]
    gt = str(reward["ground_truth"]).strip()
    category = "Other"
    try:
        category = json.loads(row.get("domain-anno") or "{}").get("Level_1") or "Other"
    except (ValueError, TypeError):
        pass

    parsed = split_mcq(question)
    if parsed:
        stem, options = parsed
        if gt not in LETTERS[:len(options)]:
            return None, "mc_answer_mismatch"
        out = {"question": f"<image 1> {stem}", "options": options, "question_type": "multiple-choice", "answer": gt}
    else:
        if re.fullmatch(r"[A-J]", gt):
            return None, "unlabeled_options"
        if parse_numeric_value(gt) is None:
            return None, "non_numeric_answer"
        golds = [gt] + [a for a in list(reward.get("equivalent_answers") or []) if str(a) != EMPTY_MARK and str(a) != gt]
        out = {"question": f"<image 1> {question}", "options": [], "question_type": "open",
               "answer": golds if len(golds) > 1 else gt}
    out.update({"id": f"dv_{index}", "category": category, "source": "deepvision-math", "source_index": index,
                "mimo_pass_rate": str(row.get("mimo-pass_rate"))})
    return out, "ok"


def balanced_sample(items, pool_size, seed=42):
    """카테고리별로 고르게 pool_size개를 뽑는다 (모자란 카테고리는 다른 카테고리로 채움)."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    for c in by_cat:
        by_cat[c].sort(key=lambda r: r["id"])
        rng.shuffle(by_cat[c])
    cats = sorted(by_cat)
    take = {c: 0 for c in cats}
    remaining = min(pool_size, len(items))
    while remaining > 0:
        progressed = False
        for c in cats:
            if remaining > 0 and take[c] < len(by_cat[c]):
                take[c] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return sorted((it for c in cats for it in by_cat[c][:take[c]]), key=lambda r: r["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--pass-lo", type=int, default=1)
    parser.add_argument("--pass-hi", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 앞 N행만")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(args.parquet)
    light = [c for c in pf.schema_arrow.names if c != "images"]

    # 1) 이미지 없이 문항 정보만 읽어 변환/거름 (7만 행이라 가볍다)
    table = pf.read(columns=light).to_pylist()
    if args.limit:
        table = table[:args.limit]
    reasons, converted, images_needed = Counter(), [], {}
    for i, row in enumerate(table):
        row = dict(row, images=[None])      # 이미지 개수는 2) 단계에서 확인한다
        out, why = convert_row(i, row, args.pass_lo, args.pass_hi)
        reasons[why] += 1
        if out:
            converted.append(out)
    pool = balanced_sample(converted, args.pool_size)
    print(f"변환 가능 {len(converted)}개 -> 후보 풀 {len(pool)}개 | 제외 사유 {dict(reasons)}")

    # 2) 풀에 든 행의 이미지만 파일로 저장 (이미지 1장인 행만 남긴다)
    want = {it["source_index"]: it for it in pool}
    kept, bad_images = [], 0
    offset = 0
    for batch in pf.iter_batches(batch_size=1000, columns=["images"]):
        for local, row in enumerate(batch.to_pylist()):
            idx = offset + local
            it = want.get(idx)
            if it is None:
                continue
            imgs = row["images"]
            if not imgs or len(imgs) != 1 or not imgs[0].get("bytes"):
                bad_images += 1
                continue
            suffix = Path(imgs[0].get("path") or "x.jpg").suffix.lower() or ".jpg"
            rel = f"images/{it['id']}{suffix}"
            (out_dir / rel).write_bytes(imgs[0]["bytes"])
            it["image_paths"] = {"image_1": rel}
            kept.append(it)
        offset += len(batch)
    kept.sort(key=lambda r: r["id"])
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report = {"rows_read": len(table), "converted": len(converted), "pool": len(pool), "written": len(kept),
              "image_problems": bad_images, "drop_reasons": dict(reasons),
              "by_category": dict(Counter(it["category"] for it in kept)),
              "by_type": dict(Counter(it["question_type"] for it in kept))}
    (out_dir / "convert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
