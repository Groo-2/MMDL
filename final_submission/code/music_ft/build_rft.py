"""선별(select_ssmr.py --save-samples)이 저장한 응답에서 RFT(거절 샘플링 미세조정) 학습 데이터를 만든다.

문항마다 base가 푼 8개 응답 중 **맞고 끝난(finish_reason=stop) 응답을 무작위로** 최대 per-prompt개 고른다.
가장 짧은 응답을 고르지 않는 이유: GRPO(E1)에서 짧은 풀이가 보상받아 풀이 방식이 짧은 정형 틀로 굳어 MMMU가 떨어졌다.
통과율이 [k-lo, k-hi]인 문항만 쓴다 (기본 1~7. 8/8은 base가 이미 아는 문항이라 신호가 적고, 0/8은 맞힌 응답이 없다).
같은 문장을 반복하는 응답(루프의 흔적)은 맞았더라도 제외한다.

  python music_ft/build_rft.py --source dv,data/general/dv_pool,train.dedup.jsonl --out data/rft/rft_dv.jsonl
  --source LABEL,DATA_DIR,ROWS_FILE[,MAX_PROMPTS]   여러 번 줄 수 있다 (예: 음악 데이터를 섞을 때). 문항 이미지는 DATA_DIR 기준.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def looks_degenerate(text, max_repeat=4):
    """같은 (공백이 아닌) 줄이 max_repeat번 넘게 반복되면 루프의 흔적으로 본다."""
    counts = Counter(line.strip() for line in text.split("\n") if len(line.strip()) >= 12)
    return bool(counts) and max(counts.values()) > max_repeat


def select_samples(samples, k_lo=1, k_hi=7, per_prompt=2, rng=None, max_chars=12000):
    """한 문항의 샘플 목록 [{'text','correct','finish_reason'}]에서 학습할 응답을 고른다. 쓰지 않으면 빈 목록."""
    rng = rng or random.Random(0)
    k = sum(1 for s in samples if s["correct"])
    if not (k_lo <= k <= k_hi):
        return []
    pool, seen = [], set()
    for s in samples:
        t = s["text"].strip()
        if s["correct"] and s["finish_reason"] == "stop" and t and t not in seen and len(t) <= max_chars \
                and not looks_degenerate(t):
            seen.add(t)
            pool.append(t)
    rng.shuffle(pool)
    return pool[:per_prompt]


def load_by_id(path):
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def build(source, rng, k_lo, k_hi, per_prompt):
    """source = (label, data_dir, rows_file, max_prompts). (행 목록, 통계)."""
    label, data_dir, rows_file, max_prompts = source
    data_dir = Path(data_dir)
    rows = load_by_id(data_dir / rows_file)
    samples = load_by_id(data_dir / "screen_samples.jsonl")
    ids = sorted(i for i in samples if i in rows)
    rng.shuffle(ids)
    out, used_prompts, stats = [], 0, Counter()
    for i in ids:
        if max_prompts and used_prompts >= max_prompts:
            break
        chosen = select_samples(samples[i]["samples"], k_lo, k_hi, per_prompt, rng)
        stats["screened_prompts"] += 1
        if not chosen:
            continue
        used_prompts += 1
        r = rows[i]
        for text in chosen:
            out.append({"id": r["id"], "source": label, "data_dir": str(data_dir), "category": r.get("category"),
                        "question": r["question"], "options": r["options"], "question_type": r["question_type"],
                        "answer": r["answer"], "image_paths": r["image_paths"], "response": text})
    stats["prompts_used"] = used_prompts
    stats["samples"] = len(out)
    return out, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="LABEL,DATA_DIR,ROWS_FILE[,MAX_PROMPTS]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-prompt", type=int, default=2)
    parser.add_argument("--k-lo", type=int, default=1)
    parser.add_argument("--k-hi", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows, report = [], {}
    for spec in args.source:
        parts = spec.split(",")
        source = (parts[0], parts[1], parts[2], int(parts[3]) if len(parts) > 3 else None)
        rows, stats = build(source, rng, args.k_lo, args.k_hi, args.per_prompt)
        all_rows += rows
        report[source[0]] = stats | {"by_category": dict(Counter(r["category"] for r in rows)),
                                     "by_type": dict(Counter(r["question_type"] for r in rows))}
        print(source[0], report[source[0]])
    rng.shuffle(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n학습 응답 {len(all_rows)}개 -> {out}")


if __name__ == "__main__":
    main()
