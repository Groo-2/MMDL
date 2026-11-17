"""선별(select_ssmr.py --save-samples)이 저장한 응답에서 DPO(선호 학습) 쌍을 만든다.

문항마다 (좋은 응답, 나쁜 응답) 쌍:
  좋은 응답(chosen)   : 맞고 끝난(finish_reason=stop) 응답 중 무작위 (루프 흔적 제외)
  나쁜 응답(rejected) : 끝나지 않은(잘린) 응답 또는 틀린 응답. 잘린 응답은 반복·재고 루프의 실제 사례라서 우선한다
                        (--p-truncated 확률로 잘린 응답을 먼저 고르고, 없으면 틀린 응답)
통과율이 [k-lo, k-hi]인 문항만 쓴다 (RFT와 같음: 맞은 응답과 나쁜 응답이 둘 다 있어야 쌍이 된다).

  python music_ft/build_dpo.py --source dv,data/general/dv_pool,train.dedup.jsonl --out data/dpo/dpo_dv.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_rft import load_by_id, looks_degenerate


def select_pair(samples, rng, p_truncated=0.6, k_lo=1, k_hi=7, max_chars=12000):
    """한 문항의 샘플 [{'text','correct','finish_reason'}]에서 (chosen, rejected, rejected 종류) 또는 None."""
    k = sum(1 for s in samples if s["correct"])
    if not (k_lo <= k <= k_hi):
        return None
    good = [s["text"].strip() for s in samples
            if s["correct"] and s["finish_reason"] == "stop" and s["text"].strip() and len(s["text"]) <= max_chars
            and not looks_degenerate(s["text"])]
    truncated = [s["text"] for s in samples if not s["correct"] and s["finish_reason"] == "length" and s["text"].strip()]
    wrong = [s["text"].strip() for s in samples if not s["correct"] and s["finish_reason"] == "stop" and s["text"].strip()]
    if not good or not (truncated or wrong):
        return None
    if truncated and (not wrong or rng.random() < p_truncated):
        return rng.choice(good), rng.choice(truncated), "truncated"
    return rng.choice(good), rng.choice(wrong), "wrong"


def build(source, rng, p_truncated, k_lo, k_hi, pairs_per_prompt):
    label, data_dir, rows_file, max_prompts = source
    data_dir = Path(data_dir)
    rows = load_by_id(data_dir / rows_file)
    samples = load_by_id(data_dir / "screen_samples.jsonl")
    ids = sorted(i for i in samples if i in rows)
    rng.shuffle(ids)
    out, used, stats = [], 0, Counter()
    for i in ids:
        if max_prompts and used >= max_prompts:
            break
        stats["screened_prompts"] += 1
        seen, got = set(), 0
        for _ in range(pairs_per_prompt * 4):
            if got >= pairs_per_prompt:
                break
            pair = select_pair(samples[i]["samples"], rng, p_truncated, k_lo, k_hi)
            if pair is None:
                break
            chosen, rejected, kind = pair
            if (chosen, rejected) in seen:
                continue
            seen.add((chosen, rejected))
            r = rows[i]
            out.append({"id": r["id"], "source": label, "data_dir": str(data_dir), "category": r.get("category"),
                        "question": r["question"], "options": r["options"], "question_type": r["question_type"],
                        "answer": r["answer"], "image_paths": r["image_paths"],
                        "chosen": chosen, "rejected": rejected, "rejected_kind": kind})
            got += 1
        if got:
            used += 1
    stats["prompts_used"] = used
    stats["pairs"] = len(out)
    stats["rejected_kind"] = dict(Counter(p["rejected_kind"] for p in out))
    return out, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="LABEL,DATA_DIR,ROWS_FILE[,MAX_PROMPTS]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs-per-prompt", type=int, default=1)
    parser.add_argument("--p-truncated", type=float, default=0.6)
    parser.add_argument("--k-lo", type=int, default=1)
    parser.add_argument("--k-hi", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows, report = [], {}
    for spec in args.source:
        parts = spec.split(",")
        source = (parts[0], parts[1], parts[2], int(parts[3]) if len(parts) > 3 else None)
        rows, stats = build(source, rng, args.p_truncated, args.k_lo, args.k_hi, args.pairs_per_prompt)
        all_rows += rows
        report[source[0]] = stats
        print(source[0], stats)
    rng.shuffle(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n쌍 {len(all_rows)}개 -> {out}")


if __name__ == "__main__":
    main()
