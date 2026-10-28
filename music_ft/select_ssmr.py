"""SSMR Train_visual에서 GRPO 학습에 쓸 문항을 고른다.

  1) 카테고리(Beat/Chord/Interval/Scale) 균형으로 후보 풀(기본 3,000개)을 뽑는다.
     원본은 Chord가 7%뿐인데 SSMR 논문의 테스트는 4개 카테고리가 25%씩이다.
  2) base 모델이 문항마다 n=8번 풀게 해서 통과율을 잰다 (vLLM 서버, 학습 rollout과 같은 sampling, 최대 2,048토큰).
     'Answer: X' 줄이 없거나 잘리면 0점이므로 학습 보상과 같은 기준이다 (reward.score_one).
  3) 통과율이 1/8 ~ 7/8인 문항만 남긴다. 0/8과 8/8은 8개 응답의 보상이 같아 GRPO에 학습 신호가 없다.
  4) 카테고리마다 per-category개를 통과율 0.5에 가까운 순으로 뽑는다.

  python music_ft/select_ssmr.py --data-dir data/ssmr --model Qwen/Qwen3-VL-4B-Instruct --revision ebb281e... --start-server
  python music_ft/select_ssmr.py --data-dir data/ssmr --select-only      # 이미 잰 통과율로 선별만 다시
결과: <data-dir>/screen_results.jsonl (문항별 통과율, resume 가능), train_subset.jsonl, select_report.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ssmr_client import CATEGORIES, build_content, chat, load_rows, score_one  # noqa: E402

SEED = 42


# ------------------------------------------------------------
# 순수 함수 (서버 없이 테스트 가능)
# ------------------------------------------------------------

def build_pool(rows, pool_size, seed=SEED):
    """카테고리별로 균등하게 pool_size개를 뽑는다. 어떤 카테고리가 모자라면 모자란 만큼을 다른 카테고리에서 채운다."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: r["id"])
        rng.shuffle(by_cat[cat])

    cats = [c for c in CATEGORIES if c in by_cat] + sorted(c for c in by_cat if c not in CATEGORIES)
    take = {c: 0 for c in cats}
    # 한 개씩 돌아가며 배분: 각 카테고리의 남은 문항이 있을 때만
    remaining = pool_size
    while remaining > 0:
        progressed = False
        for c in cats:
            if remaining > 0 and take[c] < len(by_cat[c]):
                take[c] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    pool = [r for c in cats for r in by_cat[c][: take[c]]]
    pool.sort(key=lambda r: r["id"])
    return pool


def select_subset(screened, per_category, lo=1 / 8, hi=7 / 8, seed=SEED):
    """screened: [{'id','category','k','n'}]. 통과율이 [lo, hi]인 문항 중 0.5에 가까운 순으로 카테고리당 per_category개.

    카테고리가 모자라면 가진 만큼 넣고, 부족분은 다른 카테고리의 남은 적격 문항(같은 정렬 기준)으로 채워 총수를 맞춘다.
    반환: (선택된 id 리스트, 보고용 dict)
    """
    rng = random.Random(seed)
    eligible = defaultdict(list)
    counts_all = Counter()
    for s in screened:
        counts_all[s["category"]] += 1
        rate = s["k"] / s["n"]
        if lo - 1e-9 <= rate <= hi + 1e-9:
            eligible[s["category"]].append((abs(rate - 0.5), rng.random(), s["id"], rate))
    for cat in eligible:
        eligible[cat].sort()

    cats = [c for c in CATEGORIES if c in counts_all] + sorted(c for c in counts_all if c not in CATEGORIES)
    chosen, notes = {}, []
    for c in cats:
        chosen[c] = eligible[c][:per_category]
        if len(chosen[c]) < per_category:
            notes.append(f"{c}: 적격 {len(eligible[c])}개 < 목표 {per_category}개")

    target_total = per_category * len(cats)
    deficit = target_total - sum(len(v) for v in chosen.values())
    if deficit > 0:
        leftovers = sorted((item, c) for c in cats for item in eligible[c][len(chosen[c]):])
        for item, c in leftovers[:deficit]:
            chosen[c].append(item)
        if deficit > len(leftovers):
            notes.append(f"전체 적격 문항이 부족해 {target_total}개 중 {sum(len(v) for v in chosen.values())}개만 선택")
        else:
            notes.append(f"부족분 {min(deficit, len(leftovers))}개를 다른 카테고리의 남은 적격 문항으로 채움")

    ids = [item[2] for c in cats for item in chosen[c]]
    rate_of = {item[2]: item[3] for c in cats for item in chosen[c]}
    report = {
        "screened_per_category": dict(counts_all),
        "eligible_per_category": {c: len(eligible[c]) for c in cats},
        "chosen_per_category": {c: len(chosen[c]) for c in cats},
        "mean_pass_rate_chosen": (sum(rate_of.values()) / len(rate_of)) if rate_of else None,
        "zero_signal_rate_screened": sum(1 for s in screened if s["k"] in (0, s["n"])) / max(1, len(screened)),
        "notes": notes,
    }
    return ids, report


# ------------------------------------------------------------
# 통과율 측정 (vLLM 서버 필요)
# ------------------------------------------------------------

def screen_rows(rows, data_dir, api_base, model, n, max_tokens, concurrency, screen_file):
    done = {}
    if screen_file.exists():
        for line in screen_file.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    print(f"[screen] 후보 {len(rows)}개 | 완료 {len(done)}개 | 남음 {len(todo)}개 | n={n} max_tokens={max_tokens}", flush=True)

    def one(row):
        content = build_content(row, data_dir)
        outs = chat(api_base, model, content, n=n, max_tokens=max_tokens, seed=SEED)
        k = sum(score_one(text, row["answer"]) for text, _ in outs)
        return {"id": row["id"], "category": row["category"], "k": int(k), "n": len(outs),
                "truncated": sum(1 for _, fr in outs if fr == "length")}

    with screen_file.open("a", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[screen] {futures[fut]['id']} 실패: {e!r}", flush=True)
                continue
            done[res["id"]] = res
            fp.write(json.dumps(res, ensure_ascii=False) + "\n")
            fp.flush()
            if i % 100 == 0:
                print(f"[screen] {i}/{len(todo)}", flush=True)
    return [done[r["id"]] for r in rows if r["id"] in done]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="build_ssmr.py 출력 폴더")
    parser.add_argument("--train-file", default="train.dedup.jsonl")
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--per-category", type=int, default=280, help="140스텝 x 8문항 = 1,120 = 4 x 280")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048, help="학습 rollout의 최대 길이와 같게")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--start-server", action="store_true", help="vLLM 서버를 직접 띄우고 끝나면 종료")
    parser.add_argument("--select-only", action="store_true", help="측정은 건너뛰고 screen_results.jsonl로 선별만")
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 후보 앞 N개만")
    parser.add_argument("--max-images", type=int, default=None,
                        help="이미지가 N장을 넘는 문항은 후보에서 제외 (24GB 학습은 1: 5장 문항은 스텝 78분+메모리 한계)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir / args.train_file)
    if args.max_images:
        before = len(rows)
        rows = [r for r in rows if len(r["image_paths"]) <= args.max_images]
        print(f"[max-images {args.max_images}] {before} → {len(rows)}문항:", dict(Counter(r["category"] for r in rows)))
    pool = build_pool(rows, args.pool_size)
    if args.limit:
        pool = pool[: args.limit]
    print(f"후보 풀 {len(pool)}개:", dict(Counter(r["category"] for r in pool)))

    screen_file = data_dir / "screen_results.jsonl"
    if args.select_only:
        screened = [json.loads(l) for l in screen_file.open(encoding="utf-8")]
        pool_ids = {r["id"] for r in pool}
        screened = [s for s in screened if s["id"] in pool_ids]
    else:
        server = None
        if args.start_server:
            sys.path.insert(0, str(HERE.parent / "lab"))
            from eval_repeat import start_server, stop_server  # noqa: E402

            server = start_server(
                ["vllm", "serve", args.model, "--revision", args.revision, "--served-model-name", "qwen3-vl-4b",
                 "--dtype", "bfloat16", "--seed", "42", "--max-model-len", "9216",
                 "--gpu-memory-utilization", "0.90", "--port", "8000"],
                8000, data_dir / "vllm_screen.log")
        try:
            screened = screen_rows(pool, data_dir, args.api_base, "qwen3-vl-4b", args.n, args.max_tokens,
                                   args.concurrency, screen_file)
        finally:
            if server is not None:
                stop_server(server)

    ids, report = select_subset(screened, args.per_category)
    by_id = {r["id"]: r for r in pool}
    rate = {s["id"]: s["k"] / s["n"] for s in screened}
    rng = random.Random(SEED)
    selected = [dict(by_id[i], pass_rate=rate[i]) for i in ids]
    rng.shuffle(selected)

    out = data_dir / "train_subset.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (data_dir / "select_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n선택 {len(selected)}개 -> {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
