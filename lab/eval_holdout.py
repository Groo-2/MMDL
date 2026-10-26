"""SSMR hold-out(Test_visual에서 뽑은 400문항, 학습에 쓰지 않음) 평가. GRPO가 SSMR 자체를 배웠는지와 모델 선택의 1차 신호.

동결 프롬프트(mmmu_pro_cot)와 평가 sampling(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5)을 쓰고,
채점은 학습 보상과 같은 규칙 파서(judge 없음)다. SSMR 논문과 같이 기본 최대 8,192토큰.
Music 30문항(MMMU)보다 문항이 훨씬 많아서 학습 효과를 보는 데 통계적으로 더 믿을 만하다.

  python lab/eval_holdout.py --tag baseline --model Qwen/Qwen3-VL-4B-Instruct --revision ebb281e...
  python lab/eval_holdout.py --tag grpo_run1 --model results/grpo_run1/merged
  python lab/eval_holdout.py --tag t --no-server --api-base http://localhost:8000/v1 --model-name qwen3-vl-4b   # 이미 떠 있는 서버
결과: results/<tag>/holdout/seed_<seed>/predictions.jsonl, summary.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "music_ft"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssmr_client import build_content, chat, load_rows, score_one  # noqa: E402
from envinfo import snapshot  # noqa: E402

# 동결 평가 recipe (mmmu_eval/run_mmmu_eval.py와 같은 값)
TEMPERATURE, TOP_P, TOP_K, PRESENCE_PENALTY, REPETITION_PENALTY = 0.7, 0.8, 20, 1.5, 1.0


def summarize(records):
    """records: [{'category','correct','finish_reason'}] -> 요약 dict"""
    n = len(records)
    done = [r for r in records if r["finish_reason"] != "length"]
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["correct"])
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n if n else None,
        "truncated": sum(1 for r in records if r["finish_reason"] == "length"),
        "accuracy_among_finished": (sum(r["correct"] for r in done) / len(done)) if done else None,
        "finished": len(done),
        "by_category": {c: {"n": len(v), "accuracy": sum(v) / len(v)} for c, v in sorted(by_cat.items())},
    }


def run_seed(rows, data_dir, api_base, model_name, seed, max_new_tokens, concurrency, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    done = {}
    if pred_file.exists():
        for line in pred_file.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    print(f"[holdout seed={seed}] 전체 {len(rows)} | 완료 {len(done)} | 남음 {len(todo)}", flush=True)

    def one(row):
        content = build_content(row, data_dir)
        text, finish = chat(api_base, model_name, content, n=1, max_tokens=max_new_tokens, temperature=TEMPERATURE,
                            top_p=TOP_P, top_k=TOP_K, presence_penalty=PRESENCE_PENALTY,
                            repetition_penalty=REPETITION_PENALTY, seed=seed)[0]
        return {"id": row["id"], "category": row["category"], "answer": row["answer"],
                "correct": bool(score_one(text, row["answer"])), "finish_reason": finish, "response": text}

    with pred_file.open("a", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[holdout] {futures[fut]['id']} 실패: {e!r} (같은 커맨드를 다시 실행하면 실패한 문항만 재시도)", flush=True)
                continue
            done[res["id"]] = res
            fp.write(json.dumps(res, ensure_ascii=False) + "\n")
            fp.flush()
            if i % 50 == 0:
                print(f"[holdout seed={seed}] {i}/{len(todo)}", flush=True)

    records = [done[r["id"]] for r in rows if r["id"] in done]
    summary = summarize(records)
    summary.update(seed=seed, max_new_tokens=max_new_tokens, missing=len(rows) - len(records))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-dir", default=str(REPO / "data" / "ssmr"))
    parser.add_argument("--holdout-file", default="holdout.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-root", default=str(REPO / "results"))
    parser.add_argument("--no-server", action="store_true", help="이미 떠 있는 서버를 쓴다 (--api-base, --model-name)")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="qwen3-vl-4b")
    parser.add_argument("--dry-run", action="store_true", help="서버 없이 첫 문항의 프롬프트 구성만 출력")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir / args.holdout_file, args.limit)
    out_root = Path(args.out_root) / args.tag / "holdout"

    if args.dry_run:
        content = build_content(rows[0], data_dir)
        print(f"문항 {len(rows)}개 | 첫 문항 id={rows[0]['id']} 정답={rows[0]['answer']} | 이미지 {sum(1 for c in content if c['type'] == 'image_url')}장")
        print(content[0]["text"])
        return

    out_root.mkdir(parents=True, exist_ok=True)
    snapshot(out_root, extra={"model": args.model, "revision": args.revision, "tool": "eval_holdout"})

    server = None
    if not args.no_server:
        from eval_repeat import start_server, stop_server  # noqa: E402

        revision_args = ["--revision", args.revision] if args.revision else []
        server = start_server(
            ["vllm", "serve", args.model, *revision_args, "--served-model-name", args.model_name, "--dtype", "bfloat16",
             "--seed", "42", "--max-model-len", str(args.max_new_tokens + 7168),
             "--gpu-memory-utilization", "0.90", "--port", "8000"],
            8000, out_root / "vllm_holdout.log")
    try:
        summaries = [run_seed(rows, data_dir, args.api_base, args.model_name, seed, args.max_new_tokens,
                              args.concurrency, out_root / f"seed_{seed}") for seed in args.seeds]
    finally:
        if server is not None:
            stop_server(server)

    for s in summaries:
        print(f"\n[hold-out seed={s['seed']}] 정확도 {s['accuracy']:.3f} ({s['n']}문항) | 잘림 {s['truncated']} | "
              f"끝까지 답한 {s['finished']}문항 정확도 {s['accuracy_among_finished']}")
        for c, v in s["by_category"].items():
            print(f"   {c:<10} {v['accuracy']:.3f} ({v['n']}문항)")


if __name__ == "__main__":
    main()
