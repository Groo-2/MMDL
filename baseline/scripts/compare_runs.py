"""두 평가 실행(Run A, Run B)을 문항 단위로 비교한다 (paired). GPU 불필요, 표준 라이브러리만 사용.

- McNemar 정확 검정: 한쪽만 맞힌 불일치 쌍으로 두 실행의 정확도 차이가 우연인지 본다
- 잘림 교차표: 32768 토큰 잘림(반복 루프)이 두 실행에서 어떻게 겹치는지
- 둘 다 끝까지 답한 문항만의 비교: 루프 영향을 뺀 추론 품질
- 과목별 차이

사용법:
  python scripts/compare_runs.py \
    --run-a results/runA_github_cot \
    --run-b results/baseline_mmmu_pro_cot \
    --output results/compare_A_vs_B.json
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def load(run_dir):

    rows = {}

    with (Path(run_dir) / "final_predictions.jsonl").open(encoding="utf-8") as f:

        for line in f:

            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row

    return rows


def mcnemar_exact(b, c):
    """불일치 쌍 b, c에 대한 이항 양측 정확 검정 p-value (H0: b와 c가 같은 확률)."""

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n

    return min(1.0, 2 * tail)


def paired_stats(pairs):
    """pairs: [(a_correct, b_correct), ...]"""

    both = sum(a and b for a, b in pairs)
    only_a = sum(a and not b for a, b in pairs)
    only_b = sum(b and not a for a, b in pairs)
    neither = sum(not a and not b for a, b in pairs)
    n = len(pairs)

    return {
        "n": n,
        "a_correct": both + only_a,
        "b_correct": both + only_b,
        "a_acc": (both + only_a) / n if n else 0.0,
        "b_acc": (both + only_b) / n if n else 0.0,
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": neither,
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }


def fmt(stats, label):

    return (
        f"{label:<28} n={stats['n']:<4} A={stats['a_correct']:<4}({stats['a_acc'] * 100:5.1f}%) "
        f"B={stats['b_correct']:<4}({stats['b_acc'] * 100:5.1f}%) | "
        f"A만 정답={stats['only_a_correct']:<3} B만 정답={stats['only_b_correct']:<3} "
        f"McNemar p={stats['mcnemar_p']:.3f}"
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    a_rows = load(args.run_a)
    b_rows = load(args.run_b)

    if set(a_rows) != set(b_rows):
        raise RuntimeError(
            f"Runs cover different items: only A={len(set(a_rows) - set(b_rows))}, "
            f"only B={len(set(b_rows) - set(a_rows))}"
        )

    ids = sorted(a_rows)

    def truncated(row):
        return row["finish_reason"] == "length"

    # ---- 전체 ----

    overall = paired_stats([(a_rows[i]["correct"], b_rows[i]["correct"]) for i in ids])

    # ---- 잘림 교차표 ----

    cells = {
        "both_completed": [i for i in ids if not truncated(a_rows[i]) and not truncated(b_rows[i])],
        "only_a_truncated": [i for i in ids if truncated(a_rows[i]) and not truncated(b_rows[i])],
        "only_b_truncated": [i for i in ids if truncated(b_rows[i]) and not truncated(a_rows[i])],
        "both_truncated": [i for i in ids if truncated(a_rows[i]) and truncated(b_rows[i])],
    }

    truncation = {
        name: paired_stats([(a_rows[i]["correct"], b_rows[i]["correct"]) for i in members])
        for name, members in cells.items()
    }

    # ---- 과목별 ----

    by_subject = defaultdict(list)

    for i in ids:
        by_subject[a_rows[i]["subject"]].append(i)

    subjects = []

    for subject, members in by_subject.items():

        stats = paired_stats([(a_rows[i]["correct"], b_rows[i]["correct"]) for i in members])

        subjects.append({
            "subject": subject,
            **stats,
            "diff_a_minus_b": stats["a_correct"] - stats["b_correct"],
            "a_truncated": sum(truncated(a_rows[i]) for i in members),
            "b_truncated": sum(truncated(b_rows[i]) for i in members),
        })

    subjects.sort(key=lambda s: (-abs(s["diff_a_minus_b"]), s["subject"]))

    # ---- 불일치 문항 ----

    discordant = [
        {
            "id": i,
            "subject": a_rows[i]["subject"],
            "gold": a_rows[i]["answer"],
            "a_pred": a_rows[i]["final_pred"],
            "b_pred": b_rows[i]["final_pred"],
            "a_correct": a_rows[i]["correct"],
            "b_correct": b_rows[i]["correct"],
            "a_finish": a_rows[i]["finish_reason"],
            "b_finish": b_rows[i]["finish_reason"],
        }
        for i in ids
        if a_rows[i]["correct"] != b_rows[i]["correct"]
    ]

    # ---- 출력 ----

    a_variant = a_rows[ids[0]].get("prompt_variant")
    b_variant = b_rows[ids[0]].get("prompt_variant")

    print(f"A = {a_variant} ({args.run_a})")
    print(f"B = {b_variant} ({args.run_b})")
    print("=" * 110)
    print(fmt(overall, "전체"))
    print(f"  차이 A−B = {overall['a_correct'] - overall['b_correct']:+d}문항 "
          f"({(overall['a_acc'] - overall['b_acc']) * 100:+.2f}%p)")
    print()
    print("잘림(finish=length) 교차표")

    for name, stats in truncation.items():
        print("  " + fmt(stats, name))

    print()
    print(f"{'과목':<38}{'A':>4}{'B':>4}{'A−B':>6}{'A만':>5}{'B만':>5}{'A잘림':>7}{'B잘림':>7}{'p':>8}")

    for s in subjects:
        print(f"{s['subject']:<38}{s['a_correct']:>4}{s['b_correct']:>4}{s['diff_a_minus_b']:>+6}"
              f"{s['only_a_correct']:>5}{s['only_b_correct']:>5}{s['a_truncated']:>7}{s['b_truncated']:>7}"
              f"{s['mcnemar_p']:>8.3f}")

    if args.output:

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "a": {"dir": args.run_a, "prompt_variant": a_variant},
            "b": {"dir": args.run_b, "prompt_variant": b_variant},
            "overall": overall,
            "truncation": truncation,
            "subjects": subjects,
            "discordant": discordant,
        }, ensure_ascii=False, indent=2))

        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
