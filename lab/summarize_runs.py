"""반복 평가(run_00 ~ run_NN)의 요약과 두 모델 비교.

  python lab/summarize_runs.py summary results/baseline
  python lab/summarize_runs.py compare results/baseline results/grpo_run1

summary: run별 정확도, 평균±표준편차(run 간), Music, 잘림, 과목별 평균을 출력하고 summary_runs.json으로 저장한다.
compare: 문항별 평균 정답률(Avg@N)의 차이를 문항 단위 paired bootstrap으로 비교한다 (전체, Music, 과목별).
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_run(run_dir):
    rows = [json.loads(line) for line in (run_dir / "final_predictions.jsonl").open(encoding="utf-8")]
    return rows


def run_dirs(root):
    return sorted(p for p in Path(root).glob("run_*") if (p / "final_predictions.jsonl").exists())


def stats(rows):
    n = len(rows)
    return {
        "n": n,
        "acc": sum(bool(r["correct"]) for r in rows) / n if n else float("nan"),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "errors": sum(r["finish_reason"] == "error" for r in rows),
    }


def summarize(root):
    root = Path(root)
    dirs = run_dirs(root)
    if not dirs:
        raise SystemExit(f"{root}에 final_predictions.jsonl이 있는 run이 없다")

    per_run, per_subject, per_question = [], {}, {}
    for d in dirs:
        rows = load_run(d)
        overall = stats(rows)
        music = stats([r for r in rows if r["subject"] == "Music"])
        per_run.append({"run": d.name, "overall": overall, "music": music})
        for r in rows:
            per_subject.setdefault(r["subject"], {}).setdefault(d.name, []).append(bool(r["correct"]))
            per_question.setdefault(r["id"], []).append(bool(r["correct"]))

    def ms(vals):
        vals = np.array(vals, dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    overall_acc = ms([p["overall"]["acc"] for p in per_run])
    music_acc = ms([p["music"]["acc"] for p in per_run if p["music"]["n"]]) if any(p["music"]["n"] for p in per_run) else None
    summary = {
        "runs": len(dirs),
        "overall_acc_mean_std": overall_acc,
        "music_acc_mean_std": music_acc,
        "truncated_mean": float(np.mean([p["overall"]["truncated"] for p in per_run])),
        "errors_total": int(sum(p["overall"]["errors"] for p in per_run)),
        "per_run": per_run,
        "per_subject_mean_acc": {s: float(np.mean([np.mean(v) for v in runs.values()])) for s, runs in sorted(per_subject.items())},
    }
    (root / "summary_runs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n=== {root.name}: {len(dirs)} runs ===")
    print(f"{'run':<10}{'전체':>9}{'Music':>9}{'잘림':>7}{'오류':>6}")
    for p in per_run:
        m = f"{p['music']['acc']:.3f}" if p["music"]["n"] else "-"
        print(f"{p['run']:<10}{p['overall']['acc']:>9.4f}{m:>9}{p['overall']['truncated']:>7}{p['overall']['errors']:>6}")
    print(f"전체 Avg@{len(dirs)}: {overall_acc[0]:.4f} ± {overall_acc[1]:.4f}")
    if music_acc:
        print(f"Music Avg@{len(dirs)}: {music_acc[0]:.4f} ± {music_acc[1]:.4f}")
    if summary["errors_total"]:
        print(f"⚠️ 요청 오류 {summary['errors_total']}건 — 같은 커맨드를 다시 실행하면 실패한 문항만 재시도합니다")
    return summary


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """문항별 평균 정답률 a, b(같은 문항 순서)의 평균 차이와 95% 신뢰구간."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def question_means(root):
    acc, subj = {}, {}
    for d in run_dirs(root):
        for r in load_run(d):
            acc.setdefault(r["id"], []).append(bool(r["correct"]))
            subj[r["id"]] = r["subject"]
    return {i: float(np.mean(v)) for i, v in acc.items()}, subj


def compare(base_root, new_root):
    base, subj = question_means(base_root)
    new, _ = question_means(new_root)
    ids = sorted(set(base) & set(new))
    if len(ids) != len(base) or len(ids) != len(new):
        print(f"⚠️ 문항 집합이 다르다: base {len(base)}, new {len(new)}, 공통 {len(ids)}")

    def row(label, sel):
        a, b = [base[i] for i in sel], [new[i] for i in sel]
        d, lo, hi = paired_bootstrap(a, b)
        flag = "  *" if lo > 0 or hi < 0 else ""
        print(f"{label:<38}{np.mean(a):>8.4f}{np.mean(b):>8.4f}{d:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{flag}")

    print(f"\n{'':<38}{'base':>8}{'new':>8}{'차이':>9}   95% 신뢰구간 (paired bootstrap, 문항 단위)")
    row("전체", ids)
    row("Music", [i for i in ids if subj[i] == "Music"])
    print("-" * 80)
    for s in sorted({subj[i] for i in ids}):
        row(s, [i for i in ids if subj[i] == s])
    print("\n* = 95% 신뢰구간이 0을 포함하지 않음. 과목당 30문항이라 과목별 차이는 노이즈가 크다.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "summary":
        summarize(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
