#!/usr/bin/env python3
"""악보 홀드아웃에서 "MusiXQA 학습이 악보 읽기 능력을 늘렸는가"를 같은 문항 안에서 본다.

  python holdout_analysis.py --results <결과 폴더> --out <발표자료 또는 report 폴더>

--results 아래에 base_mx, mx_grpo_team_mx, dpo_dv1_re_mx, dpo_dvmx1_mx 폴더가 있어야 한다(각 holdout/seed_42, seed_43/predictions.jsonl).

왜 같은 문항인가: 모델이 많이 끝까지 답할수록(잘림이 줄수록) 전체 정확도가 오른다. 그래서 전체 점수만으로는 "읽는 능력이 늘었다"와
"끝내지 못하던 문항이 답을 내게 됐다"를 구분할 수 없다. base와 모델이 둘 다 끝까지 답한 문항만 골라 정확도를 비교하면 이 두 가지가 분리된다.
(잘림은 최대 8,192토큰 안에 끝나지 않은 응답이고 오답으로 채점된다.)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_all import write_table  # noqa: E402

TAGS = {"base": "base_mx", "team": "mx_grpo_team_mx", "dv1_re": "dpo_dv1_re_mx", "dvmx1": "dpo_dvmx1_mx"}
NAME = {"team": "MusiXQA GRPO(팀원)", "dv1_re": "DPO(DeepVision)", "dvmx1": "DPO(DeepVision+MusiXQA)"}
SEEDS = (42, 43)


def load(results, tag, seed):
    p = Path(results) / tag / "holdout" / f"seed_{seed}" / "predictions.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open(encoding="utf-8"))}


def finished(r):
    return r["finish_reason"] != "length"


def boot_ci(values, n=10000, seed=0):
    v = np.array(values, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return v.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = {k: {s: load(a.results, t, s) for s in SEEDS} for k, t in TAGS.items()}
    ids = sorted(D["base"][42])
    cat = {i: D["base"][42][i]["category"] for i in ids}

    rows, rows_cat = [], []
    for m in ("team", "dv1_re", "dvmx1"):
        sel = [i for i in ids if all(finished(D["base"][s][i]) and finished(D[m][s][i]) for s in SEEDS)]
        diff = [np.mean([D[m][s][i]["correct"] - D["base"][s][i]["correct"] for s in SEEDS]) for i in sel]
        mu, lo, hi = boot_ci(diff)
        acc = lambda k, ss: np.mean([D[k][s][i]["correct"] for i in ss for s in SEEDS]) * 100  # noqa: E731
        # base만 안 끝낸 문항(루프 해소): seed별로 짝지어 평균
        g = defaultdict(lambda: [0, 0])
        for s in SEEDS:
            for i in ids:
                if not finished(D["base"][s][i]) and finished(D[m][s][i]):
                    g["n"][0] += 1
                    g["n"][1] += D[m][s][i]["correct"]
        n_loop, c_loop = g["n"][0] / 2, g["n"][1] / 2
        total = sum(sum(D[m][s][i]["correct"] - D["base"][s][i]["correct"] for i in ids) for s in SEEDS) / 2
        sig = " *" if (lo > 0 or hi < 0) else ""
        rows.append([NAME[m], len(sel), f"{acc('base', sel):.1f} → {acc(m, sel):.1f}", f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]{sig}",
                     f"{n_loop:.0f}", f"{c_loop / max(n_loop, 1) * 100:.0f}", f"{total:+.0f}"])
        for c in ("Clef", "KeySignature", "MeasureCount", "PitchReading"):
            ss = [i for i in sel if cat[i] == c]
            rows_cat.append([NAME[m], c, len(ss), "-" if len(ss) < 5 else f"{acc('base', ss):.1f} → {acc(m, ss):.1f}"])
    write_table(a.out, "table10_holdout_same_questions",
                "표 10. 악보 홀드아웃에서 같은 문항 안의 비교 (base와 모델이 둘 다 끝까지 답한 문항만)",
                ["모델", "둘 다 끝낸 문항 수", "둘 다 끝낸 문항의 정확도 % (base → 모델)", "정확도 차이 %p [95% 구간]",
                 "base만 못 끝낸 문항 수", "그 문항의 모델 정확도 %", "전체 정답 수 변화(400문항)"], rows,
                "구간은 문항 단위 bootstrap(10,000회), 두 seed 모두 둘 다 끝낸 문항 기준. * = 0을 포함하지 않음. base는 이 문항들을 끝내지 못해 0점이므로 '그 문항의 모델 정확도'는 base와 비교할 수 없다.")
    write_table(a.out, "table11_holdout_same_questions_by_type",
                "표 11. 둘 다 끝낸 문항의 유형별 정확도 % (base → 모델)",
                ["모델", "유형", "문항 수", "정확도 % (base → 모델)"], rows_cat,
                "PitchReading은 base가 거의 끝내지 못해(약 76%가 잘림) 공통 문항이 5~8개뿐이라 비교할 수 없다(문항 수 5 미만이면 '-').")
    print("완료")


if __name__ == "__main__":
    main()
