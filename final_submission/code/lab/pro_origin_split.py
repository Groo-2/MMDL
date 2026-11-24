"""MMMU-Pro 결과를 문항의 출처(MMMU validation에서 온 것 / MMMU test에서 온 것)로 나눠 정확도를 본다.

MMMU-Pro 1,730문항은 MMMU validation 577문항과 MMMU test 1,153문항에서 만든 것이다(문항 id의 접두 validation_ / test_).
우리는 모델을 MMMU validation 점수로 골랐기 때문에, validation에서 온 577문항의 결과는 선택에 쓰지 않은 데이터가 아니다.
선택과 무관한 결과를 보려면 test에서 온 1,153문항(그리고 MMMU test 전체)을 본다.

  python lab/pro_origin_split.py results/dpo_dvmx1/mmmu_pro_standard10
  python lab/pro_origin_split.py results/base_130/mmmu_pro_vision results/dpo_dvmx1/mmmu_pro_vision    # 여러 폴더를 나란히
"""

import json
import sys
from pathlib import Path


def origin(row_id):
    return "validation" if str(row_id).startswith("validation_") else "test"


def split_accuracy(run_dir):
    """run 폴더 하나의 final_predictions.jsonl을 출처별로 센다: {출처: (문항 수, 정답 수)}"""

    counts = {"validation": [0, 0], "test": [0, 0]}

    for line in (Path(run_dir) / "final_predictions.jsonl").read_text(encoding="utf-8").splitlines():

        if not line.strip():
            continue

        row = json.loads(line)
        c = counts[origin(row["id"])]
        c[0] += 1
        c[1] += bool(row["correct"])

    return {k: tuple(v) for k, v in counts.items()}


def summarize(root):
    """results/<태그>/<벤치마크> 아래 run_NN들을 출처별로 요약해 문자열로 돌려준다."""

    runs = sorted(p for p in Path(root).glob("run_*") if (p / "final_predictions.jsonl").exists())

    if not runs:
        return f"{root}: 끝난 run이 없다"

    lines = [f"{root} ({len(runs)}개 run)"]
    sums = {"validation": [0, 0], "test": [0, 0]}

    for run in runs:

        part = split_accuracy(run)
        cells = []

        for key in ("validation", "test"):
            n, k = part[key]
            sums[key][0] += n
            sums[key][1] += k
            cells.append(f"{key}에서 온 {n}문항 {k / n * 100:.2f}%" if n else f"{key}에서 온 0문항")

        lines.append(f"  {run.name}: " + " | ".join(cells))

    n_all = sum(v[0] for v in sums.values())
    k_all = sum(v[1] for v in sums.values())
    avg = " | ".join(f"{key} {v[1] / v[0] * 100:.2f}%" for key, v in sums.items() if v[0])
    lines.append(f"  run 평균: {avg} | 전체 {k_all / n_all * 100:.2f}%")
    lines.append("  (validation 쪽은 모델을 고르는 데 쓴 문항이라 선택 편향이 있다. test 쪽을 기준으로 본다)")

    return "\n".join(lines)


def main():

    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    for root in sys.argv[1:]:
        print(summarize(root))
        print()


if __name__ == "__main__":
    main()
