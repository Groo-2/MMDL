"""MMMU test와 MMMU-Pro 평가 데이터 점검 (GPU와 vLLM이 필요 없다. 데이터는 내려받는다).

긴 평가를 시작하기 전에 데이터가 코드가 기대하는 모양인지 확인한다.
  - 과목별 문항 수가 benchmarks.py에 적힌 값과 같은가 (test 10,500 / MMMU-Pro 1,730)
  - 정답이 비어 있지 않은가, 객관식 정답 글자가 선택지 수 안에 있는가
  - 문항 id가 겹치지 않는가
  - 과목마다 앞 몇 문항으로 실제 요청 내용(텍스트와 이미지 수)이 만들어지는가

  python lab/check_benchmarks.py                                   # 네 벤치마크 전체(내려받는 데 시간이 걸린다)
  python lab/check_benchmarks.py --benchmarks mmmu_pro_vision      # 하나만
  python lab/check_benchmarks.py --benchmarks mmmu_test --subjects Music,Math    # 일부 과목만
"""

import argparse
import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "mmmu_eval"))

import run_mmmu_eval as R  # noqa: E402
from prompting import build_content  # noqa: E402

BENCHMARKS = ["mmmu_pro_standard10", "mmmu_pro_standard4", "mmmu_pro_vision", "mmmu_test"]


def check_subject(bench, subject, data_root, preview):
    """과목 하나를 점검하고 (문항 수, 문제 목록, 선택지 수 분포)를 돌려준다."""

    problems = []
    dataset = R.load_subject(subject, data_root, bench)      # 문항 수가 다르면 여기서 RuntimeError

    ids = dataset["id"]
    answers = dataset["answer"]
    options = dataset["options"]

    if len(set(ids)) != len(ids):
        problems.append(f"{subject}: 문항 id가 겹친다")

    qtypes = dataset["question_type"] if "question_type" in dataset.column_names else ["multiple-choice"] * len(ids)
    option_counts = collections.Counter()

    for i, (answer, opts, qtype) in enumerate(zip(answers, options, qtypes)):

        if not str(answer).strip():
            problems.append(f"{ids[i]}: 정답이 비어 있다")
            continue

        n = len(R.parse_options(opts))
        option_counts[n] += 1

        if qtype == "multiple-choice" and (len(str(answer).strip()) != 1 or ord(str(answer).strip().upper()) - ord("A") >= n):
            problems.append(f"{ids[i]}: 정답 {answer!r}이 선택지 {n}개 밖이다")

    for i in range(min(preview, len(dataset))):

        sample = bench.normalize(dataset[i])
        content, parsed = build_content(sample, bench.prompt)
        n_images = sum(1 for part in content if part["type"] == "image_url")

        # 주관식(open) 문항은 선택지가 없는 것이 정상이다. 객관식인데 선택지를 못 읽은 경우만 문제로 본다.
        if not parsed and qtypes[i] == "multiple-choice":
            problems.append(f"{ids[i]}: 객관식인데 선택지를 읽지 못했다")

        if bench.kind == "pro_vision" and n_images != 1:
            problems.append(f"{ids[i]}: vision인데 이미지가 {n_images}장이다")

    return len(dataset), problems, option_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--subjects", type=lambda v: [s.strip() for s in v.split(",") if s.strip()], default=None,
                        help="쉼표로 구분한 과목만 (기본 30과목 전부)")
    parser.add_argument("--data-root", default=str(REPO / "data" / "hf_mmmu"))
    parser.add_argument("--preview", type=int, default=3, help="과목마다 요청 내용을 만들어 볼 문항 수")
    args = parser.parse_args()

    subjects = args.subjects or R.SUBJECTS
    bad = [s for s in subjects if s not in R.SUBJECTS]
    if bad:
        raise SystemExit(f"알 수 없는 과목: {bad}")

    all_problems = []

    for name in args.benchmarks:

        bench = R.BENCHMARKS[name]
        total, options_dist = 0, collections.Counter()
        print(f"\n=== {name} ({bench.repo}{' / ' + bench.config if bench.config else ''}, {bench.split}, revision {bench.revision[:8]})", flush=True)

        for subject in subjects:

            try:
                n, problems, counts = check_subject(bench, subject, args.data_root, args.preview)
            except Exception as e:  # noqa: BLE001
                all_problems.append(f"[{name}] {subject}: {type(e).__name__}: {e}")
                print(f"  {subject:<36} 실패: {type(e).__name__}: {e}", flush=True)
                continue

            total += n
            options_dist.update(counts)
            all_problems += [f"[{name}] {p}" for p in problems]
            print(f"  {subject:<36} {n:>4}문항" + (f"  문제 {len(problems)}건" if problems else "  OK"), flush=True)

        expected = sum(bench.counts[s] for s in subjects)
        print(f"  합계 {total}문항 (기대 {expected}) | 선택지 수 분포 {dict(sorted(options_dist.items()))}")

        if total != expected:
            all_problems.append(f"[{name}] 합계가 기대와 다르다: {total} != {expected}")

    print()

    if all_problems:
        print(f"문제 {len(all_problems)}건:")
        for p in all_problems[:40]:
            print("  -", p)
        raise SystemExit(1)

    print("모두 정상이다.")


if __name__ == "__main__":
    main()
