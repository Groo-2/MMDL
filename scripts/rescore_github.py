"""팀 GitHub baseline(qwen-mmmu-62-44, 562/900)의 응답을 혼합 파서(rule 단계)로 다시 채점한다.

GPU가 필요 없다. baseline 비교표에서 "파서 차이"와 "프롬프트/생성 설정 차이"를 분리하는 기준선으로 쓴다.

사용법:
  python scripts/rescore_github.py \
    --predictions legacy/github_predictions.jsonl \
    --data-root /path/to/hf_cache/mmmu \
    --output results/github_rescored.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mmmu_eval"))

from parsing import is_correct, rule_parse  # noqa: E402
from prompting import parse_options  # noqa: E402
from run_mmmu_eval import SUBJECTS, load_subject, read_jsonl  # noqa: E402


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    meta = {}

    for subject in SUBJECTS:

        dataset = load_subject(subject, args.data_root).remove_columns(
            [f"image_{i}" for i in range(1, 8)]
        )

        for sample in dataset:
            meta[sample["id"]] = sample

    rows = read_jsonl(Path(args.predictions))

    per_subject = Counter()
    per_subject_github = Counter()
    methods = Counter()
    changed = []

    for github_row in rows:

        sample = meta[github_row["id"]]

        row = {
            "question_type": sample["question_type"],
            "options": parse_options(sample["options"]),
            "answer": sample["answer"],
            "raw_output": github_row["raw_output"] or "",
        }

        prediction = rule_parse(row)
        correct = is_correct(row, prediction)

        methods["answer_line" if prediction is not None else "needs_judge"] += 1
        per_subject[github_row["subject"]] += correct
        per_subject_github[github_row["subject"]] += bool(github_row["correct"])

        if correct != bool(github_row["correct"]):
            changed.append({
                "id": github_row["id"],
                "question_type": sample["question_type"],
                "gold": sample["answer"],
                "github_pred": github_row["prediction"],
                "github_parser": github_row["parser_method"],
                "hybrid_pred": prediction,
                "github_correct": bool(github_row["correct"]),
                "hybrid_correct": correct,
                "finish_reason": github_row["finish_reason"],
            })

    total = len(rows)
    github_correct = sum(per_subject_github.values())
    hybrid_correct = sum(per_subject.values())

    result = {
        "samples": total,
        "github_parser": {"correct": github_correct, "accuracy": github_correct / total},
        "hybrid_rule_only": {"correct": hybrid_correct, "accuracy": hybrid_correct / total},
        "hybrid_methods": dict(methods),
        "note": "hybrid_rule_only는 judge 없이 채점한 값이다. needs_judge 문항은 모두 오답으로 계산됨.",
        "per_subject": {
            s: {"github": per_subject_github[s], "hybrid_rule_only": per_subject[s]}
            for s in SUBJECTS
        },
        "changed": changed,
    }

    print(f"GitHub parser     : {github_correct}/{total} = {github_correct / total * 100:.2f}%")
    print(f"hybrid (rule only): {hybrid_correct}/{total} = {hybrid_correct / total * 100:.2f}%")
    print(f"hybrid methods    : {dict(methods)}")
    print(f"changed verdicts  : {len(changed)}")

    for c in changed:
        print(f"  {c['id']}: gold={c['gold']} github={c['github_pred']}({c['github_parser']}) "
              f"hybrid={c['hybrid_pred']} finish={c['finish_reason']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
