"""pilot20(악보 판독)을 팀 baseline(Run B) 파이프라인 그대로 평가한다.

baseline 파일(baseline/mmmu_eval/*.py, baseline/scripts/run_mmmu_eval.sh)은 한 줄도 수정하지 않고
import해서 쓴다. 프롬프트(mmmu_pro_cot), 요청 payload, rule 파서, judge는 baseline 코드 그대로다.
baseline과 다른 점은 아래 두 가지뿐이다.

  1. 데이터: baseline 로더는 MMMU 30과목 HF 데이터 전용이라 pilot20을 읽을 수 없다.
     또 baseline은 질문에 <image N> 표시가 없는 이미지를 입력에서 버리는데, pilot20 질문에는 그 표시가 없어
     그대로 두면 모델이 이미지 없이 답하게 된다. 그래서 질문 앞에 "<image 1> "을 붙여
     baseline 방식(표시를 [image]로 치환하고 이미지는 텍스트 뒤에 붙임)이 그대로 적용되게 했다.
  2. 순수 greedy (재현성용): temperature 기본 0 (baseline 0.7), presence_penalty 기본 0 (baseline 1.5).
     temperature 0만으로는 penalty가 그대로 적용돼 순수 greedy가 아니므로 두 값을 함께 바꾼다.
     baseline의 상수(TEMPERATURE, PRESENCE_PENALTY)를 고치면 MMMU baseline 재현이 깨지므로 실행할 때만 덮어쓴다.
     repetition_penalty는 baseline 값 1.0이 이미 penalty 없음이고, temperature 0에서는 top_p/top_k가 결과에
     영향을 주지 않아서 baseline 값(0.8 / 20)을 그대로 둔다. seed/max_new_tokens도 baseline 그대로다.

서브커맨드 (baseline의 infer / judge / score와 같은 구조)
  infer : 평가 모델 서버에 20문항 요청 -> predictions.jsonl (resume 지원)
  judge : rule 파서가 실패한 문항만 judge 서버로 판정 -> judge_results.jsonl (baseline cmd_judge를 그대로 호출)
  score : baseline 방식(Answer: 줄 -> judge -> 오답)으로 채점 -> summary.json / scores.csv 등
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                   # scoring.py
sys.path.insert(0, str(HERE.parent / "baseline" / "mmmu_eval"))  # baseline 모듈 (수정 없이 import)

import run_mmmu_eval as base  # noqa: E402  (baseline 원본)
import scoring  # noqa: E402  (pilot20 참고용 채점)
from parsing import is_correct, rule_parse  # noqa: E402
from PIL import Image  # noqa: E402

BASELINE_TEMPERATURE = base.TEMPERATURE  # 덮어쓰기 전의 baseline 값 (기록용)
BASELINE_PRESENCE_PENALTY = base.PRESENCE_PENALTY
PROMPT_VARIANT = "mmmu_pro_cot"          # baseline 동결 프롬프트
N_QUESTIONS = 20
DEFAULT_DATA_DIR = HERE / "data" / "pilot20"


# ============================================================
# 데이터: pilot20 -> baseline이 기대하는 sample 형태
# ============================================================

def load_pilot(data_dir):

    data_dir = Path(data_dir)
    rows = [json.loads(l) for l in open(data_dir / "pilot_test.jsonl", encoding="utf-8") if l.strip()]

    if len(rows) != N_QUESTIONS:
        raise RuntimeError(f"pilot20 should have {N_QUESTIONS} questions, got {len(rows)}")

    samples = []

    for r in rows:

        if "<image" in r["question"]:
            raise RuntimeError(f"{r['id']}: question already has an <image N> marker")

        image = Image.open(data_dir / r["image"])
        image.load()

        samples.append({
            "id": r["id"],
            "question": "<image 1> " + r["question"],   # baseline이 이미지를 넣는 조건 (모듈 docstring 참고)
            "options": "[]",
            "answer": r["answer"],
            "question_type": "open",                     # 정답이 자유 텍스트라 baseline의 주관식 경로를 쓴다
            "image_1": image,
        })

    return rows, samples


# ============================================================
# infer
# ============================================================

def cmd_infer(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.jsonl"
    sessions_file = output_dir / "sessions.json"

    existing = base.read_jsonl(predictions_file)

    # 같은 폴더에 생성 설정이 다른 응답이 섞이면 결과를 해석할 수 없으므로 막는다.
    if existing and sessions_file.exists():
        previous = {
            (s.get("temperature"), s.get("presence_penalty"))
            for s in json.loads(sessions_file.read_text()) if s.get("stage") == "infer"
        }
        if previous and previous != {(args.temperature, args.presence_penalty)}:
            raise RuntimeError(
                f"{output_dir} already has responses with (temperature, presence_penalty)={sorted(previous, key=str)}. "
                "Use a different --output-dir for different generation settings."
            )

    done = {r["id"] for r in existing if r["finish_reason"] != "error"}

    base.check_server(args.api_base)

    rows, samples = load_pilot(args.data_dir)
    categories = {r["id"]: r["category"] for r in rows}

    count = len(samples) if args.limit is None else min(args.limit, len(samples))

    base.TEMPERATURE = args.temperature     # baseline의 call_model이 실행 시점에 이 값들을 읽는다
    base.PRESENCE_PENALTY = args.presence_penalty

    call_args = SimpleNamespace(
        prompt=PROMPT_VARIANT,
        served_model_name=args.served_model_name,
        api_base=args.api_base,
        max_new_tokens=args.max_new_tokens,
        request_timeout=args.request_timeout,
    )

    tasks = [
        (categories[samples[i]["id"]], samples, i)
        for i in range(count)
        if samples[i]["id"] not in done
    ]

    print(f"[infer] prompt={PROMPT_VARIANT} temperature={args.temperature} presence_penalty={args.presence_penalty} "
          f"max_new_tokens={args.max_new_tokens} concurrency={args.concurrency} | done={len(done)} todo={len(tasks)}")

    session_start = time.time()
    started_at = base.now_iso()
    finished = 0

    with predictions_file.open("a", encoding="utf-8") as fp, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:

        futures = [pool.submit(base.infer_one, subject, dataset, index, call_args) for subject, dataset, index in tasks]

        for future in as_completed(futures):

            row = future.result()

            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()

            finished += 1

            print(
                f"[{finished}/{len(tasks)}] {row['id']} finish={row['finish_reason']} "
                f"tokens={row['usage'].get('completion_tokens')} {row['elapsed_sec']:.1f}s"
                + (f" error={row['error']}" if row["error"] else ""),
                flush=True,
            )

    base.append_session(output_dir, {
        "stage": "infer",
        "started_at": started_at,
        "wall_sec": round(time.time() - session_start, 3),
        "requests": len(tasks),
        "prompt_variant": PROMPT_VARIANT,
        "temperature": args.temperature,
        "presence_penalty": args.presence_penalty,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "served_model_name": args.served_model_name,
    })

    errors = sum(
        1 for r in base.latest_by_id(base.read_jsonl(predictions_file), ok=lambda r: r["finish_reason"] != "error").values()
        if r["finish_reason"] == "error"
    )

    if errors:
        print(f"⚠️  {errors} requests failed. Re-run the same command to retry only those (resume).")


# ============================================================
# judge: baseline의 cmd_judge를 그대로 호출 (rule 실패 문항만 판정)
# ============================================================

def cmd_judge(args):
    base.cmd_judge(args)


# ============================================================
# score: baseline cmd_score와 같은 규칙 (Answer: 줄 -> judge -> 오답)
# ============================================================

def cmd_score(args):

    output_dir = Path(args.output_dir)

    predictions = base.latest_by_id(
        base.read_jsonl(output_dir / "predictions.jsonl"),
        ok=lambda r: r["finish_reason"] != "error",
    )

    judged = base.latest_by_id(base.read_jsonl(output_dir / "judge_results.jsonl"), ok=lambda r: r.get("error") is None)

    rows, _ = load_pilot(args.data_dir)
    order = {r["id"]: i for i, r in enumerate(rows)}
    gold_rows = {r["id"]: r for r in rows}
    labels = scoring.label_sets(rows)

    final_rows = []

    for row in sorted(predictions.values(), key=lambda r: order[r["id"]]):

        rule_pred = rule_parse(row)

        if row["finish_reason"] == "error":
            method, final_pred = "request_error", None

        elif rule_pred is not None:
            method, final_pred = "answer_line", rule_pred

        elif row["id"] in judged and judged[row["id"]].get("error") is None:
            final_pred = judged[row["id"]]["judge_pred"]
            method = "judge" if final_pred is not None else "judge_failed"

        else:
            method, final_pred = "unparsed", None

        reference = scoring.score_one(gold_rows[row["id"]], row["raw_output"], labels)

        final_rows.append({
            **row,
            "rule_pred": rule_pred,
            "rule_correct": is_correct(row, rule_pred),
            "parser_method": method,
            "final_pred": final_pred,
            "correct": is_correct(row, final_pred),
            "judge": judged.get(row["id"]),
            "reference_strict": reference["strict"],
            "reference_lenient": reference["lenient"],
        })

    n = len(final_rows)
    correct = sum(r["correct"] for r in final_rows)
    correct_rule = sum(r["rule_correct"] for r in final_rows)

    if args.limit is None and n != N_QUESTIONS:
        raise RuntimeError(f"Expected {N_QUESTIONS} scored questions, got {n}. Run infer to completion first.")

    by_category = {}

    for r in final_rows:
        category = gold_rows[r["id"]]["category"]
        entry = by_category.setdefault(category, {"samples": 0, "correct": 0})
        entry["samples"] += 1
        entry["correct"] += r["correct"]

    for entry in by_category.values():
        entry["accuracy"] = entry["correct"] / entry["samples"]

    def counts(key):
        stats = {}
        for r in final_rows:
            stats[str(r[key])] = stats.get(str(r[key]), 0) + 1
        return stats

    sessions_file = output_dir / "sessions.json"
    sessions = json.loads(sessions_file.read_text()) if sessions_file.exists() else []
    infer_sessions = [s for s in sessions if s["stage"] == "infer"]

    summary = {
        "experiment": "pilot20_baseline_pipeline",
        "overall_accuracy": correct / n if n else 0.0,
        "samples": n,
        "correct": correct,
        "rule_only_accuracy": correct_rule / n if n else 0.0,
        "correct_rule_only": correct_rule,
        "parse_fallback_rate": sum(r["parser_method"] != "answer_line" for r in final_rows) / n if n else 0.0,
        "truncation_rate": sum(r["finish_reason"] == "length" for r in final_rows) / n if n else 0.0,
        "parser_method_counts": counts("parser_method"),
        "finish_reason_counts": counts("finish_reason"),
        "judge_used": bool(judged),
        "reference_scoring_py": {
            "note": "pilot20 전용 규칙 기반 채점(scoring.py). baseline 채점이 아니라 참고용이다.",
            "strict": sum(r["reference_strict"] for r in final_rows),
            "lenient": sum(r["reference_lenient"] for r in final_rows),
        },
        "by_category": by_category,
        "generation": {
            "temperature": sorted({s.get("temperature") for s in infer_sessions}, key=str),
            "presence_penalty": sorted({s.get("presence_penalty") for s in infer_sessions}, key=str),
            "deviation_from_baseline": (
                f"순수 greedy(재현성용): temperature {BASELINE_TEMPERATURE} -> 위 값, "
                f"presence_penalty {BASELINE_PRESENCE_PENALTY} -> 위 값. 나머지는 baseline 그대로."
            ),
            "top_p": base.TOP_P,
            "top_k": base.TOP_K,
            "top_p_top_k_note": "temperature 0에서는 결과에 영향이 없다 (baseline 값 유지)",
            "repetition_penalty": base.REPETITION_PENALTY,
            "seed": base.SEED,
            "max_new_tokens": sorted({s["max_new_tokens"] for s in infer_sessions}),
        },
        "prompt_variant": PROMPT_VARIANT,
        "image_processing": {"min_pixels": base.MIN_PIXELS, "max_pixels": base.MAX_PIXELS, "encoding": "JPEG quality=95"},
        "model": {
            "repo": base.MODEL_REPO,
            "revision": base.MODEL_REVISION,
            "served": sorted({s.get("model_path") for s in infer_sessions} - {None}),
            "served_revision": sorted({s.get("model_revision") for s in infer_sessions} - {None}),
            "dtype": "bfloat16",
        },
        "judge": {"repo": base.JUDGE_REPO, "revision": base.JUDGE_REVISION},
        "timing": {
            "infer_wall_sec_total": round(sum(s["wall_sec"] for s in infer_sessions), 3),
            "judge_wall_sec_total": round(sum(s["wall_sec"] for s in sessions if s["stage"] == "judge"), 3),
        },
        "peak_gpu_mem_mib": base.read_peak_gpu_mem_mib(output_dir),
    }

    with (output_dir / "final_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (output_dir / "failed_parses.jsonl").open("w", encoding="utf-8") as f:
        for r in final_rows:
            if r["parser_method"] != "answer_line":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "category", "gold", "final_pred", "correct", "rule_pred", "rule_correct", "parser_method",
                  "reference_strict", "reference_lenient", "finish_reason", "completion_tokens", "raw_output"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in final_rows:
            writer.writerow({
                "id": r["id"], "category": gold_rows[r["id"]]["category"], "gold": r["answer"],
                "final_pred": r["final_pred"], "correct": r["correct"], "rule_pred": r["rule_pred"],
                "rule_correct": r["rule_correct"], "parser_method": r["parser_method"],
                "reference_strict": r["reference_strict"], "reference_lenient": r["reference_lenient"],
                "finish_reason": r["finish_reason"], "completion_tokens": r["usage"].get("completion_tokens"),
                "raw_output": r["raw_output"],
            })

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("=" * 76)
    print("FINAL RESULT (pilot20, baseline pipeline)")
    print("=" * 76)
    print(f"samples            : {n}")
    print(f"correct            : {correct}  ({summary['overall_accuracy'] * 100:.1f}%)")
    print(f"rule-only          : {correct_rule}  ({summary['rule_only_accuracy'] * 100:.1f}%)")
    print(f"parser methods     : {summary['parser_method_counts']}")
    print(f"finish reasons     : {summary['finish_reason_counts']}")
    print(f"reference (scoring.py): strict {summary['reference_scoring_py']['strict']}/{n}, "
          f"lenient {summary['reference_scoring_py']['lenient']}/{n}")
    print("by category:")
    for category, entry in by_category.items():
        print(f"  {category:<16} {entry['correct']}/{entry['samples']}")
    print(f"summary            : {output_dir / 'summary.json'}")


# ============================================================
# CLI
# ============================================================

def add_common(parser):
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (스모크 테스트용. 평가 수치로 쓸 수 없음)")


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer", help="평가 모델 추론")
    add_common(p_infer)
    p_infer.add_argument("--temperature", type=float, default=0.0, help="기본 0(greedy). baseline recipe는 0.7")
    p_infer.add_argument("--presence-penalty", type=float, default=0.0, help="기본 0(순수 greedy). baseline recipe는 1.5")
    p_infer.add_argument("--max-new-tokens", type=int, default=base.DEFAULT_MAX_NEW_TOKENS)
    p_infer.add_argument("--concurrency", type=int, default=32)
    p_infer.add_argument("--request-timeout", type=int, default=3600)
    p_infer.add_argument("--api-base", default="http://localhost:8000/v1")
    p_infer.add_argument("--served-model-name", default="qwen3-vl-4b")
    p_infer.add_argument("--model-path", default=base.MODEL_REPO, help="기록용: 서빙 중인 모델 (HF repo 또는 체크포인트 경로)")
    p_infer.add_argument("--model-revision", default=base.MODEL_REVISION, help="기록용: 서빙 중인 revision (체크포인트면 'local')")
    p_infer.set_defaults(func=cmd_infer)

    p_judge = sub.add_parser("judge", help="rule 파서가 실패한 문항만 LLM judge로 판정")
    add_common(p_judge)
    p_judge.add_argument("--judge-api-base", default="http://localhost:8001/v1")
    p_judge.add_argument("--judge-model", default="judge", help="judge 서버의 served-model-name")
    p_judge.add_argument("--judge-revision", default=base.JUDGE_REVISION, help="기록용")
    p_judge.add_argument("--concurrency", type=int, default=16)
    p_judge.set_defaults(func=cmd_judge)

    p_score = sub.add_parser("score", help="채점하고 summary.json 생성")
    add_common(p_score)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
