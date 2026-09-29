"""Qwen3-VL-4B-Instruct MMMU validation 평가.

팀 GitHub(Groo-2/MMDL, qwen-mmmu-62-44) code/run_mmmu_reasoning_4096.py를 바탕으로 다시 구성했다.
원본은 legacy/run_mmmu_reasoning_4096.orig.py에 보관.

원본 대비 바뀐 점 (assignment_guidance.md 기준)
  - seed 3407 → 42                                 (§1.3: Qwen 공식 recipe)
  - 로컬 parquet glob → HF config별 load_dataset(revision=...)   (§1.2)
  - 경로와 서버 주소 하드코딩 → CLI 인자                (템플릿 §1)
  - max_new_tokens 4096 → 32768                     (Qwen 공식 infer_instruct.sh 값. 4096에서 900문항 중 128문항이 잘림)
  - 순차 요청 → 동시 요청 + resume                   (32768 토큰과 이후 MMMU test/MMMU-Pro 규모 대비)
  - 프롬프트: MMMU-Pro 공식 CoT (prompting.py), 파서: 혼합 방식 (parsing.py + judge_utils.py)

서브커맨드
  infer : 평가 대상 모델 서버(vLLM)에 요청해서 predictions.jsonl 생성 (resume 지원)
  judge : rule 파서가 실패한 문항만 judge 서버에 보내서 judge_results.jsonl 생성 (resume 지원)
  score : 위 두 파일로 채점해서 final_predictions.jsonl / subject_scores.csv / summary.json 생성
"""

import argparse
import csv
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_utils import (  # noqa: E402
    JUDGE_MAX_INPUT_TOKENS,
    JUDGE_MAX_TOKENS,
    JUDGE_SEED,
    JUDGE_TEMPERATURE,
    judge_row,
)
from parsing import is_correct, rule_parse  # noqa: E402
from prompting import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    PROMPT_VARIANTS,
    build_content,
    parse_options,
)


# ============================================================
# 1. 고정값 (assignment_guidance.md §1.1, §1.2)
# ============================================================

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

DATASET_REPO = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SPLIT = "validation"
SAMPLES_PER_SUBJECT = 30

JUDGE_REPO = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

# 출처: MMMU-Benchmark/MMMU mmmu/utils/data_utils.py DOMAIN_CAT2SUB_CAT
CATEGORIES = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


# ============================================================
# 2. Sampling recipe (assignment_guidance.md §1.3: 공식 recipe를 찾아서 쓰고 출처 명시)
# 출처: QwenLM/Qwen3-VL evaluation/mmmu/infer_instruct.sh 및 README "Instruct Models"
#       (temperature/top_p/top_k/repetition_penalty/presence_penalty/max_new_tokens)
#       seed=42: 같은 저장소 evaluation/mmmu/run_mmmu.py의 LLM(..., seed=42)
# ============================================================

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.0
PRESENCE_PENALTY = 1.5
SEED = 42
DEFAULT_MAX_NEW_TOKENS = 32768


# ============================================================
# 3. 데이터
# ============================================================

def load_subject(subject, data_root):
    """assignment_guidance.md §1.2: 과목마다 별도 config로, revision을 pin해서 로드한다."""

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_REPO,
        subject,
        split=SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(data_root),
    )

    if len(dataset) != SAMPLES_PER_SUBJECT:
        raise RuntimeError(f"{subject}: expected {SAMPLES_PER_SUBJECT} samples, got {len(dataset)}")

    return dataset


def select_subjects(args):
    return SUBJECTS if args.all else [args.subject]


# ============================================================
# 4. JSONL 입출력 (resume용)
# ============================================================

def read_jsonl(path):

    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def latest_by_id(rows, ok=lambda r: True):
    """같은 id가 여러 번 기록돼 있으면(resume 재시도) 성공한 마지막 행을 쓴다."""

    result = {}

    for row in rows:

        if ok(row) or row["id"] not in result:
            result[row["id"]] = row

    return result


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# 5. 서버 호출
# ============================================================

_local = threading.local()


def session():

    if not hasattr(_local, "session"):
        _local.session = requests.Session()

    return _local.session


def check_server(api_base):

    try:
        response = requests.get(f"{api_base}/models", timeout=10)
        response.raise_for_status()

    except Exception as e:
        raise RuntimeError(
            f"vLLM server is not reachable at {api_base}. Start the server first "
            "(scripts/run_mmmu_eval.sh starts it automatically)."
        ) from e


def call_model(content, args):

    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "seed": SEED,
        "max_tokens": args.max_new_tokens,
        "mm_processor_kwargs": {
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
    }

    # 32768 토큰 응답은 부하가 걸리면 10분을 넘길 수 있다 (원본 timeout=600에서 늘림)
    response = session().post(
        f"{args.api_base}/chat/completions",
        json=payload,
        timeout=args.request_timeout,
    )
    response.raise_for_status()

    result = response.json()
    choice = result["choices"][0]

    return choice["message"]["content"] or "", choice.get("finish_reason"), result.get("usage", {})


# ============================================================
# 6. infer
# ============================================================

def infer_one(subject, dataset, index, args):

    sample = dataset[index]

    t_start = time.time()

    try:
        content, options = build_content(sample, args.prompt)
        raw_output, finish_reason, usage = call_model(content, args)
        error = None

    except Exception as e:
        options = parse_options(sample["options"])
        raw_output, finish_reason, usage, error = "", "error", {}, repr(e)

    t_end = time.time()

    return {
        "id": sample["id"],
        "subject": subject,
        "question_type": sample["question_type"],
        "question": sample["question"],
        "options": options,
        "answer": sample["answer"],
        "prompt_variant": args.prompt,
        "raw_output": raw_output,
        "finish_reason": finish_reason,
        "usage": usage,
        "error": error,
        "t_start": t_start,
        "t_end": t_end,
        "elapsed_sec": round(t_end - t_start, 3),
    }


def cmd_infer(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.jsonl"

    existing = read_jsonl(predictions_file)

    other_variants = {r.get("prompt_variant") for r in existing} - {args.prompt}

    if other_variants:
        raise RuntimeError(
            f"{predictions_file} already has prompt_variant={other_variants}. "
            "Use a different --output-dir for a different prompt."
        )

    done = {r["id"] for r in existing if r["finish_reason"] != "error"}

    check_server(args.api_base)

    tasks = []

    for subject in select_subjects(args):

        dataset = load_subject(subject, args.data_root)

        count = SAMPLES_PER_SUBJECT if args.limit is None else min(args.limit, SAMPLES_PER_SUBJECT)

        for index in range(count):

            if dataset[index]["id"] not in done:
                tasks.append((subject, dataset, index))

    print(f"[infer] prompt={args.prompt} max_new_tokens={args.max_new_tokens} "
          f"concurrency={args.concurrency} | done={len(done)} todo={len(tasks)}")

    session_start = time.time()
    started_at = now_iso()
    finished = 0

    with predictions_file.open("a", encoding="utf-8") as fp, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:

        futures = [
            pool.submit(infer_one, subject, dataset, index, args)
            for subject, dataset, index in tasks
        ]

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

    session_record = {
        "stage": "infer",
        "started_at": started_at,
        "wall_sec": round(time.time() - session_start, 3),
        "requests": len(tasks),
        "prompt_variant": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "served_model_name": args.served_model_name,
    }

    append_session(output_dir, session_record)

    errors = sum(
        1 for r in latest_by_id(read_jsonl(predictions_file), ok=lambda r: r["finish_reason"] != "error").values()
        if r["finish_reason"] == "error"
    )

    if errors:
        print(f"⚠️  {errors} requests failed. Re-run the same command to retry only those (resume).")


def append_session(output_dir, record):

    sessions_file = output_dir / "sessions.json"

    sessions = json.loads(sessions_file.read_text()) if sessions_file.exists() else []
    sessions.append(record)

    sessions_file.write_text(json.dumps(sessions, ensure_ascii=False, indent=2))


# ============================================================
# 7. judge
# ============================================================

def cmd_judge(args):

    output_dir = Path(args.output_dir)

    predictions = latest_by_id(
        read_jsonl(output_dir / "predictions.jsonl"),
        ok=lambda r: r["finish_reason"] != "error",
    )

    judge_file = output_dir / "judge_results.jsonl"

    judged = latest_by_id(read_jsonl(judge_file), ok=lambda r: r.get("error") is None)
    judged_ok = {i for i, r in judged.items() if r.get("error") is None}

    targets = [
        row for row in predictions.values()
        if row["finish_reason"] != "error"
        and rule_parse(row) is None
        and row["id"] not in judged_ok
    ]

    print(f"[judge] model={args.judge_model} targets={len(targets)} (already judged={len(judged_ok)})")

    if not targets:
        return

    check_server(args.judge_api_base)

    session_start = time.time()
    started_at = now_iso()

    def run(row):

        try:
            result = judge_row(row, args.judge_api_base, args.judge_model)
            result["error"] = None

        except Exception as e:
            result = {"id": row["id"], "judge_model": args.judge_model, "judge_pred": None, "error": repr(e)}

        return result

    with judge_file.open("a", encoding="utf-8") as fp, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:

        for i, future in enumerate(as_completed([pool.submit(run, row) for row in targets]), 1):

            result = future.result()

            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            fp.flush()

            print(f"[{i}/{len(targets)}] {result['id']} judge_pred={result.get('judge_pred')}"
                  + (f" error={result['error']}" if result["error"] else ""), flush=True)

    append_session(output_dir, {
        "stage": "judge",
        "started_at": started_at,
        "wall_sec": round(time.time() - session_start, 3),
        "requests": len(targets),
        "judge_model": args.judge_model,
        "judge_revision": args.judge_revision,
    })


# ============================================================
# 8. score
# ============================================================

def read_peak_gpu_mem_mib(output_dir):
    """run_mmmu_eval.sh가 nvidia-smi로 기록한 gpu_mem_<stage>.log의 최댓값(MiB)."""

    peaks = {}

    for log in output_dir.glob("gpu_mem_*.log"):

        values = []

        for line in log.read_text().splitlines():

            try:
                values.append(int(line.strip().split()[0]))
            except (ValueError, IndexError):
                pass

        if values:
            peaks[log.stem.replace("gpu_mem_", "")] = max(values)

    return peaks


def cmd_score(args):

    output_dir = Path(args.output_dir)

    predictions = latest_by_id(
        read_jsonl(output_dir / "predictions.jsonl"),
        ok=lambda r: r["finish_reason"] != "error",
    )

    judged = latest_by_id(read_jsonl(output_dir / "judge_results.jsonl"), ok=lambda r: r.get("error") is None)

    subjects = select_subjects(args)

    rows = [r for r in predictions.values() if r["subject"] in subjects]
    rows.sort(key=lambda r: (SUBJECTS.index(r["subject"]), int(r["id"].rsplit("_", 1)[1])))

    final_rows = []

    for row in rows:

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

        final_rows.append({
            **row,
            "rule_pred": rule_pred,
            "rule_correct": is_correct(row, rule_pred),
            "parser_method": method,
            "final_pred": final_pred,
            "correct": is_correct(row, final_pred),
            "judge": judged.get(row["id"]),
        })

    # ---- 과목별 ----

    subject_results = []

    for subject in subjects:

        subject_rows = [r for r in final_rows if r["subject"] == subject]

        if not subject_rows:
            continue

        n = len(subject_rows)

        subject_results.append({
            "subject": subject,
            "samples": n,
            "correct": sum(r["correct"] for r in subject_rows),
            "accuracy": sum(r["correct"] for r in subject_rows) / n,
            "correct_rule_only": sum(r["rule_correct"] for r in subject_rows),
            "accuracy_rule_only": sum(r["rule_correct"] for r in subject_rows) / n,
            # 동시 요청이라 과목별 '벽시계 시간'이 겹친다. 요청 지연의 합과 첫 요청~마지막 응답 구간을 함께 기록.
            "latency_sum_sec": round(sum(r["elapsed_sec"] for r in subject_rows), 3),
            "wall_span_sec": round(max(r["t_end"] for r in subject_rows) - min(r["t_start"] for r in subject_rows), 3),
            "truncated": sum(r["finish_reason"] == "length" for r in subject_rows),
        })

    def macro(key):
        return sum(r[key] for r in subject_results) / len(subject_results) if subject_results else 0.0

    total_samples = sum(r["samples"] for r in subject_results)
    total_correct = sum(r["correct"] for r in subject_results)
    total_correct_rule = sum(r["correct_rule_only"] for r in subject_results)

    macro_accuracy = macro("accuracy")
    micro_accuracy = total_correct / total_samples if total_samples else 0.0

    # ---- 카테고리별 (MMMU 6개 분야) ----

    category_results = {}

    for category, members in CATEGORIES.items():

        rows_in = [r for r in subject_results if r["subject"] in members]

        if rows_in:
            category_results[category] = {
                "samples": sum(r["samples"] for r in rows_in),
                "correct": sum(r["correct"] for r in rows_in),
                "accuracy": sum(r["correct"] for r in rows_in) / sum(r["samples"] for r in rows_in),
            }

    # ---- 전수 검증 (GitHub 원본 검증 유지) ----

    request_errors = sum(r["parser_method"] == "request_error" for r in final_rows)

    if args.all and args.limit is None:

        if len(subject_results) != len(SUBJECTS):
            raise RuntimeError(f"Expected {len(SUBJECTS)} subjects, got {len(subject_results)}. Run infer to completion first.")

        if total_samples != len(SUBJECTS) * SAMPLES_PER_SUBJECT:
            raise RuntimeError(f"Expected {len(SUBJECTS) * SAMPLES_PER_SUBJECT} samples, got {total_samples}.")

        if not math.isclose(macro_accuracy, micro_accuracy, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("Macro and micro accuracy differ.")

    # ---- 통계 ----

    def count(key):
        stats = {}
        for r in final_rows:
            stats[str(r[key])] = stats.get(str(r[key]), 0) + 1
        return stats

    judge_rows = [r["judge"] for r in final_rows if r["judge"]]

    sessions_file = output_dir / "sessions.json"
    sessions = json.loads(sessions_file.read_text()) if sessions_file.exists() else []

    variants = {r["prompt_variant"] for r in final_rows}

    summary = {
        "experiment": "mmmu_val_" + "_".join(sorted(variants)),
        "overall_accuracy": macro_accuracy,
        "overall_definition": "macro average of per-subject accuracy (equals micro when every subject has 30 samples)",
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "rule_only_macro_accuracy": macro("accuracy_rule_only"),
        "subjects": len(subject_results),
        "samples": total_samples,
        "correct": total_correct,
        "correct_rule_only": total_correct_rule,
        "request_errors": request_errors,
        "parse_fallback_rate": sum(r["parser_method"] != "answer_line" for r in final_rows) / len(final_rows) if final_rows else 0.0,
        "truncation_rate": sum(r["finish_reason"] == "length" for r in final_rows) / len(final_rows) if final_rows else 0.0,
        "parser_method_counts": count("parser_method"),
        "finish_reason_counts": count("finish_reason"),
        "judge_input_truncated": sum(bool(j.get("judge_input_truncated")) for j in judge_rows),
        "completion_tokens": {
            "total": sum(r["usage"].get("completion_tokens", 0) for r in final_rows),
            "max": max((r["usage"].get("completion_tokens", 0) for r in final_rows), default=0),
        },
        "prompt_tokens_max": max((r["usage"].get("prompt_tokens", 0) for r in final_rows), default=0),
        "timing": {
            "sessions": sessions,
            "infer_wall_sec_total": round(sum(s["wall_sec"] for s in sessions if s["stage"] == "infer"), 3),
            "judge_wall_sec_total": round(sum(s["wall_sec"] for s in sessions if s["stage"] == "judge"), 3),
        },
        "peak_gpu_mem_mib": read_peak_gpu_mem_mib(output_dir),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "served": sorted({s.get("model_path") for s in sessions if s["stage"] == "infer"} - {None}),
            "served_revision": sorted({s.get("model_revision") for s in sessions if s["stage"] == "infer"} - {None}),
            "dtype": "bfloat16",
        },
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "split": SPLIT},
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "seed": SEED,
            "max_new_tokens": sorted({s["max_new_tokens"] for s in sessions if s["stage"] == "infer"}),
        },
        "image_processing": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "encoding": "JPEG quality=95"},
        "prompt_variant": sorted(variants),
        "parser": {
            "name": "hybrid_answer_line_plus_llm_judge",
            "multiple_choice": ["last 'Answer:' line (MMMU-Pro official step 1)", "LLM judge (VLMEvalKit MCQ prompt)", "wrong"],
            "open": ["last 'Answer:' line + exact/numeric match", "LLM judge extraction + same match", "wrong"],
            "random_fallback": False,
        },
        "judge": {
            "repo": JUDGE_REPO,
            "revision": JUDGE_REVISION,
            "temperature": JUDGE_TEMPERATURE,
            "seed": JUDGE_SEED,
            "max_tokens": JUDGE_MAX_TOKENS,
            "max_input_tokens_of_response": JUDGE_MAX_INPUT_TOKENS,
        },
        "category_results": category_results,
        "subject_results": subject_results,
    }

    # ---- 저장 ----

    with (output_dir / "final_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (output_dir / "failed_parses.jsonl").open("w", encoding="utf-8") as f:
        for r in final_rows:
            if r["parser_method"] != "answer_line":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (output_dir / "subject_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subject_results[0].keys()) if subject_results else ["subject"])
        writer.writeheader()
        writer.writerows(subject_results)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---- 출력 ----

    print("=" * 76)
    print("FINAL RESULT")
    print("=" * 76)
    print(f"samples          : {total_samples}")
    print(f"correct          : {total_correct}")
    print(f"overall (macro)  : {macro_accuracy * 100:.2f}%")
    print(f"overall (micro)  : {micro_accuracy * 100:.2f}%")
    print(f"rule-only (macro): {macro('accuracy_rule_only') * 100:.2f}%")
    print(f"parser methods   : {summary['parser_method_counts']}")
    print(f"finish reasons   : {summary['finish_reason_counts']}")

    if request_errors:
        print(f"⚠️  request errors: {request_errors} (counted as wrong — re-run infer to retry them)")

    print(f"summary          : {output_dir / 'summary.json'}")


# ============================================================
# 9. CLI
# ============================================================

def add_common(parser):

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", choices=SUBJECTS)
    group.add_argument("--all", action="store_true")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="과목당 앞 N문항만 (스모크 테스트용. baseline 수치로 쓸 수 없음)")


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer", help="평가 모델 추론")
    add_common(p_infer)
    p_infer.add_argument("--data-root", required=True, help="MMMU HF 데이터셋 캐시 디렉터리")
    p_infer.add_argument("--prompt", choices=sorted(PROMPT_VARIANTS), default="mmmu_pro_cot")
    p_infer.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p_infer.add_argument("--concurrency", type=int, default=32)
    p_infer.add_argument("--request-timeout", type=int, default=3600)
    p_infer.add_argument("--api-base", default="http://localhost:8000/v1")
    p_infer.add_argument("--served-model-name", default="qwen3-vl-4b")
    p_infer.add_argument("--model-path", default=MODEL_REPO, help="기록용: 서빙 중인 모델 (HF repo 또는 체크포인트 경로)")
    p_infer.add_argument("--model-revision", default=MODEL_REVISION, help="기록용: 서빙 중인 revision (체크포인트면 'local')")
    p_infer.set_defaults(func=cmd_infer)

    p_judge = sub.add_parser("judge", help="rule 파서가 실패한 문항만 LLM judge로 판정")
    add_common(p_judge)
    p_judge.add_argument("--judge-api-base", default="http://localhost:8001/v1")
    p_judge.add_argument("--judge-model", default="judge", help="judge 서버의 served-model-name")
    p_judge.add_argument("--judge-revision", default=JUDGE_REVISION, help="기록용")
    p_judge.add_argument("--concurrency", type=int, default=16)
    p_judge.set_defaults(func=cmd_judge)

    p_score = sub.add_parser("score", help="채점하고 summary.json 생성")
    add_common(p_score)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
