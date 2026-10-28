"""실습실(RTX 4090 24GB)용 MMMU validation 반복 평가. 같은 모델을 seed만 바꿔 N번 평가한다 (Avg@N).

평가 코드는 mmmu_eval/을 수정 없이 불러온다. 프롬프트, sampling(temperature/top_p/top_k/presence_penalty),
파서, judge, max_new_tokens(32768), 해상도는 baseline(Run B)과 같다. 바뀌는 것은 요청에 실리는 seed뿐이다.
  run 0 = seed 42 (동결 recipe 그대로), run k = seed 42+k

24GB에는 Qwen(약 9GB)과 Llama judge(약 16GB)를 동시에 올릴 수 없어서 순차로 실행한다.
  A) Qwen 서버를 한 번 띄워 모든 run을 추론 → 서버 종료
  B) Llama 서버를 한 번 띄워 rule 파서가 실패한 문항만 모든 run에서 판정 → 서버 종료
  C) 채점 → 요약
중간에 끊겨도 같은 커맨드를 다시 실행하면 끝난 문항은 건너뛴다.

  python lab/eval_repeat.py --tag baseline --model Qwen/Qwen3-VL-4B-Instruct \\
      --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --runs 8
  python lab/eval_repeat.py --tag grpo_run1 --model results/grpo_run1/merged --preset music4
  python lab/eval_repeat.py --tag smoke --model ... --runs 1 --subject Music    # 시간 측정용

프리셋 (결과는 results/<tag>/<music|full>/run_NN)
  music4        Music 30문항 x seed 42~45
  full2         전체 900문항 x seed 42, 43
  music_extend  Music에 seed 46~49를 추가해 8번으로 확장. 사용자 확인 없이는 실행되지 않는다:
                환경변수 CONFIRM_EXTEND_TO_8=yes 를 사용자가 직접 설정해야 한다 (music4를 먼저 끝내야 함)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for cand in (REPO / "baseline" / "mmmu_eval", REPO / "mmmu_eval"):
    if cand.is_dir():
        sys.path.insert(0, str(cand))
        break
else:
    raise SystemExit("mmmu_eval 폴더를 찾지 못했다")

import requests  # noqa: E402
import run_mmmu_eval as R  # noqa: E402  (평가 코드 그대로 사용)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_runs  # noqa: E402
from envinfo import load_env_file, snapshot  # noqa: E402

JUDGE = R.JUDGE_REPO
JUDGE_REV = R.JUDGE_REVISION
QWEN_PORT, JUDGE_PORT = 8000, 8001
# 24GB: 40960 = 32768 출력 + 최대 프롬프트 여유. 128000(Qwen 공식)은 KV cache가 모자라 서버가 뜨지 않는다.
MAX_MODEL_LEN = 40960
JUDGE_MAX_MODEL_LEN = 20480
GPU_MEM_UTIL = 0.90


# ============================================================
# vLLM 서버 도우미 (experiments/music_resolution_4090.ipynb와 같은 방식)
# ============================================================

def kill_leftover_servers():
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(10)


def show_log_errors(log_path, tail=60):
    lines = Path(log_path).read_text(errors="ignore").splitlines()
    keys = ("error", "out of memory", "cuda", "kv cache", "max seq len", "free memory")
    hits = [l for l in lines if any(k in l.lower() for k in keys)]
    print("--- 오류로 보이는 줄 ---")
    print("\n".join(hits[-30:]) or "(없음)")
    print(f"--- 로그 마지막 {tail}줄 ---")
    print("\n".join(lines[-tail:]))


def server_env():
    """vLLM 서버 환경변수. FlashInfer 샘플러는 실행 중 컴파일(nvcc)이 필요해서 기본으로 끈다
    (이 PC의 2026-10-20 pilot20 평가도 VLLM_USE_FLASHINFER_SAMPLER=0으로 실행했다). 이미 설정돼 있으면 그 값을 쓴다."""
    return {**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0")}


def start_server(cmd, port, log_path):
    kill_leftover_servers()
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=server_env())
    for _ in range(720):  # 최대 60분 (첫 실행은 모델 다운로드 포함)
        try:
            if requests.get(f"http://localhost:{port}/v1/models", timeout=5).ok:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            show_log_errors(log_path)
            raise RuntimeError(f"vLLM 서버가 종료됐습니다. 전체 로그: {log_path}")
        time.sleep(5)
    stop_server(proc)
    raise RuntimeError(f"vLLM 서버가 제시간에 준비되지 않았습니다. 로그: {log_path}")


def stop_server(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=120)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    time.sleep(10)  # GPU 메모리 해제 대기


# ============================================================

def scope(args):
    """run_mmmu_eval의 --subject/--all 인자를 Namespace 필드로 옮긴다."""
    if args.subject:
        return dict(subject=args.subject, all=False, limit=args.limit)
    return dict(subject=None, all=True, limit=args.limit)


# (과목, run 수, 시작 run 번호, 결과 하위 폴더)
PRESETS = {
    "music4": dict(subject="Music", runs=4, run_start=0, group="music"),
    "full2": dict(subject=None, runs=2, run_start=0, group="full"),
    "music_extend": dict(subject="Music", runs=4, run_start=4, group="music"),
}
EXTEND_MESSAGE = (
    "music_extend(Music을 8번으로 확장)는 사용자 확인 없이 실행하지 않는다.\n"
    "  music4 결과를 확인한 뒤 확장하기로 했다면 직접 설정하고 다시 실행하세요:\n"
    "    CONFIRM_EXTEND_TO_8=yes bash lab/run_pipeline.sh music_extend"
)


def resolve_runs(args):
    """프리셋/인자를 해석해 (출력 폴더, [(run 번호, seed, 폴더)])를 돌려준다. GPU나 서버 없이 호출할 수 있다."""
    out_root = Path(args.out_root) / args.tag
    if args.preset:
        p = PRESETS[args.preset]
        args.subject, args.runs, args.run_start = p["subject"], p["runs"], p["run_start"]
        out_root = out_root / p["group"]
    runs = [(k, args.seed0 + k, out_root / f"run_{k:02d}") for k in range(args.run_start, args.run_start + args.runs)]
    return out_root, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="결과 폴더 이름 (results/<tag>/...)")
    parser.add_argument("--model", required=True, help="HF repo 또는 merge한 체크포인트 폴더")
    parser.add_argument("--revision", default=None, help="HF repo일 때만. 체크포인트 폴더는 생략")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--run-start", type=int, default=0, help="run 번호의 시작 (같은 폴더에 이어 붙일 때)")
    parser.add_argument("--seed0", type=int, default=R.SEED, help="run 0의 seed (동결 recipe = 42). run k의 seed = seed0 + k")
    parser.add_argument("--subject", choices=R.SUBJECTS, default=None, help="한 과목만 (생략하면 30과목 900문항)")
    parser.add_argument("--limit", type=int, default=None, help="과목당 앞 N문항 (스모크 테스트용. 보고용 수치로 쓸 수 없음)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--no-judge", action="store_true", help="judge 생략 (rule 파서 실패 문항은 오답 처리)")
    parser.add_argument("--data-root", default=str(REPO / "data" / "hf_mmmu"))
    parser.add_argument("--out-root", default=str(REPO / "results"))
    parser.add_argument("--gpu-mem-util", type=float, default=GPU_MEM_UTIL)
    parser.add_argument("--dry-run", action="store_true", help="서버를 띄우지 않고 실행 계획(폴더, seed)만 출력")
    args = parser.parse_args()

    # 8번 확장 안전장치: 서버를 띄우거나 무엇을 하기 전에 가장 먼저 확인한다.
    if args.preset == "music_extend" and os.environ.get("CONFIRM_EXTEND_TO_8") != "yes":
        print(EXTEND_MESSAGE)
        sys.exit(3)

    out_root, runs = resolve_runs(args)

    if args.preset == "music_extend":
        missing = [k for k in range(4) if not (out_root / f"run_{k:02d}" / "final_predictions.jsonl").exists()]
        if missing:
            raise SystemExit(f"music4가 끝나지 않았다 (없는 run: {missing}). 먼저 music4를 완료하세요: {out_root}")

    print(f"계획: {args.tag} | preset={args.preset} | 과목={args.subject or '전체 30과목'} | "
          f"run {[k for k, _, _ in runs]} seed {[s for _, s, _ in runs]} | 결과 {out_root}")
    if args.dry_run:
        return

    load_env_file(REPO / ".env")
    if not args.no_judge and not os.environ.get("HF_TOKEN"):
        raise SystemExit("judge(Llama-3.1-8B-Instruct)는 gated 모델이라 HF_TOKEN이 필요합니다 "
                         "(export HF_TOKEN=... 또는 저장소 루트의 .env). judge 없이 돌리려면 --no-judge")

    out_root.mkdir(parents=True, exist_ok=True)
    sc = scope(args)
    revision_args = ["--revision", args.revision] if args.revision else []

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    snapshot(out_root, extra={"model": args.model, "revision": args.revision, "preset": args.preset,
                              "vllm_use_flashinfer_sampler": server_env()["VLLM_USE_FLASHINFER_SAMPLER"]})
    print(f"GPU: {gpu}\n모델: {args.model} {args.revision or ''}")

    # ---------- A) Qwen 서버 1회, 모든 run 추론 ----------
    t0 = time.time()
    server = start_server(
        ["vllm", "serve", args.model, *revision_args, "--served-model-name", "qwen3-vl-4b", "--dtype", "bfloat16",
         "--seed", "42", "--max-model-len", str(MAX_MODEL_LEN), "--gpu-memory-utilization", str(args.gpu_mem_util),
         "--port", str(QWEN_PORT)],
        QWEN_PORT, out_root / "vllm_infer.log")
    try:
        for k, seed, out in runs:
            out.mkdir(parents=True, exist_ok=True)
            R.SEED = seed  # call_model이 요청마다 이 값을 읽는다
            (out / "run_meta.json").write_text(json.dumps({
                "run": k, "seed": seed, "model": args.model, "revision": args.revision, "gpu": gpu,
                "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": args.gpu_mem_util,
                "subject": args.subject, "limit": args.limit, "preset": args.preset}, indent=2))
            print(f"\n========== [{args.tag}] run {k} seed={seed} ==========", flush=True)
            R.cmd_infer(Namespace(
                output_dir=str(out), data_root=args.data_root, prompt="mmmu_pro_cot", max_new_tokens=32768,
                concurrency=args.concurrency, request_timeout=3600, api_base=f"http://localhost:{QWEN_PORT}/v1",
                served_model_name="qwen3-vl-4b", model_path=args.model,
                model_revision=args.revision or "local", **sc))
    finally:
        stop_server(server)
    print(f"[A] 추론 완료 {(time.time() - t0) / 60:.0f}분")

    # ---------- B) Llama judge 서버 1회, 모든 run 판정 ----------
    if not args.no_judge:
        judge = start_server(
            ["vllm", "serve", JUDGE, "--revision", JUDGE_REV, "--served-model-name", "judge", "--dtype", "bfloat16",
             "--seed", "42", "--max-model-len", str(JUDGE_MAX_MODEL_LEN),
             "--gpu-memory-utilization", str(args.gpu_mem_util), "--port", str(JUDGE_PORT)],
            JUDGE_PORT, out_root / "vllm_judge.log")
        try:
            for k, seed, out in runs:
                R.cmd_judge(Namespace(
                    output_dir=str(out), judge_api_base=f"http://localhost:{JUDGE_PORT}/v1", judge_model="judge",
                    judge_revision=JUDGE_REV, concurrency=16, **sc))
        finally:
            stop_server(judge)

    # ---------- C) 채점 + 요약 ----------
    for k, seed, out in runs:
        R.SEED = seed  # summary.json에 이 run의 seed가 기록되도록
        R.cmd_score(Namespace(output_dir=str(out), **sc))

    summarize_runs.summarize(out_root)


if __name__ == "__main__":
    main()
