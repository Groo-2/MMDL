"""평가 대기열: 여러 모델을 순서대로 평가한다. 수업 시간에 끊겨도 같은 명령을 다시 실행하면 이어진다.

  python lab/eval_queue.py \\
      base_130=Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17:1 \\
      mx_grpo_team=results/mx_grpo_team/merged:2 \\
      dpo_dv1_re=results/dpo_dv1_re/merged:2

항목 형식은 `태그=모델[@리비전]:run수` 이다. 결과는 eval_repeat.py와 같은 위치(results/<태그>/run_00, run_01 ...)에 쌓인다. run 하나의 seed는 42 + 번호다.
  - run 폴더에 summary.json이 있으면 그 run은 끝난 것이다. 항목의 run이 모두 끝났으면 서버를 띄우지 않고 건너뛴다.
  - 하나라도 남았으면 eval_repeat.py를 `--runs 전체 수`로 부른다. 끝난 run은 eval_repeat가 알아서 건너뛰고 응답 단위로 이어받는다.
  - 평가 하나가 실패하면 대기열을 멈추고 알린다(그 뒤 항목은 실행하지 않는다). 원인을 해결한 뒤 같은 명령을 다시 실행한다.
  - 합친 모델 폴더에 `.merge_ok` 표시가 없으면 경고한다(예전 방식으로 합친 폴더이거나 합치다 끊긴 폴더일 수 있다).
  --dry-run 이면 무엇이 남았는지만 출력하고 아무것도 실행하지 않는다.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_spec(spec):
    """`태그=모델[@리비전]:run수` → (태그, 모델, 리비전 또는 None, run 수)."""
    if "=" not in spec:
        raise ValueError(f"형식이 틀렸다(태그=모델[@리비전]:run수): {spec}")
    tag, rest = spec.split("=", 1)
    runs = 1
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if tail.isdigit():
            rest, runs = head, int(tail)
    model, revision = rest, None
    if "@" in rest:
        model, revision = rest.rsplit("@", 1)
    if not tag or not model or runs < 1:
        raise ValueError(f"형식이 틀렸다(태그=모델[@리비전]:run수): {spec}")
    return tag, model, revision, runs


def run_status(results_dir, tag, runs):
    """run 번호별로 끝났는지(summary.json) 돌려준다: [(번호, 끝남 여부)]."""
    base = Path(results_dir) / tag
    return [(k, (base / f"run_{k:02d}" / "summary.json").exists()) for k in range(runs)]


def build_command(tag, model, revision, runs, python=sys.executable, repo=REPO):
    cmd = [python, str(Path(repo) / "lab" / "eval_repeat.py"), "--tag", tag, "--model", model, "--runs", str(runs)]
    if revision:
        cmd += ["--revision", revision]
    return cmd


def merge_marker_warning(model, repo=REPO):
    """합친 모델 폴더라면 .merge_ok가 있는지 본다. 경고 문구 또는 None."""
    p = Path(model)
    if not p.is_absolute():
        p = Path(repo) / p
    if not p.is_dir():
        return None
    if (p / ".merge_ok").exists():
        return None
    if any(p.glob("*.safetensors")):
        return f"{model}: .merge_ok 표시가 없다(예전 방식으로 합쳤거나 합치다 끊겼을 수 있다). 의심되면 merge_lora.py를 다시 실행한다."
    return f"{model}: 모델 가중치(*.safetensors)가 없다. 합치기가 끝났는지 확인한다."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+", help="태그=모델[@리비전]:run수 (여러 개)")
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        items = [parse_spec(s) for s in args.specs]
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"[queue] 시작 {time.strftime('%F %T')} | python {sys.version.split()[0]}", flush=True)
    plan = []
    for tag, model, revision, runs in items:
        status = run_status(args.results, tag, runs)
        todo = [k for k, done in status if not done]
        plan.append((tag, model, revision, runs, todo))
        print(f"[queue] {tag}: run {runs}개 중 끝난 것 {runs - len(todo)}개" + (f", 남은 run {todo}" if todo else " → 건너뜀"), flush=True)
        warn = merge_marker_warning(model)
        if warn and todo:
            print(f"[queue] ⚠️ {warn}", flush=True)
    if args.dry_run:
        return 0

    for i, (tag, model, revision, runs, todo) in enumerate(plan, 1):
        if not todo:
            continue
        cmd = build_command(tag, model, revision, runs)
        print(f"\n[queue] ({i}/{len(plan)}) {tag} 시작 {time.strftime('%F %T')}\n[queue] 실행: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"[queue] {tag} 종료 코드 {rc} | {(time.time() - t0) / 3600:.2f}시간 | {time.strftime('%F %T')}", flush=True)
        if rc != 0:
            print(f"[queue] ❌ {tag}에서 실패해서 대기열을 멈춘다. 서버가 남아 있으면 `pkill -f \"vllm serve\"`로 정리하고, 원인을 해결한 뒤 같은 명령을 다시 실행하세요.", flush=True)
            return rc
    print(f"\n[queue] 모두 끝났다 {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
