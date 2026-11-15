"""진행 상태 한눈에 보기: 선별, 학습, 합치기, 평가가 어디까지 저장돼 있는지.

원격이 끊겼다 돌아왔을 때 "무엇이 끝났고 무엇을 같은 명령으로 다시 실행하면 이어지는지" 확인하는 용도다. 읽기만 하고 아무것도 바꾸지 않는다.

  python lab/status.py                 # 저장소 루트에서
  python lab/status.py --results results --data data
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_rows(path):
    """깨진 줄은 건너뛰며 jsonl을 읽는다 (읽기 전용이라 파일은 고치지 않는다)."""
    rows, bad = [], 0
    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    bad += 1
    except OSError:
        pass
    return rows, bad


def ago(path):
    """파일을 마지막으로 쓴 지 얼마나 됐는지."""
    try:
        sec = time.time() - Path(path).stat().st_mtime
    except OSError:
        return "?"
    if sec < 90:
        return f"{sec:.0f}초 전"
    if sec < 5400:
        return f"{sec / 60:.0f}분 전"
    return f"{sec / 3600:.1f}시간 전"


def screening(data_dir):
    lines = []
    for p in sorted(Path(data_dir).glob("*/screen_results.jsonl")):
        rows, bad = read_rows(p)
        by = defaultdict(Counter)
        for r in rows:
            by[r["category"]][r["k"]] += 1
        samples, bad_s = read_rows(p.parent / "screen_samples.jsonl")
        sample_ids = {r["id"] for r in samples}
        orphan = [r["id"] for r in rows if r["id"] not in sample_ids]
        lines.append(f"  {p.parent.name}: {len(rows)}문항 완료 | 샘플 {len(sample_ids)}문항 | 마지막 저장 {ago(p)}"
                     + (f" | 깨진 줄 {bad + bad_s}" if bad + bad_s else ""))
        for cat, cnt in sorted(by.items()):
            n = sum(cnt.values())
            ok = sum(v for k, v in cnt.items() if 1 <= k <= 7)
            lines.append(f"    {cat:13s} {n:5d}문항 | 쌍 가능(1~7/8) {ok}")
        if orphan:
            lines.append(f"    ⚠️ 통과율만 있고 응답 샘플이 없는 문항 {len(orphan)}개 (쌍을 만들 수 없음)")
    return lines


def training(results_dir):
    lines = []
    for d in sorted(Path(results_dir).iterdir()) if Path(results_dir).exists() else []:
        if not d.is_dir():
            continue
        allcps = sorted((int(m.group(1)), c) for c in d.glob("checkpoint-*") if (m := re.fullmatch(r"checkpoint-(\d+)", c.name)))
        cps = [(n, c) for n, c in allcps if (c / "trainer_state.json").exists()]      # trainer_state.json이 마지막에 쓰이므로 없으면 저장 중에 끊긴 것
        incomplete = [c.name for n, c in allcps if (c / "trainer_state.json").exists() is False] + [c.name for c in d.glob("incomplete-checkpoint-*")]
        has_final = (d / "final_lora" / "adapter_model.safetensors").exists()
        if not allcps and not has_final and not incomplete:
            continue
        msg = f"  {d.name}: "
        if cps:
            step, cp = cps[-1]
            total = None
            try:
                total = json.loads((cp / "trainer_state.json").read_text()).get("max_steps")
            except Exception:  # noqa: BLE001
                pass
            msg += f"최근 체크포인트 step {step}" + (f"/{total}" if total and total > 0 else "") + f" ({ago(cp)})"
        msg += " | final_lora 있음(학습 완료)" if has_final else " | final_lora 없음(학습 중이거나 중단됨 → 같은 명령에 --resume)"
        if incomplete:
            msg += f" | ⚠️ 저장 중 끊긴 체크포인트 {len(incomplete)}개({', '.join(incomplete)}): --resume이 자동으로 건너뜀"
        lines.append(msg)
    return lines


def merging(results_dir):
    lines = []
    for m in sorted(Path(results_dir).glob("*/merged")):
        ok = (m / ".merge_ok").exists()
        has_weights = any(m.glob("*.safetensors"))
        state = "정상(.merge_ok)" if ok else ("가중치는 있으나 표시 없음(예전 방식으로 합쳤거나 도중에 중단됨, 의심되면 다시 합치기)" if has_weights else "비어 있음/중단")
        lines.append(f"  {m.parent.name}/merged: {state}")
    return lines


def evaluating(results_dir):
    lines = []
    base = Path(results_dir)
    for p in sorted(base.rglob("predictions.jsonl")):
        rel = p.relative_to(base)
        if "checkpoint-" in str(rel) or "jpeg" in rel.parts:
            continue
        rows, bad = read_rows(p)
        ok_ids = {r["id"] for r in rows if r.get("finish_reason") != "error"}
        err = sum(1 for r in rows if r.get("finish_reason") == "error")
        d = p.parent
        judge, _ = read_rows(d / "judge_results.jsonl")
        done = (d / "summary.json").exists()
        state = "완료(summary.json 있음)" if done else "진행 중이거나 중단됨 → 같은 평가 명령을 다시 실행하면 이어짐"
        lines.append(f"  {rel.parent}: 예측 {len(ok_ids)}문항" + (f"(오류 {err})" if err else "")
                     + f" | judge {len(judge)} | {state} | 마지막 저장 {ago(p)}" + (f" | 깨진 줄 {bad}" if bad else ""))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--data", default=str(REPO / "data"))
    args = parser.parse_args()
    sections = [("선별 (data/*/screen_results.jsonl)", screening(args.data)),
                ("학습 (results/*/checkpoint-*, final_lora)", training(args.results)),
                ("합치기 (results/*/merged)", merging(args.results)),
                ("평가 (results/**/predictions.jsonl)", evaluating(args.results))]
    for title, lines in sections:
        print(f"[{title}]")
        print("\n".join(lines) if lines else "  (없음)")
        print()
    print("깨진 줄이 있거나 중단된 것이 있으면 같은 명령을 그대로 다시 실행하세요. 이어받기 전에 깨진 마지막 줄은 자동으로 바로잡습니다.")


if __name__ == "__main__":
    main()
