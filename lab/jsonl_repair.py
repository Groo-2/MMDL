"""JSONL 이어쓰기 전에 마지막 줄이 깨졌으면 바로잡는다.

왜 필요한가: 평가(추론·judge)와 선별은 응답 하나가 끝날 때마다 jsonl에 한 줄씩 덧붙이고(flush), 다시 실행하면 그 파일을 읽어서
이미 끝난 문항을 건너뛴다. 그런데 수업 시간에 원격이 꺼지거나 프로세스가 강제로 죽으면 **줄을 쓰던 중간에 끊겨** 마지막 줄이 반쪽이 된다.
이 상태에서 다시 실행하면 (1) 읽을 때 json.loads가 예외를 내서 멈추고, (2) 설령 건너뛰어도 다음 줄이 반쪽 뒤에 이어 붙어서 두 레코드가 함께 깨진다.

repair_jsonl은 마지막 줄만 본다.
  - 줄바꿈이 없는데 내용이 온전한 JSON이면 줄바꿈만 붙인다 (기록은 보존).
  - 마지막 줄이 JSON으로 읽히지 않으면(반쪽이거나 0으로 채워짐) 그 줄만 잘라 낸다. 그 문항은 아직 안 끝난 것이라 다시 풀린다.
  - 그 앞의 줄은 건드리지 않는다.

  python lab/jsonl_repair.py results/dpo_dv1_re/full data/musixqa      # 폴더 아래의 알려진 jsonl을 모두 검사
"""

import json
import sys
from pathlib import Path

# 이어쓰기로 쓰는 파일 이름들 (평가: 예측·judge, 선별: 통과율·응답)
KNOWN_NAMES = ("predictions.jsonl", "judge_results.jsonl", "screen_results.jsonl", "screen_samples.jsonl")


def repair_jsonl(path):
    """마지막 줄이 깨졌으면 바로잡고, 잘라 낸 바이트 수를 돌려준다 (0이면 손댈 것이 없었다)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    data = p.read_bytes()
    if data.endswith(b"\n"):
        body = data[:-1]
        start = body.rfind(b"\n") + 1
        last = body[start:]
        end_with_newline = True
    else:
        start = data.rfind(b"\n") + 1
        last = data[start:]
        end_with_newline = False
    try:
        json.loads(last.decode("utf-8"))
        ok = True
    except Exception:  # noqa: BLE001  (UnicodeDecodeError, JSONDecodeError 모두)
        ok = False
    if ok:
        if not end_with_newline:
            with p.open("ab") as f:
                f.write(b"\n")
        return 0
    with p.open("r+b") as f:
        f.truncate(start)
    return len(data) - start


def repair_tree(root, names=KNOWN_NAMES):
    """root 아래의 알려진 jsonl을 모두 검사한다. {경로: 잘라 낸 바이트}에서 0이 아닌 것만 돌려준다."""
    fixed = {}
    root = Path(root)
    if root.is_file():
        candidates = [root]
    else:
        candidates = [p for n in names for p in root.rglob(n)] if root.exists() else []
    for p in sorted(candidates):
        dropped = repair_jsonl(p)
        if dropped:
            fixed[str(p)] = dropped
    return fixed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    total = {}
    for a in argv:
        total.update(repair_tree(a))
    if total:
        for p, n in total.items():
            print(f"반쪽 마지막 줄을 잘라 냈다 ({n}바이트): {p}")
    else:
        print("깨진 마지막 줄이 없다.")


if __name__ == "__main__":
    main()
